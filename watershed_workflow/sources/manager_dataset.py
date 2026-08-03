"""Base class for managers that provide datasets.

Provides base classes for a Task object, used to store requests for
datasets in a non-blocking way, and for a ManagerDataset, used to do
the fetching.
"""

import abc
import concurrent.futures
import dataclasses
import enum
from typing import Optional, List
import numpy as np
import shapely.geometry
import xarray as xr
import rioxarray  # registers .rio accessor on xr.Dataset and xr.DataArray
import cftime
import geopandas as gpd
import pandas as pd
import logging
import time

from watershed_workflow.crs import CRS
import watershed_workflow.crs
import watershed_workflow.utils.warp
import watershed_workflow.utils.data

from .manager import Manager, ManagerAttributes


class RequestState(enum.Enum):
    """State machine for a dataset request."""
    SERVER_PENDING = 'server_pending'
    SERVER_READY = 'server_ready'
    DOWNLOADING = 'downloading'
    READY = 'ready'


class ManagerDataset(Manager, abc.ABC):
    """Managers that provide xarray Datasets should inherit from this class.

    There are three possible use patterns for this class:

    1. Blocking:

       data = mgr.getDataset(...)

    2. Nonblocking request until the data is needed, then block:

       # establish the request
       request = mgr.requestDataset(...)

       # do other work
       ...

       # really, we need the result now
       data = mgr.fetchDataset(request)

    3. Nonblocking, impatient user checks for data repeatedly (e.g. in a notebook):

       # establish the request
       request = mgr.requestDataset(...)

       # do some work
       ...

       # is it ready?
       mgr.isReady(request)

       # not yet, do more work...
       mgr.isReady(request)

       # it's ready!
       data = mgr.fetchDataset(request)

    Developer notes: derived classes must implement:

    - __init__() : Constructor that creates a ManagerAttributes instance and
      passes it to super().__init__(attrs).

    - Request _requestDataset(Request) : Abstract method that establishes the
      request and returns as quickly as possible. Should set
      request.state = SERVER_PENDING or SERVER_READY if no server job is needed.

    - bool _isServerReady(Request) : Poll the server; trivially return True if
      there is no server waiting step.

    - Request _downloadDataset(Request) : Abstract method that downloads the
      data to disk.

    - Dataset _loadDataset(Request) : Opens files from disk and returns the
      raw dataset.

    """

    @dataclasses.dataclass
    class Request:
        """A PoD "future" identifying the requested data.

        May be derived or just have more PoD attached by other classes.
        """
        # manager that created the request
        manager: object

        # Requested geometry, in the native_crs_in of the manager
        geometry: shapely.geometry.Polygon
        out_crs: CRS | None

        # time range and resampling
        start: cftime._cftime.datetime | None
        end: cftime._cftime.datetime | None
        temporal_resampling: str | None

        # requested variables
        variables: List[str] | None

        # status of the request
        state: RequestState = RequestState.SERVER_PENDING

        # background future, set by requestDataset
        _future: concurrent.futures.Future | None = None

    _poll_interval: int = 120

    def __init__(self, attrs: ManagerAttributes):
        """Initialize dataset manager with a ManagerAttributes instance.

        Parameters
        ----------
        attrs : ManagerAttributes
            Plain-data object holding all metadata for this manager.
        """
        super().__init__(attrs)

        # Detect calendar from native dates
        self.native_calendar = self._detectCalendar()

        # Background executor for _waitAndDownload tasks
        self._executor = concurrent.futures.ThreadPoolExecutor()


    def requestDataset(self,
                       geometry,
                       geometry_crs=None,
                       start=None,
                       end=None,
                       variables=None,
                       out_crs=None,
                       temporal_resampling=None,
                       **kwargs):
        """Establish a request for a dataset for the given geometry and time range.

        Parameters
        ----------
        geometry : shapely.geometry.Polygon | gpd.GeoDataFrame
            Input geometry.
        geometry_crs : CRS, optional
            Coordinate system of geometry (required if geometry is Polygon).
        start : str | int | cftime._cftime.datetime | None, optional
            Start date.
        end : str | int | cftime._cftime.datetime | None, optional
            End date.
        variables : list of str, optional
            Variables to retrieve. For multi-variable datasets, defaults to
            default_variables if None. Ignored for single-variable datasets.
        out_crs : CRS, optional
            If provided, reprojects the data into this CRS.  The spatial
            resampling method is chosen automatically based on the variable
            dtypes: ``'nearest'`` for integer (categorical) variables,
            ``'bilinear'`` for float (continuous) variables.  Override
            ``_spatialResamplingMethod`` on the manager to change this.
        temporal_resampling : str, optional
            Temporal resampling method applied to time-series datasets
            (e.g. ``'monthly'``).  Unrelated to spatial resampling.

        Returns
        -------
        Request
            Request object for this dataset.

        """
        request = self._preprocessParameters(geometry, geometry_crs, start, end, variables, out_crs, temporal_resampling)

        # Start the server-side job
        request = self._requestDataset(request, **kwargs)

        # Submit background polling + download to the thread pool
        request._future = self._executor.submit(self._waitAndDownload, request)

        return request


    def isReady(self, request: Request) -> bool:
        """Check whether the background download is complete.

        Parameters
        ----------
        request : Request
            Request object returned by requestDataset.

        Returns
        -------
        bool
            True if the download is finished (or failed), False otherwise.
        """
        return request._future.done()


    def fetchDataset(self, request: Request) -> xr.Dataset:
        """Block until the request is complete and return the dataset.

        Parameters
        ----------
        request : Request
            Request object returned by requestDataset.

        Returns
        -------
        xr.Dataset
            Dataset for the requested geometry and time range.
        """
        # Blocks until done; re-raises any exception from the background thread
        request._future.result()
        dataset = self._loadDataset(request)
        return self._postprocessDataset(request, dataset)


    def getDataset(self, geometry, geometry_crs=None,
                   start=None, end=None, variables=None,
                   out_crs=None, temporal_resampling=None,
                   **kwargs
                   ):
        """Get dataset for the given geometry and time range.

        Blocking request.

        Parameters
        ----------
        geometry : shapely.geometry.Polygon | gpd.GeoDataFrame
            Input geometry.
        geometry_crs : CRS, optional
            Coordinate system of geometry (required if geometry is Polygon).
        start : str | int | cftime._cftime.datetime | None, optional
            Start date.
        end : str | int | cftime._cftime.datetime | None, optional
            End date.
        variables : list of str, optional
            Variables to retrieve. For multi-variable datasets, defaults to
            default_variables if None. Ignored for single-variable datasets.
        out_crs : CRS, optional
            If provided, reprojects the data into this CRS.  The spatial
            resampling method is chosen automatically from variable dtypes.
        temporal_resampling : str, optional
            Temporal resampling method for time-series datasets.

        Returns
        -------
        xr.Dataset
            Dataset for the requested geometry and time range.
        """
        request = self.requestDataset(geometry, geometry_crs, start, end, variables,
                                      out_crs, temporal_resampling, **kwargs)
        return self.fetchDataset(request)


    #
    # Background thread method
    #
    def _waitAndDownload(self, request: Request) -> None:
        """Poll for server readiness then download; runs in background thread.

        Parameters
        ----------
        request : Request
            Request object to poll and download.
        """
        while not self._isServerReady(request):
            logging.info(f'{self.name}: server not ready, sleeping {self._poll_interval}s...')
            time.sleep(self._poll_interval)
        request.state = RequestState.SERVER_READY
        request.state = RequestState.DOWNLOADING
        self._downloadDataset(request)
        request.state = RequestState.READY


    #
    # Helper functions
    #
    def _detectCalendar(self) -> str:
        """Detect the calendar type from native start/end dates.

        Returns
        -------
        str
            Calendar type ('noleap', 'standard', etc.) or 'standard' as default.
        """
        for date in [self.native_start, self.native_end]:
            if date is not None and hasattr(date, 'calendar'):
                return date.calendar
        return 'standard'

    def _parseDate(self, date: str | int | cftime._cftime.datetime | None, is_start: bool) -> cftime._cftime.datetime | None:
        """Parse date input into cftime datetime.

        Parameters
        ----------
        date : str | int | cftime._cftime.datetime | None
            Date to parse.
        is_start : bool
            True if this is a start date (uses Jan 1 for int years).
            False if this is an end date (uses Dec 31 for int years).

        Returns
        -------
        cftime._cftime.datetime or None
            Parsed datetime or None.
        """
        if date is None:
            return self.native_start if is_start else self.native_end
        elif isinstance(date, str):
            return cftime.datetime.strptime(date, "%Y-%m-%d", calendar=self.native_calendar)
        elif isinstance(date, int):
            if is_start:
                return cftime.datetime(date, 1, 1, calendar=self.native_calendar)
            else:
                return cftime.datetime(date, 12, 31, calendar=self.native_calendar)
        elif isinstance(date, cftime._cftime.datetime):
            if date.calendar != self.native_calendar:
                var = 'start' if is_start else 'end'
                raise ValueError(f'Invalid value passed for {var}: expected date for calendar '
                                 f'"{self.native_calendar}" but got date in calendar "{date.calendar}".')
            return date
        else:
            raise TypeError(f"Unsupported date type: {type(date)}")

    def _validateDate(self, date: cftime._cftime.datetime | None, bound: cftime._cftime.datetime | None, is_start: bool) -> None:
        """Validate date against bounds.

        Parameters
        ----------
        date : cftime._cftime.datetime or None
            Date to validate.
        bound : cftime._cftime.datetime or None
            Bound to check against.
        is_start : bool
            True if validating start date, False for end date.
        """
        if date is not None and bound is not None:
            if is_start and date < bound:
                raise ValueError(f"Start date {date} is before dataset start {bound}")
            elif not is_start and date > bound:
                raise ValueError(f"End date {date} is after dataset end {bound}")

    def _validateVariables(self, variables: str | List[str] | None):
        """Validate and normalize requested variables."""
        if variables is None:
            return self.default_variables

        if self.valid_variables is None:
            raise ValueError("This dataset does not support variable selection")

        if isinstance(variables, str):
            variables = [variables,]

        for var in variables:
            if var not in self.valid_variables:
                raise ValueError(f"Invalid variable '{var}'. Valid variables: {', '.join(self.valid_variables)}")
        return variables


    #
    # Dataset methods -- can be overridden but probably don't need to be
    #
    def _preprocessParameters(self,
                              geometry: shapely.geometry.Polygon | gpd.GeoDataFrame,
                              geometry_crs: CRS | None,
                              start: str | int | cftime._cftime.datetime | None,
                              end: str | int | cftime._cftime.datetime | None,
                              variables: List[str] | None,
                              out_crs: CRS | None,
                              temporal_resampling: str | None
                              ) -> Request:
        """Process and validate parameters for getDataset calls.

        Parameters
        ----------
        geometry : shapely.geometry.Polygon | gpd.GeoDataFrame
            Input geometry.
        geometry_crs : CRS, optional
            Coordinate system of geometry (required if geometry is Polygon).
        start : str | int | cftime._cftime.datetime | None, optional
            Start date.
        end : str | int | cftime._cftime.datetime | None, optional
            End date.
        variables : list of str or None, optional
            Variables to retrieve.
        out_crs : CRS or None
            Output coordinate reference system, or None.
        temporal_resampling : str or None
            Temporal resampling method for time-series data, or None.

        Returns
        -------
        Request
            The "future" object storing the metadata for the data request.

        """
        # Process geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry_crs is not None:
                raise ValueError("geometry_crs should not be provided with GeoDataFrame")
            geometry_crs = geometry.crs
            polygon = geometry.union_all()
        elif isinstance(geometry, shapely.geometry.base.BaseGeometry):
            if geometry_crs is None:
                raise ValueError("geometry_crs is required when geometry is a Polygon")
            polygon = geometry
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")

        # Transform to native input CRS and buffer
        polygon = watershed_workflow.utils.warp.warpShply(polygon, geometry_crs, self.native_crs_in)
        logging.info(f'Incoming shape area = {polygon.area}')
        logging.info(f'... buffering incoming shape by 3x native resolution = {3 * self.native_resolution}')
        polygon = polygon.buffer(3 * self.native_resolution)
        logging.info(f'... buffered shape area = {polygon.area}')

        # Parse and validate dates, temporal_resampling
        parsed_start = self._parseDate(start, True)
        self._validateDate(parsed_start, self.native_start, True)

        parsed_end = self._parseDate(end, False)
        self._validateDate(parsed_end, self.native_end, False)

        if temporal_resampling is not None and parsed_start is None:
            raise ValueError("Cannot resample non-temporal dataset.")

        if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
            raise ValueError(f"start ({parsed_start}) must not be after end ({parsed_end}).")

        # Parse and validate variables
        parsed_variables = self._validateVariables(variables)

        return self.Request(self, polygon, out_crs, parsed_start, parsed_end, temporal_resampling, parsed_variables)


    def _spatialResamplingMethod(self, dataset: xr.Dataset) -> str:
        """Choose a spatial resampling method appropriate for this dataset.

        The default implementation returns ``'nearest'`` if any data variable
        has an integer dtype (categorical / indicator data), and ``'bilinear'``
        otherwise (continuous float data).  Derived classes may override this
        to force a specific method.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset whose variables are inspected to choose the method.

        Returns
        -------
        str
            A rasterio resampling method name accepted by
            ``watershed_workflow.utils.warp.warpDataset``.
        """
        for var in dataset.data_vars:
            if np.issubdtype(dataset[var].dtype, np.integer):
                return 'nearest'
        return 'bilinear'

    def _postprocessDataset(self,
                            request: Request,
                            dataset: xr.Dataset,
                            ):
        """Clip, time-slice, and (optionally) reproject the dataset.

        Derived classes that need to change dtypes or units should do so
        *before* calling ``super()._postprocessDataset()``, so that the
        spatial resampling method is chosen from the final dtypes.
        """
        # check the CRS out
        assert hasattr(dataset, 'rio')
        if dataset.rio.crs is None:
            dataset = dataset.rio.write_crs(self.native_crs_out)
        else:
            assert watershed_workflow.crs.isEqual(watershed_workflow.crs.from_rasterio(dataset.rio.crs),
                                           self.native_crs_out)

        # Spatial clip to the buffered geometry bounds in native_crs_out.
        clip_bounds = watershed_workflow.utils.warp.warpBounds(
            request.geometry.bounds, self.native_crs_in, self.native_crs_out)
        dataset = dataset.rio.clip_box(*clip_bounds,
                                       crs=watershed_workflow.crs.to_rasterio(self.native_crs_out))


        # Convert and clip all time dimensions.
        # Detect by index type (CFTimeIndex or DatetimeIndex) rather than by
        # name: more robust and handles per-variable dims like 'time_LAI'.
        # Float-encoded time is decoded automatically by xarray on open, so
        # by this point all time dims will have a CFTimeIndex or DatetimeIndex.
        for dim in [d for d in dataset.dims
                    if isinstance(dataset.indexes.get(d),
                                  (xr.CFTimeIndex, pd.DatetimeIndex))]:
            if isinstance(dataset.indexes[dim], pd.DatetimeIndex):
                new_time = watershed_workflow.utils.data.convertTimesToCFTime(dataset[dim].values)
                if self.native_calendar == 'noleap':
                    new_time = watershed_workflow.utils.data.convertTimesToCFTimeNoleap(new_time)
                dataset[dim] = new_time

            dataset = dataset.sel({dim: slice(request.start, request.end)})

        # Reproject to out_crs if requested.
        # The resampling method is chosen based on the (already-converted) dtypes:
        # integer variables use 'nearest'; float variables use 'bilinear'.
        if request.out_crs is not None:
            # Normalise coordinate names to x/y before reprojecting.
            if 'x' in dataset.coords and 'y' in dataset.coords:
                pass
            elif 'lon' in dataset.coords and 'lat' in dataset.coords:
                dataset = dataset.rename({'lon': 'x', 'lat': 'y'})
            elif 'longitude' in dataset.coords and 'latitude' in dataset.coords:
                dataset = dataset.rename({'longitude': 'x', 'latitude': 'y'})

            resampling = self._spatialResamplingMethod(dataset)
            dataset = watershed_workflow.utils.warp.warpDataset(dataset, request.out_crs, resampling)

        # Ensure y coordinate is monotonically increasing (xarray/matplotlib
        # convention: south-first).  Some sources store data north-first
        # (descending y), which causes imshow to render upside-down.
        y_dim = next((d for d in dataset.dims if d in ('y', 'lat', 'latitude')), None)
        if y_dim is not None:
            y_vals = dataset[y_dim].values
            if len(y_vals) > 1 and y_vals[0] > y_vals[-1]:
                dataset = dataset.isel({y_dim: slice(None, None, -1)})

        # Add product and source to dataset attributes
        dataset.attrs['product'] = self.product
        dataset.attrs['source'] = self.source

        return dataset


    #
    # abstract methods that must be provided by derived datasets
    # ------------------------------------------------------------------
    # Hooks to be overridden by deriving classes.
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _requestDataset(self, request: Request) -> Request:
        """Submit server job and return quickly.

        If the source has no asynchronous server step, this method may return
        immediately. State transitions are managed entirely by the base class.

        Parameters
        ----------
        request : Request
            Dataset request storing the input parameters.

        Returns
        -------
        Request
            Dataset request storing any extra needed data to identify the request.
        """
        pass


    @abc.abstractmethod
    def _isServerReady(self, request: Request) -> bool:
        """Poll the server to check whether the submitted job is complete.

        For sources with no asynchronous server step, simply return True.

        Parameters
        ----------
        request : Request
            Dataset request to poll.

        Returns
        -------
        bool
            True if the server-side job is complete and data can be downloaded.
        """
        pass


    @abc.abstractmethod
    def _downloadDataset(self, request: Request) -> None:
        """Download data files to disk.

        Parameters
        ----------
        request : Request
            Dataset request storing the input parameters.
        """
        pass


    @abc.abstractmethod
    def _loadDataset(self, request: Request) -> xr.Dataset:
        """Open files from disk and return the raw dataset.

        Parameters
        ----------
        request : Request
            Dataset request populated by _downloadDataset.

        Returns
        -------
        xr.Dataset
            The raw dataset before postprocessing.
        """
        pass
