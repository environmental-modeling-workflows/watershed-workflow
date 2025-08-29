"""Base class for managers that provide datasets.

Provides base classes for a Task object, used to store requests for
datasets in a non-blocking way, and for a ManagerDataset, used to do
the fetching.

"""

import abc
from typing import Optional, overload, List
import shapely.geometry
import xarray as xr
import cftime
import geopandas as gpd
import attr
import logging
import time

from watershed_workflow.crs import CRS
import watershed_workflow.crs
import watershed_workflow.warp
import watershed_workflow.data

class ManagerDataset(abc.ABC):
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
       data = mgr.waitForDataset(request)

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
       data = mgr.fetchRequest(request)
    
    Developer notes: derived classes must implement:

    - __init__() : Constructor that supplies native data properties as
      parameters by calling super().__init__()

    - Request _requestDataset(Request) : Abstract method that establishes the request

    - bool isReady(Request) (optional, if not ready immediately)

    - Dataset _fetchDataset(request) : Abstract method that fetches the data.

    """

    @attr.define
    class Request:
        """A PoD class for storing requested info -- intended to be
        derived, adding in manager-specific references for the data.
        """
        manager : 'ManagerDataset'
        is_ready : bool

        # parameters to the original request
        geometry: shapely.geometry.Polygon
        start: cftime._cftime.datetime
        end: cftime._cftime.datetime
        variables: List
        out_crs : Optional[CRS] = None
        resampling : Optional[str] = None

        def copyFromExisting(self, other):
            self.manager = other.manager
            self.is_ready = other.is_ready
            self.geometry = other.geometry
            self.start = other.start
            self.end = other.end
            self.variables = other.variables
            self.out_crs = other.out_crs
            self.resampling = other.resampling


    def __init__(self, 
                 name: str,
                 source: str,
                 native_resolution: float,
                 native_crs_in: CRS | None,
                 native_crs_out: CRS | None,
                 native_start: cftime._cftime.datetime | None,
                 native_end: cftime._cftime.datetime | None,
                 valid_variables: List[str] | None,
                 default_variables: List[str] | None):
        """Initialize dataset manager with native data properties.
        
        Parameters
        ----------
        name : str
            Name of the dataset manager.
        source : str
            Data source or API used to retrieve the data.
        native_resolution : float
            Inherent resolution of the data in native_crs_in units.
        native_crs_in : CRS
            Expected CRS of the incoming geometry.
        native_crs_out : CRS
            CRS of the data fetched.
        native_start : cftime._cftime.datetime | None
            Earliest start date of the data, None for non-temporal data.
        native_end : cftime._cftime.datetime | None
            Latest end date of the data, None for non-temporal data.
        valid_variables : List[str] | None
            Valid variable names, None for single-variable datasets.
        default_variables : List[str] | None
            Default variables to retrieve, None for single-variable datasets.
        """
        self.name = name
        self.source = source
        self.native_resolution = native_resolution
        self.native_crs_in = native_crs_in
        self.native_crs_out = native_crs_out
        self.native_start = native_start
        self.native_end = native_end
        self.valid_variables = valid_variables
        self.default_variables = default_variables

        # Detect calendar from native dates
        self.native_calendar = self._detectCalendar()


    def requestDataset(self, geometry, geometry_crs=None,
                       start=None, end=None, variables=None,
                       out_crs=None, resampling=None
                       ):
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
        variables : List[str], optional
            Variables to retrieve. For multi-variable datasets, defaults to
            default_variables if None. Ignored for single-variable datasets.
        out_crs : CRS, optional
            If provided, resamples the data into this CRS.
        resampling : str, optional
            If out_crs is provided, this gives the resampling method.
            Note that the right choice here can depend upon the data
            type -- for instance 'nearest' (the default) is good for
            categorical data but 'bilinear' may be better for
            continuous data.  See rasterio.warp.reproject for a list
            of options.

        Returns
        -------
        xr.Dataset
            Dataset for the requested geometry and time range.

        """
        # pre-request allows downloading files, etc
        self._prerequestDataset()

        # Use extracted parameter processing
        logging.info('calling preprocess')
        request = self._preprocessParameters(geometry, geometry_crs, start, end, variables)
        request.out_crs = out_crs
        request.resampling = resampling if resampling is not None else 'nearest'

        # Get dataset
        request = self._requestDataset(request)
        return request

    
    def isReady(self, request : Request) -> bool:
        """Is the request ready for fetching?"""
        return request.is_ready


    def fetchRequest(self, request : Request) -> xr.Dataset:
        """Fetch the request and get the actual data."""
        data = self._fetchDataset(request)
        data = self._postprocessDataset(request, data)
        return data


    def waitForDataset(self,
                       request : Request,
                       interval: int = 120,
                       tries: int = 100) -> xr.Dataset:
        """Block until task is complete and return Dataset.
        
        Parameters
        ----------
        request : Request
            Request object to wait for.
        interval : int, optional
            Sleep interval in seconds between checks.
        tries : int, optional
            Maximum number of tries before giving up.
            
        Returns
        -------
        xr.Dataset
            Dataset when task is complete.
        """
        count = 0
        is_ready = request.is_ready
        while count < tries and not is_ready:
            is_ready = self.isReady(request)

            if not is_ready:
                logging.info(f'{request.manager.name} request not ready, sleeping {interval}s...')
                time.sleep(interval)
                count += 1

        if not is_ready:
            raise RuntimeError(f'{request.manager.name} request not completing after {tries} tries and {interval*tries} seconds.')

        data = self.fetchRequest(request)
        return data


    def getDataset(self, geometry, geometry_crs=None,
                   start=None, end=None, variables=None,
                   out_crs=None, resampling=None,
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
        variables : List[str], optional
            Variables to retrieve. For multi-variable datasets, defaults to
            default_variables if None. Ignored for single-variable datasets.

        Returns
        -------
        xr.Dataset
            Dataset for the requested geometry and time range.
        """
        request = self.requestDataset(geometry, geometry_crs, start, end, variables, out_crs, resampling)
        data = self.waitForDataset(request)
        return data


    def _prerequestDataset(self) -> None:
        """Managers should overload this method to do stuff prior to processing.

        This can be used to update native variables if they cannot be
        known on construction (e.g. downloaded files).
        """
        pass


    @abc.abstractmethod
    def _requestDataset(self, request : Request) -> Request:
        """Managers should overload this method to request the data.

        Parameters
        ----------
        Request
            Dataset request storing the input parameters.

        Returns
        -------
        Request
            Dataset request storing any extra needed data to identify the request.
        """
        pass


    @abc.abstractmethod
    def _fetchDataset(self, request : Request) -> xr.Dataset:
        """Managers should overload this method to get the data.

        Parameters
        ----------
        Request
            Dataset request storing the input parameters.

        Returns
        -------
        xr.Dataset
            The data requested.
        """
        pass

        
    def _detectCalendar(self) -> str:
        """Detect the calendar type from native start/end dates.
        
        Returns
        -------
        str
            Calendar type ('noleap', 'standard', etc.) or 'standard' as default.
        """
        # Check native_start first, then native_end
        for date in [self.native_start, self.native_end]:
            if date is not None and hasattr(date, 'calendar'):
                return date.calendar
        
        # Default to standard calendar if no dates or no calendar info
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
        cftime._cftime.datetime | None
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
            return date
        else:
            raise TypeError(f"Unsupported date type: {type(date)}")

    def _validateDate(self, date: cftime._cftime.datetime | None, bound: cftime._cftime.datetime | None, is_start: bool) -> None:
        """Validate date against bounds.

        Parameters
        ----------
        date : cftime._cftime.datetime | None
            Date to validate.
        bound : cftime._cftime.datetime | None
            Bound to check against.
        is_start : bool
            True if validating start date, False for end date.
        """
        if date is not None and bound is not None:
            if is_start and date < bound:
                raise ValueError(f"Start date {date} is before dataset start {bound}")
            elif not is_start and date > bound:
                raise ValueError(f"End date {date} is after dataset end {bound}")


    def _preprocessParameters(self, 
                              geometry: shapely.geometry.Polygon | gpd.GeoDataFrame,
                              geometry_crs: CRS | None,
                              start: str | int | cftime._cftime.datetime | None,
                              end: str | int | cftime._cftime.datetime | None,
                              variables: List[str] | None
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
        variables : Optional[List[str]], optional
            Variables to retrieve.
            
        Returns
        -------
        Request
            The "future" object storing the metadata for the data request.
        
        """
        logging.info('prepocess called!')
        # Process geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry_crs is not None:
                raise ValueError("geometry_crs should not be provided with GeoDataFrame")
            geometry_crs = geometry.crs
            polygon = geometry.union_all()
        elif isinstance(geometry, shapely.geometry.Polygon):
            if geometry_crs is None:
                raise ValueError("geometry_crs is required when geometry is a Polygon")
            polygon = geometry
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")

        # Transform to native input CRS and buffer
        polygon = watershed_workflow.warp.shply(polygon, geometry_crs, self.native_crs_in)
        logging.info(f'incoming shape area = {polygon.area}')
        logging.info(f'buffering incoming shape by = {self.native_resolution}')
        polygon = polygon.buffer(self.native_resolution)
        logging.info(f'buffered shape area = {polygon.area}')

        # Parse and validate dates
        parsed_start = self._parseDate(start, True)
        self._validateDate(parsed_start, self.native_start, True)

        parsed_end = self._parseDate(end, False)
        self._validateDate(parsed_end, self.native_end, False)

        # Handle variables
        if variables is None:
            variables = self.default_variables
        elif self.valid_variables is None:
            raise ValueError("This dataset does not support variable selection")
        else:
            # Validate each variable
            for var in variables:
                if var not in self.valid_variables:
                    raise ValueError(f"Invalid variable '{var}'. Valid variables: {', '.join(self.valid_variables)}")

        return ManagerDataset.Request(self, False, polygon, parsed_start, parsed_end, variables)


    def _postprocessDataset(self,
                            request : Request,
                            dataset : xr.Dataset,
                            ):
        """Time dtype converions and clipping, check CRS is applied, and check post conditions."""
        # check the CRS out
        assert hasattr(dataset, 'rio')
        if dataset.rio.crs is None:
            dataset = dataset.rio.write_crs(self.native_crs_out)
        else:
            assert watershed_workflow.crs.isEqual(watershed_workflow.crs.from_rasterio(dataset.rio.crs),
                                           self.native_crs_out)

        # make sure the time dimension is in the right calendar and dtype, and clip if needed
        if 'time' in dataset.dims:
            if not isinstance(dataset['time'][0], cftime._cftime.datetime):
                new_time = watershed_workflow.data.convertTimesToCFTime(dataset['time'].values)
                if self.native_calendar == 'noleap':
                    new_time = watershed_workflow.data.convertTimesToCFTimeNoleap(new_time)
                dataset['time'] = new_time

            # Clip in time if temporal data
            dataset = dataset.sel(time=slice(request.start, request.end))

        # change coordinate system if requested
        if request.out_crs is not None:
            # guess the coordinate names -- they must be in x,y
            if 'x' in dataset.coords and 'y' in dataset.coords:
                pass
            elif 'lon' in dataset.coords and 'lat' in dataset.coords:
                dataset = dataset.rename({'lon' : 'x', 'lat' : 'y'})
            elif 'longitude' in dataset.coords and 'latitude' in dataset.coords:
                dataset = dataset.rename({'longitude' : 'x', 'latitude' : 'y'})
            
            dataset = watershed_workflow.warp.dataset(dataset, request.out_crs, request.resampling)
            
        # Add name and source to dataset attributes
        dataset.attrs['name'] = self.name
        dataset.attrs['source'] = self.source

        return dataset
        
    
        

    
