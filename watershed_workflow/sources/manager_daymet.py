"""Manager for interacting with DayMet datasets."""
from typing import List, Optional

import os, sys
import logging
import numpy as np
import requests
import requests.exceptions
import shapely.geometry
import cftime, datetime
import xarray as xr

import watershed_workflow.crs
from watershed_workflow.crs import CRS
import watershed_workflow.utils.warp
import watershed_workflow.utils.data

from . import utils as source_utils
from . import manager_dataset
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import CacheInfo, _snapBounds


_CACHE_INFO = CacheInfo(
    category='meteorology',
    subcategory='daymet',
    name='daymet',
    snap_resolution=1000.0,
    is_temporal=True,
)


@cached_dataset_manager(_CACHE_INFO)
class ManagerDaymet(manager_dataset.ManagerDataset):
    """Daymet meterological datasets.

    Daymet is a historic, spatially interpolated product which ingests large
    number of point-sources of meterological data, aggregates them to daily
    time series, and spatially interpolates them onto a 1km gridded product that
    covers all of North America from 1980 to present [Daymet]_.

    Variable names and descriptions

    .. list-table::
        :widths: 25 25 75
        :header-rows: 1

        * - name
          - units
          - description
        * - prcp
          - :math:`mm / day`
          - Total daily precipitation
        * - tmin, tmax
          - :math:`^\\circ C`
          - Min/max daily air temperature
        * - srad
          - :math:`W / m^2`
          - Incoming solar radiation - per DAYLIT time!
        * - vp
          - :math:`Pa`
          - Vapor pressure
        * - swe
          - :math:`Kg / m^2`
          - Snow water equivalent
        * - dayl
          - :math:`s / day`
          - Duration of sunlight

    .. [Daymet] https://daymet.ornl.gov
    """

    # DayMet-specific constants
    URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/daymet_v4_daily_na_{variable}_{year}.nc"

    def __init__(self):
        native_resolution = 1000.0  # 1km in meters
        native_start = cftime.datetime(1980, 1, 1, calendar='noleap')
        native_end = cftime.datetime(2023, 12, 31, calendar='noleap')
        native_crs_daymet = CRS.from_proj4(
            '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
        )
        valid_variables = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
        default_variables = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']

        super().__init__(
            name='DayMet 1km',
            source='ORNL DAAC THREDDS API',
            native_resolution=native_resolution,
            native_crs_in=CRS.from_epsg(4326),
            native_crs_out=native_crs_daymet,
            native_start=native_start,
            native_end=native_end,
            valid_variables=valid_variables,
            default_variables=default_variables,
        )

    def isComplete(self, dir: str, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True if all per-variable, per-year files exist in the cache directory.

        Parameters
        ----------
        dir : str
            Absolute path to a candidate cache directory.
        request : ManagerDataset.Request
            The request being fulfilled.

        Returns
        -------
        bool
            True if all ``{var}_{year}.nc`` files exist for every requested
            variable and year.
        """
        start_year = request.start.year
        end_year = (request.end - datetime.timedelta(days=1)).year
        for var in request.variables:
            for year in range(start_year, end_year + 1):
                if not os.path.isfile(os.path.join(dir, f'{var}_{year}.nc')):
                    return False
        return True

    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request) -> manager_dataset.ManagerDataset.Request:
        """No op -- no server side request"""
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — DayMet THREDDS is synchronous."""
        return True

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Download per-variable, per-year DayMet NetCDF files.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
            Files are written to ``request._download_path/{var}_{year}.nc``.
        """
        start_year = request.start.year
        end_year = (request.end - datetime.timedelta(days=1)).year

        # Snap bounds in WGS84 degrees for use in the THREDDS URL parameters.
        snapped_bounds = _snapBounds(request.geometry.bounds, _CACHE_INFO.snap_resolution)

        for var in request.variables:
            for year in range(start_year, end_year + 1):
                filename = os.path.join(request._download_path, f'{var}_{year}.nc')
                self._downloadFile(var, year, snapped_bounds, filename)

    def _downloadFile(self,
                      var: str,
                      year: int,
                      snapped_bounds: tuple,
                      filename: str,
                      force: bool = False) -> None:
        """Download a single DayMet NetCDF file.

        Parameters
        ----------
        var : str
            Variable name, see class documentation.
        year : int
            A year in the valid range (currently 1980-2023).
        snapped_bounds : tuple of float
            (xmin, ymin, xmax, ymax) in WGS84 degrees.
        filename : str
            Destination NetCDF file path.
        force : bool, optional
            If true, re-download even if the file already exists.
        """
        logging.info(f"Collecting DayMet file to tile bounds: {snapped_bounds}")

        if (not os.path.exists(filename)) or force:
            url_dict = {'year': str(year), 'variable': var}
            url = self.URL.format(**url_dict)
            logging.info(f"  Downloading: {url}")
            logging.info(f"      to file: {filename}")

            xmin, ymin, xmax, ymax = snapped_bounds
            request_params = [
                ('var', 'lat'), ('var', 'lon'), ('var', var),
                ('west',  f'{xmin:.4f}'),
                ('south', f'{ymin:.4f}'),
                ('east',  f'{xmax:.4f}'),
                ('north', f'{ymax:.4f}'),
                ('horizStride', '1'),
                ('time_start', f'{year}-01-01T12:00:00Z'),
                ('time_end',   f'{year}-12-31T12:00:00Z'),
                ('timeStride', '1'),
                ('accept', 'netcdf'),
            ]

            r = requests.get(url, params=request_params, verify=source_utils.getVerifyOption())
            r.raise_for_status()

            with open(filename, 'wb') as fid:
                fid.write(r.content)
        else:
            logging.info(f"  Using existing: {filename}")

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Open all per-variable, per-year files and concatenate into a dataset.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object with ``_download_path`` set.

        Returns
        -------
        xr.Dataset
            Dataset containing all requested DayMet data.
        """
        start_year = request.start.year
        end_year = request.end.year
        variables = request.variables

        file_lists = []
        for var in variables:
            vfiles = [os.path.join(request._download_path, f'{var}_{year}.nc')
                      for year in range(start_year, end_year + 1)]
            file_lists.append(vfiles)

        return self._openFiles(file_lists, variables)

    def _openFiles(self,
                   file_lists: List[List[str]],
                   variables: List[str]) -> xr.Dataset:
        """Open all files and concatenate them into a single xarray dataset."""
        fnames_by_var = list(zip(file_lists, variables))
        ds_list_allvars = []

        for fnames, var in fnames_by_var:
            ds_list = [xr.open_dataset(fname) for fname in fnames]
            ds_concat = xr.concat(ds_list, dim="time")
            ds_list_allvars.append(ds_concat[var])

        ds_combined = xr.Dataset({da.name: da for da in ds_list_allvars})

        # convert x/y coordinates from km to meters
        attrs_ref = ds_combined.x.attrs.copy()
        attrs_ref['units'] = 'm'
        new_x = ds_combined.x * 1000
        new_y = ds_combined.y * 1000
        new_time = watershed_workflow.utils.data.convertTimesToCFTimeNoleap(
            watershed_workflow.utils.data.convertTimesToCFTime(ds_combined['time'].values))

        ds_combined = ds_combined.assign_coords(x=new_x, y=new_y, time=new_time)
        ds_combined.x.attrs = attrs_ref
        ds_combined.y.attrs = attrs_ref
        ds_combined.attrs = ds_concat.attrs

        return ds_combined
