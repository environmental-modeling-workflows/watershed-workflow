"""Manager for interacting with DayMet datasets."""
from typing import List, Optional

import attr
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
import watershed_workflow.warp
import watershed_workflow.data

from . import utils as source_utils
from . import manager_dataset


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

    @attr.define
    class Request(manager_dataset.ManagerDataset.Request):
        """DayMet-specific request that adds download information."""
        start_year: int = attr.field(default=None)
        end_year: int = attr.field(default=None)

    def __init__(self):
        native_resolution = 1000.0  # 1km in meters
        native_start = cftime.datetime(1980, 1, 1, calendar='noleap')
        _today = datetime.date.today()
        native_end = cftime.datetime(_today.year, _today.month, _today.day, calendar='noleap')
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
            cache_category='meteorology',
            cache_extension='nc',
            has_varname=True,
            short_name='DayMet',
        )

    def _download(self,
                  var: str,
                  year: int,
                  snapped_bounds: tuple,
                  filename: str,
                  force: bool = False) -> None:
        """Download a NetCDF file covering the bounds.

        Parameters
        ----------
        var : str
            Variable name, see class documentation.
        year : int
            A year in the valid range (currently 1980-2023).
        snapped_bounds : tuple of float
            (xmin, ymin, xmax, ymax) in WGS84, from request.snapped_bounds.
        filename : str
            Destination NetCDF file path.
        force : bool, optional
            If true, re-download even if the file already exists.
        """
        logging.info(f"Collecting DayMet file to tile bounds: {snapped_bounds}")

        os.makedirs(self._cacheFolder(), exist_ok=True)

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


    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request) -> Request:
        """Request DayMet data - check if files exist or need downloading.

        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request with preprocessed parameters.

        Returns
        -------
        Request
            DayMet-specific request with year range and readiness status.
        """
        assert request.start is not None, "Start date is required for DayMet data"
        assert request.end is not None, "End date is required for DayMet data"
        assert request.variables is not None, "Variables are required for DayMet data"

        start_year = request.start.year
        end_year = (request.end - datetime.timedelta(days=1)).year
        if start_year > end_year:
            raise RuntimeError(f"Start year {start_year} is after end year {end_year}")

        daymet_request = self.Request(
            manager=request.manager,
            is_ready=True,
            geometry=request.geometry,
            start=request.start,
            end=request.end,
            variables=request.variables,
            snapped_bounds=request.snapped_bounds,
            start_year=start_year,
            end_year=end_year,
        )
        return daymet_request


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
        new_time = watershed_workflow.data.convertTimesToCFTimeNoleap(
            watershed_workflow.data.convertTimesToCFTime(ds_combined['time'].values))

        ds_combined = ds_combined.assign_coords(x=new_x, y=new_y, time=new_time)
        ds_combined.x.attrs = attrs_ref
        ds_combined.y.attrs = attrs_ref
        ds_combined.attrs = ds_concat.attrs

        return ds_combined


    def _fetchDataset(self, request: Request) -> xr.Dataset:
        """Fetch DayMet data for the request.

        Parameters
        ----------
        request : Request
            DayMet-specific request.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested DayMet data.
        """
        variables = request.variables
        start_year = request.start_year
        end_year = request.end_year
        snapped_bounds = request.snapped_bounds
        geometry_bounds = request.geometry.bounds  # buffered, un-snapped — for superset check

        file_lists = []
        for var in variables:
            vfiles = []
            for year in range(start_year, end_year + 1):
                filename = self._cacheFilename(snapped_bounds, var=var,
                                               start_year=year, end_year=year)
                if not os.path.exists(filename):
                    superset = self._checkCache(geometry_bounds, snapped_bounds,
                                                var=var, start_year=year, end_year=year)
                    if superset is not None:
                        logging.info(f'  Using superset cache: {superset}')
                        filename = superset
                if not os.path.exists(filename):
                    self._download(var, year, snapped_bounds, filename)
                vfiles.append(filename)
            file_lists.append(vfiles)

        return self._openFiles(file_lists, variables)
