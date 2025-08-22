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

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.crs
from watershed_workflow.crs import CRS
import watershed_workflow.warp
import watershed_workflow.sources.names
from watershed_workflow.sources.manager_dataset import ManagerDataset


class ManagerDaymet(ManagerDataset):
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
    class Request(ManagerDataset.Request):
        """DayMet-specific request that adds download information."""
        bounds: list = attr.field(default=None)
        start_year: int = attr.field(default=None) 
        end_year: int = attr.field(default=None)

    def __init__(self):
        # DayMet native data properties
        native_resolution = 1000.0  # 1km resolution in meters
        native_start = cftime.datetime(1980, 1, 1, calendar='noleap')
        native_end = cftime.datetime(2023, 12, 31, calendar='noleap')
        native_crs_daymet = CRS.from_proj4(
            '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
        )
        valid_variables = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
        default_variables = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']
        
        # Initialize base class with native properties
        super().__init__(
            name='DayMet 1km',
            source='ORNL DAAC THREDDS API', 
            native_resolution=native_resolution,
            native_crs_in=CRS.from_epsg(4326),      # Accept WGS84 input  
            native_crs_out=native_crs_daymet,       # Return in DayMet LCC projection
            native_start=native_start,
            native_end=native_end,
            valid_variables=valid_variables,
            default_variables=default_variables
        )
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'daymet',
            'daymet_{var}_{year}_{north}x{west}_{south}x{east}.nc')

    def _download(self, 
                  var : str,
                  year : int,
                  bounds_str : List[str],
                  filename : str,
                  force : bool = False) -> None:
        """Download a NetCDF file covering the bounds.

        Parameters
        ----------
        var : str
          Name the variable, see table in the class documentation.
        year : int
          A year in the valid range (currently 1980-2023)
        bounds_str : [xmin, ymin, xmax, ymax]
          Desired bounds, in the LatLon CRS.
        force : bool, optional
          If true, re-download even if a file already exists.
        filename : str
          Resulting NetCDF file.
        """
        logging.info("Collecting DayMet file to tile bounds: {}".format(bounds_str))

        # check directory structure
        os.makedirs(self.names.folder_name(), exist_ok=True)

        if (not os.path.exists(filename)) or force:
            url_dict = { 'year': str(year), 'variable': var }
            url = self.URL.format(**url_dict)
            logging.info("  Downloading: {}".format(url))
            logging.info("      to file: {}".format(filename))

            request_params = [('var', 'lat'), ('var', 'lon'), ('var', var), ('west', bounds_str[0]),
                              ('south', bounds_str[1]), ('east', bounds_str[2]), ('north', bounds_str[3]),
                              ('horizStride', '1'),
                              ('time_start', '{}-01-01T12:00:00Z'.format(year)),
                              ('time_end', '{}-12-31T12:00:00Z'.format(year)), ('timeStride', '1'),
                              ('accept', 'netcdf')]

            r = requests.get(url, params=request_params, verify=source_utils.getVerifyOption())
            r.raise_for_status()

            with open(filename, 'wb') as fid:
                fid.write(r.content)

        else:
            logging.info("  Using existing: {}".format(filename))


    def _cleanBounds(self, geometry : shapely.geometry.Polygon) -> list[float]:
        """Compute bounds in lat-lon CRS for DayMet API requests."""
        # geometry is already in WGS84 (native_crs_in), so just get bounds
        bounds_ll = geometry.bounds
        return [np.round(b, 4) for b in bounds_ll]

    def _requestDataset(self, request: ManagerDataset.Request) -> 'ManagerDaymet.Request':
        """Request DayMet data - check if files exist or need downloading.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request with preprocessed parameters.
            
        Returns
        -------
        ManagerDaymet.Request
            DayMet-specific request with download info and readiness status.
        """
        # Extract parameters from request
        geometry = request.geometry
        start = request.start
        end = request.end
        variables = request.variables
        
        assert start is not None, "Start date is required for DayMet data"
        assert end is not None, "End date is required for DayMet data"
        assert variables is not None, "Variables are required for DayMet data"

        start_year = start.year
        end_year = (end - datetime.timedelta(days=1)).year
        if start_year > end_year:
            raise RuntimeError(f"Start year {start_year} is after end year {end_year}")

        # Get bounds for API requests
        bounds = self._cleanBounds(geometry)
        
        # Create DayMet-specific request with download info
        daymet_request = self.Request(
            manager=request.manager,
            is_ready=True,
            geometry=request.geometry,
            start=request.start,
            end=request.end,
            variables=request.variables,
            bounds=bounds,
            start_year=start_year,
            end_year=end_year,
        )
        
        return daymet_request

    def _openFiles(self,
                   filenames : List[List[str]],
                   variables : List[str]) -> xr.Dataset:
        """Open all files and concatenate them into a single xarray dataset."""

        fnames_by_var = list(zip(filenames, variables))
        ds_list_allvars = []

        for info in fnames_by_var:
            
            var = info[1]
            fnames = info[0]
            
            ds_list = []
            for fname in fnames:
                ds = xr.open_dataset(fname)
                ds_list.append(ds)

            ds_concat = xr.concat(ds_list, dim="time")
            ds_list_allvars.append(ds_concat[var])

        ds_combined = xr.Dataset({da.name: da for i, da in enumerate(ds_list_allvars)})

        # convert the x and y coordinates from km to meters and update the attributes
        attrs_ref = ds_combined.x.attrs 
        attrs_ref['units'] = 'm'

        new_x = ds_combined.x * 1000
        new_y = ds_combined.y * 1000
        new_time = watershed_workflow.data.convertTimesToCFTimeNoleap(
            watershed_workflow.data.convertTimesToCFTime(ds_combined['time'].values))

        ds_combined = ds_combined.assign_coords(x=new_x, y=new_y, time=new_time)
        ds_combined.x.attrs = attrs_ref
        ds_combined.y.attrs = attrs_ref

        # deal with attrs
        ds_combined.attrs = ds_concat.attrs
        
        return ds_combined

    def _fetchDataset(self, request: 'ManagerDaymet.Request') -> xr.Dataset:
        """Fetch DayMet data for the request.
        
        Parameters
        ----------
        request : ManagerDaymet.Request
            DayMet-specific request with preprocessed parameters and download info.
            
        Returns
        -------
        xr.Dataset
            Dataset containing the requested DayMet data.
        """
        # Extract parameters from DayMet request
        variables = request.variables
        bounds = request.bounds
        start_year = request.start_year
        end_year = request.end_year

        # get the target filename
        bounds_str = [f"{b:.4f}" for b in bounds]


        # Download any missing files
        filenames = []
        for var in variables:
            vfiles = []
            for year in range(start_year, end_year + 1):
                filename = self.names.file_name(var=var,
                                                year=year,
                                                north=bounds_str[3],
                                                east=bounds_str[2],
                                                west=bounds_str[0],
                                                south=bounds_str[1])
                vfiles.append(filename)
                if not os.path.exists(filename):
                    # Download the file
                    self._download(var, year, bounds_str, filename)
            filenames.append(vfiles)

        # Open and return dataset (base class handles time clipping and CRS)
        return self._openFiles(filenames, variables)

