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

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.crs
from watershed_workflow.crs import CRS
import watershed_workflow.warp
import watershed_workflow.sources.names


def _previous_month():
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 1
    if month == 0:
        year = year - 1
        month = 12
    return cftime.datetime(year, month, 1, calendar='noleap')


class ManagerDaymet:
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

    _START = cftime.datetime(1980, 1, 1, calendar='noleap')
    _END = _previous_month()
    VALID_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    DEFAULT_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']
    URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/daymet_v4_daily_na_{variable}_{year}.nc"

    def __init__(self):
        self.name = 'DayMet 1km'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'daymet',
            'daymet_{var}_{year}_{north}x{west}_{south}x{east}.nc')

    def _download(self, 
                  var : str,
                  year : int,
                  bounds : list[float],
                  force : bool = False) -> str:
        """Download a NetCDF file covering the bounds.

        Parameters
        ----------
        var : str
          Name the variable, see table in the class documentation.
        year : int
          A year in the valid range (currently 1980-2018)
        bounds : [xmin, ymin, xmax, ymax]
          Desired bounds, in the LatLon CRS.
        force : bool, optional
          If true, re-download even if a file already exists.

        Returns
        -------
        filename : str
          Resulting NetCDF file.
        """
        logging.info("Collecting DayMet file to tile bounds: {}".format(bounds))

        # check directory structure
        os.makedirs(self.names.folder_name(), exist_ok=True)

        # get the target filename
        bounds_str = [f"{b:.4f}" for b in bounds]
        filename = self.names.file_name(var=var,
                                        year=year,
                                        north=bounds_str[3],
                                        east=bounds_str[2],
                                        west=bounds_str[0],
                                        south=bounds_str[1])

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

        return filename

    def _cleanDate(self, date : str | cftime.DatetimeNoLeap) -> cftime.DatetimeNoLeap:
        """Returns a string of the format needed for use in the filename and request."""
        if type(date) is str:
            date_split = date.split('-')
            date = cftime.datetime(int(date_split[0]),
                                   int(date_split[1]),
                                   int(date_split[2]),
                                   calendar='noleap')
        if date < self._START:
            raise ValueError(f"Invalid date {date}, must be after {self._START}.")
        if date > self._END:
            raise ValueError(f"Invalid date {date}, must be before {self._END}.")
        return date

    def _cleanBounds(self, 
                     geometry : shapely.geometry.base.BaseGeometry,
                     geometry_crs : CRS,
                     buffer : float) -> list[float]:
        """Compute bounds in the required CRS from a polygon or bounds in a given crs"""
        bounds_ll = watershed_workflow.warp.shply(geometry, geometry_crs,
                                                  watershed_workflow.crs.latlon_crs).bounds
        feather_bounds = list(bounds_ll[:])
        feather_bounds[0] = np.round(feather_bounds[0] - buffer, 4)
        feather_bounds[1] = np.round(feather_bounds[1] - buffer, 4)
        feather_bounds[2] = np.round(feather_bounds[2] + buffer, 4)
        feather_bounds[3] = np.round(feather_bounds[3] + buffer, 4)
        return feather_bounds


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

        new_x = ds.x * 1000
        new_y = ds.y * 1000
        new_time = watershed_workflow.data._convertTimesToCFTimeNoleap(
            watershed_workflow.data._convertTimesToCFTime(ds_combined['time'].values))

        ds_combined = ds_combined.assign_coords(x=new_x, y=new_y, time=new_time)
        ds_combined.x.attrs = attrs_ref
        ds_combined.y.attrs = attrs_ref

        # deal with attrs
        ds_combined.attrs = ds_concat.attrs
        try:
            crs = next(da.rio.crs for da in ds_list_allvars if da.rio.crs is not None)
        except StopIteration:
            crs = watershed_workflow.crs.daymet_crs

        ds_combined.rio.write_crs(crs, inplace=True)
        for var in ds_combined.variables:
            ds_combined[var] = ds_combined[var].rio.write_crs(crs)
        return ds_combined

    def getDataset(self,
                   geometry : shapely.geometry.base.BaseGeometry,
                   geometry_crs : CRS,
                   start : Optional[str | cftime.DatetimeNoLeap] = None,
                   end : Optional[str | cftime.DatetimeNoLeap] = None,
                   variables : Optional[List[str]] = None,
                   force_download : bool = False,
                   buffer : float = 0.01) -> xr.Dataset:
        """Gets file for a single year and single variable.

        Parameters
        ----------
        geometry : shapely.geometry.base.BaseGeometry
            The geometry for which the dataset is to be retrieved. 
        geometry_crs : str, optional
            The coordinate reference system of the geometry. If not provided, it defaults
            to the CRS of the geometry if available, otherwise assumes 'epsg:4326'.
        start : str or datetime.date object, optional
          Date for the beginning of the data, in YYYY-MM-DD. Valid is
          >= 2002-07-01.
        end : str or datetime.date object, optional
          Date for the end of the data, in YYYY-MM-DD. Valid is
          < the current month (DayMet updates monthly.)
        variables : str or list, optional
          Name the variables to download, see class-level
          documentation for choices.  Default is
          [prcp,tmin,tmax,vp,srad].
        force_download : bool
          Download or re-download the file if true.
        buffer : float
          Buffer the bounds by this amount, in degrees. The default is 0.01.

        Returns
        -------
        xarray.Dataset
          Dataset object containing the met data.
        """
        if start is None:
            start = self._START
        start = self._cleanDate(start)
        assert not isinstance(start, str)
        start_year = start.year

        if end is None:
            end = self._END
        end = self._cleanDate(end)
        assert not isinstance(end, str)
        end_year = (end - datetime.timedelta(days=1)).year
        if start_year > end_year:
            raise RuntimeError(
                f"Provided start year {start_year} is after provided end year {end_year}")

        if variables is None:
            variables = self.DEFAULT_VARIABLES
        for var in variables:
            if var not in self.VALID_VARIABLES:
                raise ValueError("DayMet data supports variables: {} (not {})".format(
                    ', '.join(self.VALID_VARIABLES), var))

        # clean bounds
        bounds = self._cleanBounds(geometry, geometry_crs, buffer=buffer)

        # download files
        filenames = []
        for var in variables:
            filename_var = []
            for year in range(start_year, end_year + 1):
                fname = self._download(var, year, bounds, force=force_download)
                filename_var.append(fname)
            filenames.append(filename_var)

        # open files
        ds = self._openFiles(filenames, variables)
        ds_sel = ds.sel(time=slice(start, end))
        ds_sel.attrs = ds.attrs

        ds_sel.rio.write_crs(ds.rio.crs, inplace=True)
        for var in ds_sel.variables:
            ds_sel[var] = ds_sel[var].rio.write_crs(ds.rio.crs)
        return ds_sel
