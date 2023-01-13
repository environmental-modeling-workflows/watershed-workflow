"""Manager for interacting with DayMet datasets."""

import os, sys
import logging
import numpy as np
import pyproj
import requests
import requests.exceptions
import shapely.geometry
import datetime
import netCDF4
import rasterio.transform

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.crs
import watershed_workflow.config
import watershed_workflow.warp
import watershed_workflow.sources.names
import watershed_workflow.datasets


def _previous_month():
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 1
    if month == 0:
        year = year - 1
        month = 12
    return datetime.date(year, month, 1)


class FileManagerDaymet:
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

    _START = datetime.date(1980, 1, 1)
    _END = _previous_month()
    VALID_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    DEFAULT_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']
    URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/daymet_v4_daily_na_{variable}_{year}.nc"

    def __init__(self):
        self.name = 'DayMet 1km'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'daymet',
            'daymet_{var}_{year}_{north}x{west}_{south}x{east}.nc')

    def _read_var(self, fname, var):
        with netCDF4.Dataset(fname, 'r') as nc:
            x = nc.variables['x'][:] * 1000.  # km to m; raw netCDF file has km unit
            y = nc.variables['y'][:] * 1000.  # km to m
            time = nc.variables['time'][:]
            assert (len(time) == 365)
            val = nc.variables[var][:]
        return x, y, val

    def _download(self, var, year, bounds, force=False):
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
        bounds = [f"{b:.4f}" for b in bounds]
        filename = self.names.file_name(var=var,
                                        year=year,
                                        north=bounds[3],
                                        east=bounds[2],
                                        west=bounds[0],
                                        south=bounds[1])

        if (not os.path.exists(filename)) or force:
            url_dict = { 'year': str(year), 'variable': var }
            url = self.URL.format(**url_dict)
            logging.info("  Downloading: {}".format(url))
            logging.info("      to file: {}".format(filename))

            request_params = [('var', 'lat'), ('var', 'lon'), ('var', var), ('west', bounds[0]),
                              ('south', bounds[1]), ('east', bounds[2]), ('north', bounds[3]),
                              ('horizStride', '1'),
                              ('time_start', '{}-01-01T12:00:00Z'.format(year)),
                              ('time_end', '{}-12-31T12:00:00Z'.format(year)), ('timeStride', '1'),
                              ('accept', 'netcdf')]

            r = requests.get(url, params=request_params)
            r.raise_for_status()

            with open(filename, 'wb') as fid:
                fid.write(r.content)

        else:
            logging.info("  Using existing: {}".format(filename))

        return filename

    def _clean_date(self, date):
        """Returns a string of the format needed for use in the filename and request."""
        if type(date) is str:
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        if date < self._START:
            raise ValueError(f"Invalid date {date}, must be after {self._START}.")
        if date > self._END:
            raise ValueError(f"Invalid date {date}, must be before {self._END}.")
        return date

    def _clean_bounds(self, polygon_or_bounds, crs):
        """Compute bounds in the required CRS from a polygon or bounds in a given crs"""
        if type(polygon_or_bounds) is dict:
            polygon_or_bounds = watershed_workflow.utils.create_shply(polygon_or_bounds)
        if type(polygon_or_bounds) is shapely.geometry.Polygon:
            bounds_ll = watershed_workflow.warp.shply(polygon_or_bounds, crs,
                                                      watershed_workflow.crs.latlon_crs()).bounds
        else:
            bounds_ll = watershed_workflow.warp.bounds(polygon_or_bounds, crs,
                                                       watershed_workflow.crs.latlon_crs())

        buffer = 0.01
        feather_bounds = list(bounds_ll[:])
        feather_bounds[0] = np.round(feather_bounds[0] - buffer, 4)
        feather_bounds[1] = np.round(feather_bounds[1] - buffer, 4)
        feather_bounds[2] = np.round(feather_bounds[2] + buffer, 4)
        feather_bounds[3] = np.round(feather_bounds[3] + buffer, 4)
        return feather_bounds

    def _open_files(self, filenames, var, start, end):
        """Opens and loads the files, making a single array."""
        # NOTE: this probably needs to be refactored to not load the whole thing into memory?
        nyears = len(filenames)
        data = None
        for i, fname in enumerate(filenames):
            x, y, v = self._read_var(fname, var)  # returned v.shape(nband, nrow, ncol)
            if data is None:
                # note nrows, ncols ordering
                data = np.zeros((nyears * 365, len(y), len(x)), 'd')

            # stuff in the right spot
            data[i * 365:(i+1) * 365, :, :] = v

        # times is a range
        origin = datetime.date(start.year, 1, 1)
        times = np.array([origin + datetime.timedelta(days=i) for i in range(365 * nyears)])

        # trim to start, end
        i_start = (start - origin).days
        i_end = i_start + (end - start).days
        times = times[i_start:i_end]
        data = data[i_start:i_end]

        # profile
        profile = dict()
        profile['crs'] = watershed_workflow.crs.daymet_crs()
        profile['width'] = len(x)
        profile['height'] = len(y)
        profile['count'] = len(times)
        profile['dx'] = (x[1:] - x[:-1]).mean()  # convert to m
        profile['dy'] = (y[1:] - y[:-1]).mean()  # convert to m
        profile['resolution'] = (profile['dx'], -profile['dy'])
        profile['driver'] = 'h5'  # hint that this was not a real raster

        profile['transform'] = rasterio.transform.from_bounds(x[0], y[-1], x[-1], y[0],
                                                              profile['width'], profile['height'])
        profile['nodata'] = -9999
        return watershed_workflow.datasets.Data(profile, times, data)

    def get_data(self,
                 polygon_or_bounds,
                 crs,
                 start=None,
                 end=None,
                 variables=None,
                 force_download=False):
        """Gets file for a single year and single variable.

        Parameters
        ----------
        polygon_or_bounds : fiona or shapely shape, or [xmin, ymin, xmax, ymax]
          Collect a file that covers this shape or bounds.
        crs : CRS object
          Coordinate system of the above polygon_or_bounds
        start : str or datetime.date object, optional
          Date for the beginning of the data, in YYYY-MM-DD. Valid is
          >= 2002-07-01.
        end : str or datetime.date object, optional
          Date for the end of the data, in YYYY-MM-DD. Valid is
          < the current month (DayMet updates monthly.)
        variables : str or list, optional
          Name the variables to download, see table in the class
          documentation for valid.  Default is [prcp,tmin,tmax,vp,srad].
        force_download : bool
          Download or re-download the file if true.

        Returns
        -------
        profile : dict
          Metadata including spatial info, as for a raster.
        times : np.array((NTIMES,), dtype=datetime.date)
          Array of date objects.
        data : dict{ str : np.ndarray((NTIMES, NX, NY), float)
          Dictionary with keys of variables and values storing the
          actual data.

        """
        if start is None:
            start = self._START
        start = self._clean_date(start)
        start_year = start.year

        if end is None:
            end = self._END
        end = self._clean_date(end)
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
        bounds = self._clean_bounds(polygon_or_bounds, crs)

        # download files
        filenames = []
        for var in variables:
            filename_var = []
            for year in range(start_year, end_year + 1):
                fname = self._download(var, year, bounds, force=force_download)
                filename_var.append(fname)
            filenames.append(filename_var)

        # open files
        s = watershed_workflow.datasets.State()
        for fnames, var in zip(filenames, variables):
            s[var] = self._open_files(fnames, var, start, end)
        return s
