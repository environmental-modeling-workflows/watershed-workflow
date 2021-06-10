"""Manager for interacting with DayMet datasets."""

import os,sys
import logging
import numpy as np
import pyproj
import requests
import requests.exceptions
import shapely.geometry

import workflow.sources.utils as source_utils
import workflow.crs
import workflow.conf
import workflow.warp
import workflow.sources.names


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
          - :math:`^\circ C`
          - Min/max daily air temperature
        * - srad
          - :math:`W / m^2`
          - Incoming solar radiation
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
    
    VALID_YEARS = (1980,2020)
    VALID_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    # URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/1328/{year}/daymet_v3_{variable}_{year}_na.nc4"
    URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/1840/daymet_v4_daily_na_{variable}_{year}.nc"

    
    def __init__(self):
        #self.layer_name = 'Daymet_{1}_{0}_{2}'.format(self.layer, self.year, self.location)
        self.name = 'DayMet 1km'
        self.names = workflow.sources.names.Names(self.name, 'meteorology', 'daymet', 'daymet_{var}_{year}_{north}x{west}_{south}x{east}.nc')
        #self.native_crs = pyproj.Proj4("")

    def get_meteorology(self, varname, year, polygon_or_bounds, crs, force_download=False):
        """Gets file for a single year and single variable.

        Parameters
        ----------
        varname : str
          Name the variable, see table in the class documentation.
        year : int
          A year in the valid range (currently 1980-2018)
        polygon_or_bounds : fiona or shapely shape, or [xmin, ymin, xmax, ymax]
          Collect a file that covers this shape or bounds.
        crs : CRS object
          Coordinate system of the above polygon_or_bounds
        force_download : bool
          Download or re-download the file if true.

        Returns
        -------
        filename : str
          Path to the data file.
        """
        if year > self.VALID_YEARS[1] or year < self.VALID_YEARS[0]:
            raise ValueError("DayMet data is available from {} to {} (does not include {})".format(self.VALID_YEARS[0], self.VALID_YEARS[1], year))
        if varname not in self.VALID_VARIABLES:
            raise ValueError("DayMet data supports variables: {} (not {})".format(', '.join(self.VALID_VARIABLES), varname))

        if type(polygon_or_bounds) is dict:
            # fiona shape object, convert to shapely to get a copy
            polygon_or_bounds = workflow.utils.shply(polygon_or_bounds)

        # convert and get a bounds
        if type(polygon_or_bounds) is list:
            # bounds
            bounds = workflow.warp.bounds(polygon_or_bounds, crs, workflow.crs.latlon_crs())
        else:
            # polygon
            bounds = workflow.warp.shply(polygon_or_bounds, crs, workflow.crs.latlon_crs()).bounds

        # feather the bounds
        # get the bounds and download
        feather_bounds = list(bounds[:])
        feather_bounds[0] = feather_bounds[0] - .01
        feather_bounds[1] = feather_bounds[1] - .01
        feather_bounds[2] = feather_bounds[2] + .01
        feather_bounds[3] = feather_bounds[3] + .01
        fname = self.download(varname, year, feather_bounds, force=force_download)
        return fname, feather_bounds

    def download(self, varname, year, bounds, force=False):
        """Download a NetCDF file covering the bounds.
        
        Note: prefer to use get_meteorology() 

        Parameters
        ----------
        varname : str
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
        filename = self.names.file_name(var = varname, year=year, north=bounds[3], east=bounds[2], west=bounds[0], south=bounds[1])

        if (not os.path.exists(filename)) or force:
            url_dict = {'year':str(year),
                        'variable':varname}
            url = self.URL.format(**url_dict)
            logging.info("  Downloading: {}".format(url))
            logging.info("      to file: {}".format(filename))

            request_params = [('var', 'lat'),
                          ('var', 'lon'),
                          ('var', varname),
                          ('west', str(bounds[0])),
                          ('south', str(bounds[1])),
                          ('east', str(bounds[2])),
                          ('north', str(bounds[3])),
                          ('horizStride', '1'),
                          ('time_start', '{}-01-01T12:00:00Z'.format(year)),
                          ('time_end', '{}-12-31T12:00:00Z'.format(year)),
                          ('timeStride', '1'),
                          ('accept', 'netcdf')
                          ]

            r = requests.get(url,params=request_params)
            r.raise_for_status()

            with open(filename, 'wb') as fid:
                fid.write(r.content)

        else:
            logging.info("  Using existing: {}".format(filename))

        return filename
        
