"""Manager for downloading MODIS products from the NASA Earthdata AppEEARS database."""

import os, sys
import logging
import requests
import time
import datetime
import shapely
import numpy as np
import netCDF4

import watershed_workflow.config
import watershed_workflow.sources.utils as source_utils
import watershed_workflow.sources.names
import watershed_workflow.warp
import watershed_workflow.crs


class FileManagerMODISAppEEARS:
    """MODIS data through the AppEEARS data portal.

    Note this portal requires authentication -- please enter a
    username and password in your .watershed_workflowrc file.  For
    now, as this is not the highest security data portal or workflow
    package, we expect you to store this password in plaintext.  Maybe
    we can improve this?  If it bothers you, please ask how you can
    contribute (the developers of this package are not security
    experts!)

    .. [MODIS] << insert link here >>
    
    Currently the variables supported here include LAI and estimated ET.

    << DOCUMENT ACCESS PATTERN HERE >>
    """
    _LOGIN_URL = "https://appeears.earthdatacloud.nasa.gov/api/login" # URL for AppEEARS rest requests
    _TASK_URL = "https://appeears.earthdatacloud.nasa.gov/api/task"
    _BUNDLE_URL_TEMPLATE = "https://appeears.earthdatacloud.nasa.gov/api/bundle/"

    _START = datetime.date(2002, 7, 1)
    _END = datetime.date(2020, 12, 30)

    _PRODUCTS = {
        'LAI': {
            "layer": "Lai_500m",
            "product": "MCD15A3H.006"
        },
        'LULC': {
            "layer": "LC_Type1",
            "product": "MCD12Q1.006"
        },
    }

    def __init__(self, login_token=None):
        self.name = 'MODIS'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'land_cover', self.name,
            'modis_{var}_{start}_{end}_{ymax}x{xmin}_{ymin}x{xmax}.nc')
        self.login_token = login_token
        if not os.path.isdir(self.names.folder_name()):
            os.mkdir(self.names.folder_name())
        self.tasks = []

    def _authenticate(self, username=None, password=None):
        """Authenticate to the AppEEARS API.

        Parameters
        ----------
        username : string, optional
          Username, defaults to value from watershed_workflowrc,
          conf['AppEEARS']['username']
        password : string, optional
          Username, defaults to value from watershed_workflowrc,
          conf['AppEEARS']['password'].  

        FIXME: Can we make this more secure? --etc
        """
        if username == None:
            username = watershed_workflow.config.rcParams['AppEEARS']['username']
        if password is None:
            password = watershed_workflow.config.rcParams['AppEEARS']['password']

        if username == "NOT_PROVIDED" or password == "NOT_PROVIDED":
            raise ValueError(
                "Username or password for AppEEARS are not set in watershed_workflowrc.")

        lr = requests.post(self._LOGIN_URL, auth=(username, password))
        lr.raise_for_status()
        return lr.json()['token']

    def _filename(self, bounds_ll, start, end, variable):
        (xmin, ymin, xmax, ymax) = tuple(bounds_ll)
        filename = self.names.file_name(var=variable,
                                        start=start,
                                        end=end,
                                        xmin=xmin,
                                        xmax=xmax,
                                        ymin=ymin,
                                        ymax=ymax)
        return filename

    def _clean_date(self, date):
        # test input: date
        if type(date) is str:
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        if date < self._START:
            date = self._START
        if date > self._END:
            date = self._END
        return date.strftime('%m-%d-%Y')

    def _clean_bounds(self, polygon_or_bounds, crs):
        """Compute bounds in the needed CRS"""
        if type(polygon_or_bounds) is dict:
            polygon_or_bounds = watershed_workflow.utils.shply(polygon_or_bounds)
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

    def _construct_request(self, bounds_ll, start, end, variable):
        """Create an AppEEARS request to download the variable from start to
        finish.  Note that this does not do the download -- it only creates
        the request.

        Parameters
        ----------
        polygon_or_bounds : fiona or shapely shape, or [xmin, ymin, xmax, ymax]
          Collect a file that covers this shape or bounds.
        crs : CRS object
          Coordinate system of the above polygon_or_bounds
        buffer : float
          buffer size in units of CRS (or degrees? FIXME --etc)
        start : int
          Year to start, must be XXXX -- YYYY.  Defaults to XXXX
        end : int
          Year to end (inclusive), must be XXXX -- YYYY and greater
          than start.  Defaults to YYYY
        
        Returns
        -------
        token : int??
          Integer token for downloading this data.

        """
        if login_token is None:
            login_token = self._authenticate()

        filename = self._filename(bounds_ll, start, end, variable)
        (xmin, ymin, xmax, ymax) = tuple(bounds_ll)

        task = {
            "task_type": "area",
            "task_name": "Area LAI",
            "params": {
                "dates": [{
                    "startDate": start,
                    "endDate": end
                }],
                "layers": [self._PRODUCTS[variable], ],
                "output": {
                    "format": {
                        "type": "netcdf4"
                    },
                    "projection": "geographic"
                },
                "geo": {
                    "type":
                    "FeatureCollection",
                    "fileName":
                    "User-Drawn-Polygon",
                    "features": [{
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type":
                            "Polygon",
                            "coordinates": [[[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin],
                                             [xmin, ymin]]]
                        }
                    }, ]
                }
            }
        }

        # submit the task request
        response = requests.post(self._TASK_URL,
                                 json=task,
                                 headers={ 'Authorization': f'Bearer {self.login_token}'})
        logging.info('Constructing Task: {response.url}')
        response.raise_for_status()
        self.tasks.append((response.json()['task_id'], filename))
        logging.info(f'Requesting dataset on {bounds_ll} response task_id {self.tasks[-1][0]}')
        return self.tasks[-1]

    def _is_ready(self, task=None):
        """Checks whether the provided token (or last token generated) is ready."""
        if task is None:
            task = self.tasks[0]

        url = self._BUNDLE_URL_TEMPLATE + task[0]
        response = requests.get(url)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            return False
        else:
            json = response.json()
            return next(f for f in json['files'] if f['file_type'] == 'nc')

    def _download(self, task=None, file_id=None):
        """Downloads the provided task/file_id.

        If file_id is not provided, is_ready() will be called.
        If file_id is provided, it is assumed is_ready() is True.

        If task is not provided, the first in the queue is used.
        """
        if file_id is None:
            file_id = self._is_ready(task)
        if file_id is False:
            return None

        if task is None:
            task = self.tasks[0]
        task_id, filename = task

        url = self._BUNDLE_URL_TEMPLATE + task_id + '/' + file_id['file_id']
        logging.info("  Downloading: {}".format(url))
        logging.info("      to file: {}".format(filename))

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as fid:
            for data in response.iter_content(chunk_size=8192):
                fid.write(data)
        return filename

    def _open(self, filename, variable):
        """Open the file and get the data -- currently these reads it all, which may not be necessary."""
        profile = dict()
        with netCDF4.Dataset(filename, 'r') as nc:
            profile['crs'] = watershed_workflow.crs.from_epsg(nc.variables['crs'].epsg_code)
            profile['width'] = nc.dimensions['lon'].size
            profile['height'] = nc.dimensions['lat'].size
            profile['count'] = nc.dimensions['time'].size
            profile['nodata'] = np.nan

            # this assumes it is a fixed dx and dy, which should be
            # pretty good for not-too-big domains.
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            profile['dx'] = (lon[1:] - lon[:-1]).mean()
            profile['dy'] = (lat[1:] - lat[:-1]).mean()
            profile['driver'] = 'netCDF4'  # hint that this was not a real reaster!
            # do we need to make an affine transform?

            varname = self._PRODUCTS[variable]['layer']
            profile['layer'] = varname
            data = nc.variables[varname][:].filled(np.nan)
        return profile, data

    def get_data(self,
                 polygon_or_bounds,
                 crs,
                 start=None,
                 end=None,
                 variable='LAI',
                 force_download=False,
                 task=None):
        """Get dataset corresponding to MODIS data from the AppEEARS data portal.

        Note that AppEEARS requires the constrution of a request, and
        then prepares the data for you.  As a result, the raster may
        (if you've downloaded it previously, or it doesn't take very
        long) or may not be ready instantly.  

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
          <= 2020-12-30.
        variable : str, optional
          Variable to download, currently one of {LAI, LULC}.  Default
          is LAI.
        force_download : bool, optional
          Force a new file to be downloaded.  Default is False.
        task : (str, str) tuple of task_id, filename
          If a request has already been created, use this task to
          access the data rather than creating a new request.  Default
          means to create a new request.
        
        Returns
        -------
        (data, profile) : (np.ndarray of shape [NX,NY,NT], dict)
          If the file already exists or the download is successful,
          returns the 3D array of data and a dictionary containing
          metadata including bounds, affine transform, CRS, etc.

        OR

        task : (task_id, filename)
          If the data is not yet ready after the wait time, returns a
          task tuple for use in a future call to get_raster().

        """
        if task is not None:
            task_id, filename = task
        else:
            # clean bounds
            bounds = self._clean_bounds(polygon_or_bounds, crs)

            # check start and end times
            start = self._clean_date(start)
            end = self._clean_date(end)

            if variable not in self._PRODUCTS:
                err = 'FileManagerMODISAppEEARS cannot provide variable {variable}.  Valid are: '
                raise ValueError(err + ', '.join(self._PRODUCTS.keys()))

            filename = self._filename(bounds, start, end, variable)
            task_id = None

        # check for existing file
        if os.path.isfile(filename):
            if force_download:
                os.path.remove(filename)
            else:
                return self._open(filename, variable)

        if task_id is None:
            # create the task
            task = self._construct_request(bounds, start, end, variable)
            assert (filename == task[1])
            task_id = task[0]

        task = (task_id, filename)
        file_id = self._is_ready(task)
        if file_id:
            self._download(task, file_id)
            return self._open(filename, variable)

        return task
