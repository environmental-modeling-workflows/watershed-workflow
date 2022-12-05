"""Manager for downloading MODIS products from the NASA Earthdata AppEEARS database."""

import os, sys
import logging
import requests
import time
import datetime
import shapely
import numpy as np
import netCDF4
import attrs

import watershed_workflow.config
import watershed_workflow.sources.utils as source_utils
import watershed_workflow.sources.names
import watershed_workflow.warp
import watershed_workflow.crs

@attrs.define
class Task:
    task_id : str
    variables : list
    filenames : dict = attrs.Factory(dict)
    urls : dict = attrs.Factory(dict)
    shas : dict = attrs.Factory(dict)

    
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

    Note this is implemented based on the API documentation here:

       https://appeears.earthdatacloud.nasa.gov/api/?python#introduction
    """
    _LOGIN_URL = "https://appeears.earthdatacloud.nasa.gov/api/login" # URL for AppEEARS rest requests
    _TASK_URL = "https://appeears.earthdatacloud.nasa.gov/api/task"
    _STATUS_URL = "https://appeears.earthdatacloud.nasa.gov/api/status/"
    _BUNDLE_URL_TEMPLATE = "https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}"

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
            os.makedirs(self.names.folder_name())
        self.tasks = []
        self.completed_tasks = []

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

    def _construct_request(self, bounds_ll, start, end, variables):
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
        variables : list
          List of variables to collect.
        
        Returns
        -------
        token : int??
          Integer token for downloading this data.

        """
        if self.login_token is None:
            self.login_token = self._authenticate()

        (xmin, ymin, xmax, ymax) = tuple(bounds_ll)
        json_vars = [self._PRODUCTS[var] for var in variables]

        task = {
            "task_type": "area",
            "task_name": "Area LAI",
            "params": {
                "dates": [{
                    "startDate": start,
                    "endDate": end
                }],
                "layers": json_vars,
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
        logging.info('Constructing Task:')
        r = requests.post(self._TASK_URL,
                          json=task,
                          headers={ 'Authorization': f'Bearer {self.login_token}'})
        r.raise_for_status()

        task = Task(r.json()['task_id'], variables, filenames=dict((v,self._filename(bounds_ll, start, end, v)) for v in variables))
        self.tasks.append(task)
        logging.info(f'Requesting dataset on {bounds_ll} response task_id {task.task_id}')
        return task

    def _check_status(self, task=None):
        """Checks and prints the status of the task.

        Returns True, False, or 'UNKNOWN' when the response is not well formed, which seems to happen sometimes...
        """
        if self.login_token is None:
            self.login_token = self._authenticate()        
        if task is None:
            task = self.tasks[0]

        logging.info(f'Checking status of task: {task.task_id}')
        r = requests.get(self._STATUS_URL,
                         headers={'Authorization': 'Bearer {0}'.format(self.login_token)})
        try:
            r.raise_for_status()
        except requests.HTTPError:
            logging.info('... http error')
            return 'UNKNOWN'
        else:
            json = r.json()
            if len(json) == 0:
                logging.info('... status not found')
                return 'UNKNOWN'
            else:
                for entry in json:
                    if entry['task_id'] == task.task_id:
                        logging.info(entry)
                        if 'status' in entry and 'done' == entry['status']:
                            return True
                        else:
                            return False
            logging.info('... status not found')
            return 'UNKNOWN'

    def _check_bundle_url(self, task=None):
        if self.login_token is None:
            self.login_token = self._authenticate()        
        if task is None:
            task = self.tasks[0]

        logging.info(f'Checking for bundle of task: {task.task_id}')
        r = requests.get(self._BUNDLE_URL_TEMPLATE.format(task.task_id), 
                         headers={'Authorization': 'Bearer {0}'.format(self.login_token)})
        try:
            r.raise_for_status()
        except requests.HTTPError:
            return False
        else:
            # does the bundle exist?
            if len(r.json()) == 0:
                logging.info('... bundle not found')
                return False

            # bundle exists -- find the url and sha for each varname
            for var in task.variables:
                product = self._PRODUCTS[var]['product']
                found = False
                for entry in r.json()['files']:
                    if entry['file_name'].startswith(product):
                        logging.info(f'... bundle found {entry["file_name"]}')
                        assert(entry['file_name'].endswith('.nc'))
                        task.urls[var] = self._BUNDLE_URL_TEMPLATE.format(task.task_id)+'/'+entry['file_id']
                        found = True
                assert(found)
            return True

    def is_ready(self, task=None):
        """Actually knowing if it is ready is a bit tricky because Appeears does not appear to be saving its status after it is complete."""
        status = self._check_status(task)
        if status != False: # note this matches True or UNKNOWN
            return self._check_bundle_url(task)
        else:
            return status

    def _download(self, task=None):
        """Downloads the provided task.

        If file_id is not provided, is_ready() will be called.
        If file_id is provided, it is assumed is_ready() is True.

        If task is not provided, the first in the queue is used.
        """
        if task is None:
            task = self.tasks[0]

        if len(task.urls) == 0:
            ready = self._check_bundle_url(task)
        else:
            ready = True

        if ready:
            assert(len(task.filenames) == len(task.urls))
            assert(len(task.variables) == len(task.urls))
            for var in task.variables:
                url = task.urls[var]
                filename = task.filenames[var]
                logging.info("  Downloading: {}".format(url))
                logging.info("      to file: {}".format(filename))
                good = source_utils.download(url, filename, headers={ 'Authorization': f'Bearer {self.login_token}'})
                assert(good)
            return True
        else:
            return False

    def _open(self, task):
        """Open all files for a task, returning the data in the order of variables requested in the task."""
        data = []
        for var in task.variables:
            prof, d = self._open_file(task.filenames[var], var)
            data.append(d)
        return prof, data

    def _open_file(self, filename, variable):
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
                 variables=None,
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
        variables : str or list, optional
          Variable to download, currently one of {LAI, LULC}.  Default
          is both LAI and LULC.
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
          task tuple for use in a future call to get_data().

        """
        if task is None:
            # clean the variables list
            if variables is None:
                variables = ['LAI', 'LULC']
            for var in variables:
                if var not in self._PRODUCTS:
                    err = 'FileManagerMODISAppEEARS cannot provide variable {variable}.  Valid are: '
                    raise ValueError(err + ', '.join(self._PRODUCTS.keys()))

            # clean bounds
            bounds = self._clean_bounds(polygon_or_bounds, crs)

            # check start and end times
            if start is None:
                start = self._START
            if end is None:
                end = self._END
            start = self._clean_date(start)
            end = self._clean_date(end)

            # create a task
            task = Task('', variables, filenames=dict((v,self._filename(bounds, start, end, v)) for v in variables))

            # check for existing file
            if all(os.path.isfile(filename) for filename in task.filenames.values()):
                if force_download:
                    for filename in filenames:
                        os.path.remove(filename)
                else:
                    return self._open(task)

        if len(task.task_id) == 0:
            # create the task
            task = self._construct_request(bounds, start, end, variables)

        if self._download(task):
            return self._open(task)
        return task
