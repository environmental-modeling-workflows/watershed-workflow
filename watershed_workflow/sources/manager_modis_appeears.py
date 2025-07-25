"""Manager for downloading MODIS products from the NASA Earthdata AppEEARS database."""
from typing import Tuple, Dict, Optional, List

import os, sys
import logging
import requests
import time
import cftime, datetime
import shapely
import numpy as np
import attr
import rasterio.transform
import xarray as xr

import watershed_workflow.config
import watershed_workflow.sources.utils as source_utils
import watershed_workflow.sources.names
import watershed_workflow.warp
from watershed_workflow.crs import CRS
import watershed_workflow.crs

_colors = {
    -1: ('Unclassified', (0, 0, 0)),
    0: ('Open Water', (140, 219, 255)),
    1: ('Evergreen Needleleaf Forests', (38, 115, 0)),
    2: ('Evergreen Broadleaf Forests', (82, 204, 77)),
    3: ('Deciduous Needleleaf Forests', (150, 196, 20)),
    4: ('Deciduous Broadleaf Forests', (122, 250, 166)),
    5: ('Mixed Forests', (137, 205, 102)),
    6: ('Closed Shrublands', (215, 158, 158)),
    7: ('Open Shrublands', (255, 240, 196)),
    8: ('Woody Savannas', (233, 255, 190)),
    9: ('Savannas', (255, 216, 20)),
    10: ('Grasslands', (255, 196, 120)),
    11: ('Permanent Wetlands', (0, 132, 168)),
    12: ('Croplands', (255, 255, 115)),
    13: ('Urban and Built up lands', (255, 0, 0)),
    14: ('Cropland Natural Vegetation Mosaics', (168, 168, 0)),
    15: ('Permanent Snow and Ice', (255, 255, 255)),
    16: ('Barren Land', (130, 130, 130)),
    17: ('Water Bodies', (140, 209, 245)),
}

colors : Dict[int, Tuple[str,Tuple[float, ...]]] = dict()
for k, v in _colors.items():
    colors[k] = (v[0], tuple(float(i) / 255.0 for i in v[1]))

indices = dict([(pars[0], id) for (id, pars) in colors.items()])


@attr.define
class Task:
    task_id: str
    variables: list
    filenames: dict = attr.Factory(dict)
    urls: dict = attr.Factory(dict)
    shas: dict = attr.Factory(dict)


class ManagerMODISAppEEARS:
    """MODIS data through the AppEEARS data portal.

    Note this portal requires authentication -- please enter a
    username and password in your .watershed_workflowrc file.  For
    now, as this is not the highest security data portal or workflow
    package, we expect you to store this password in plaintext.  Maybe
    we can improve this?  If it bothers you, please ask how you can
    contribute (the developers of this package are not security
    experts!)

    To enter the username and password, register for a login in the
    AppEEARs data portal at:

    .. [AppEEARs](https://appeears.earthdatacloud.nasa.gov/)
    
    Currently the variables supported here include LAI and estimated
    ET.  

    All data returned includes a time variable, which is in units of
    [days past Jan 1, 2000, 0:00:00.

    Note this is implemented based on the API documentation here:

       https://appeears.earthdatacloud.nasa.gov/api/?python#introduction

    """
    _LOGIN_URL = "https://appeears.earthdatacloud.nasa.gov/api/login"  # URL for AppEEARS rest requests
    _TASK_URL = "https://appeears.earthdatacloud.nasa.gov/api/task"
    _STATUS_URL = "https://appeears.earthdatacloud.nasa.gov/api/status/"
    _BUNDLE_URL_TEMPLATE = "https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}"

    _START = datetime.date(2002, 7, 1)
    _END = datetime.date(2021, 1, 1)

    _PRODUCTS = {
        'LAI': {
            "layer": "Lai_500m",
            "product": "MCD15A3H.061"
        },
        'LULC': {
            "layer": "LC_Type1",
            "product": "MCD12Q1.061"
        },
    }

    colors = colors
    indices = indices

    def __init__(self,
                 login_token : Optional[str] = None,
                 remove_leap_day : bool = True):
        """Create a new manager for MODIS data."""
        self.name : str = 'MODIS'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'land_cover', self.name,
            'modis_{var}_{start}_{end}_{ymax}x{xmin}_{ymin}x{xmax}.nc')
        self.login_token = login_token
        if not os.path.isdir(self.names.folder_name()):
            os.makedirs(self.names.folder_name())
        self.tasks : List[Task] = []

    def _authenticate(self,
                      username : Optional[str] = None,
                      password : Optional[str] = None) -> str | None:
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

        try:
            lr = requests.post(self._LOGIN_URL, auth=(username, password))
            lr.raise_for_status()
            return lr.json()['token']
        except Exception as err:
            logging.warn('Unable to authenticate at Appeears database:')
            logging.warn('Message: {err}')
            return None

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

    def _cleanDate(self,
                   date : str | datetime.datetime | datetime.date) -> str:
        """Returns a string of the format needed for use in the filename and request."""
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        elif isinstance(date, datetime.datetime):
            date = date.date()
        assert isinstance(date, datetime.date)
        if date < self._START:
            raise ValueError(f"Invalid date {date}, must be after {self._START}.")
        if date > self._END:
            raise ValueError(f"Invalid date {date}, must be before {self._END}.")
        return date.strftime('%m-%d-%Y')

    def _cleanBounds(self,
                     geometry : shapely.geometry.base.BaseGeometry | Tuple[float, float, float, float],
                     crs : CRS) -> Tuple[float,float,float,float]:
        """Compute bounds in the required CRS from a polygon or bounds in a given crs"""
        if isinstance(geometry, shapely.geometry.base.BaseGeometry):
            bounds = geometry.bounds
        else:
            bounds = geometry
        bounds_ll = watershed_workflow.warp.bounds(bounds, crs, watershed_workflow.crs.latlon_crs)

        buffer = 0.01
        feather_bounds = list(bounds_ll[:])
        feather_bounds[0] = np.round(feather_bounds[0] - buffer, 4)
        feather_bounds[1] = np.round(feather_bounds[1] - buffer, 4)
        feather_bounds[2] = np.round(feather_bounds[2] + buffer, 4)
        feather_bounds[3] = np.round(feather_bounds[3] + buffer, 4)
        return tuple(feather_bounds)

    def _constructRequest(self,
                          bounds_ll : Tuple[float,float,float,float],
                          start : str,
                          end : str,
                          variables : List[str]) -> Task:
        """Create an AppEEARS request to download the variable from start to
        finish.  Note that this does not do the download -- it only creates
        the request.

        Parameters
        ----------
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

        task_data = {
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
                          json=task_data,
                          headers={ 'Authorization': f'Bearer {self.login_token}'})
        r.raise_for_status()

        task = Task(r.json()['task_id'],
                    variables,
                    filenames=dict(
                        (v, self._filename(bounds_ll, start, end, v)) for v in variables))
        self.tasks.append(task)
        logging.info(f'Requesting dataset on {bounds_ll} response task_id {task.task_id}')
        return task

    def _checkStatus(self,
                     task : Optional[Task] = None) -> str | bool:
        """Checks and prints the status of the task.

        Returns True, False, or 'UNKNOWN' when the response is not well formed, which seems to happen sometimes...
        """
        if self.login_token is None:
            self.login_token = self._authenticate()
        if task is None:
            task = self.tasks[0]

        logging.info(f'Checking status of task: {task.task_id}')
        r = requests.get(self._STATUS_URL,
                         headers={ 'Authorization': 'Bearer {0}'.format(self.login_token) },
                         verify=source_utils.getVerifyOption())
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

    def _checkBundleURL(self,
                        task : Optional[Task] = None) -> bool:
        if self.login_token is None:
            self.login_token = self._authenticate()
        if task is None:
            task = self.tasks[0]

        logging.info(f'Checking for bundle of task: {task.task_id}')
        r = requests.get(self._BUNDLE_URL_TEMPLATE.format(task.task_id),
                         headers={ 'Authorization': 'Bearer {0}'.format(self.login_token) },
                         verify=source_utils.getVerifyOption())

        try:
            r.raise_for_status()
        except requests.HTTPError as err:
            logging.info('... HTTPError checking for bundle:')
            logging.info(f'{err}')
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
                        assert (entry['file_name'].endswith('.nc'))
                        task.urls[var] = self._BUNDLE_URL_TEMPLATE.format(
                            task.task_id) + '/' + entry['file_id']
                        found = True
                assert (found)
            return True

    def is_ready(self, task : Optional[Task] = None) -> str | bool:
        """Actually knowing if it is ready is a bit tricky because Appeears does not appear to be saving its status after it is complete."""
        status = self._checkStatus(task)
        if status != False:  # note this matches True or UNKNOWN
            return self._checkBundleURL(task)
        else:
            return status

    def _download(self, task : Optional[Task] = None) -> bool:
        """Downloads the provided task.

        If file_id is not provided, is_ready() will be called.
        If file_id is provided, it is assumed is_ready() is True.

        If task is not provided, the first in the queue is used.
        """
        if task is None:
            task = self.tasks[0]

        if len(task.urls) == 0:
            ready = self._checkBundleURL(task)
        else:
            ready = True

        if ready:
            assert (len(task.filenames) == len(task.urls))
            assert (len(task.variables) == len(task.urls))
            for var in task.variables:
                url = task.urls[var]
                filename = task.filenames[var]
                logging.info("  Downloading: {}".format(url))
                logging.info("      to file: {}".format(filename))
                good = source_utils.download(
                    url, filename, headers={ 'Authorization': f'Bearer {self.login_token}'})
                assert (good)
            return True
        else:
            return False

    def _readData(self, task : Task) -> Dict[str, xr.DataArray]:
        """Read all files for a task, returning the data in the order of variables requested in the task."""
        darrays = dict((var, self._readFile(task.filenames[var], var)) for var in task.variables)
        return darrays

    def _readFile(self, filename : str, variable : str) -> xr.DataArray:
        """Open the file and get the data -- currently these reads it all, which may not be necessary."""
        with xr.open_dataset(filename) as fid:
            layer = self._PRODUCTS[variable]['layer']
            data = fid[layer]

        data.name = variable
        if data.rio.crs is None:
            data.rio.write_crs(watershed_workflow.crs.latlon_crs, inplace=True)
        assert data.rio.crs is not None
        return data

    def getDataset(self,
                geometry : Optional[shapely.geometry.base.BaseGeometry | Tuple[float,float,float,float]]= None,
                 crs : Optional[CRS] = None,
                 start : Optional[str | datetime.datetime | datetime.date] = None,
                 end : Optional[str | datetime.datetime | datetime.date] = None,
                 variables : Optional[List[str]] = None,
                 force_download : Optional[bool] = False,
                 task : Optional[Task] = None,
                 filenames : Optional[List[str]] = None) -> Dict[str, xr.DataArray] | Task:
        """Get dataset corresponding to MODIS data from the AppEEARS data portal.

        Note that AppEEARS requires the constrution of a request, and
        then prepares the data for you.  As a result, the raster may
        (if you've downloaded it previously, or it doesn't take very
        long) or may not be ready instantly.  

        Parameters
        ----------
        geometry : fiona or shapely shape, or [xmin, ymin, xmax, ymax]
          Collect a file that covers this shape or bounds.
        crs : CRS object
          Coordinate system of the above geometry
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
        filenames : list of str, optional
            If a list of filenames is provided, use these rather than creating a new request.
        
        Returns
        -------
        dict : { variable : (profile, times, data) }
          Returns a dictionary of (variable, data) pairs. For each
          variable, profile is a dictionary of standard raster profile
          information, times is an array of datetime objects of length
          NTIMES, and data is an array of shape (NTIMES, NX, NY) storing
          the actual values.

        OR

        task : (task_id, filename)
          If the data is not yet ready after the wait time, returns a
          task tuple for use in a future call to getDataset().

        """
        if filenames is not None:
            # read the file
            assert variables is not None, "Must provide variables if providing filenames."
            darrays = dict((var, self._readFile(filename, var)) for (filename, var) in zip(filenames, variables))
            return darrays

        if task is None and filenames is None:
            if geometry is None or crs is None:
                raise RuntimeError(
                    'Must provide either polgyon_or_bounds and crs or task arguments.')

            # clean the variables list
            if variables is None:
                variables = ['LAI', 'LULC']
            for var in variables:
                if var not in self._PRODUCTS:
                    err = 'FileManagerMODISAppEEARS cannot provide variable {variable}.  Valid are: '
                    raise ValueError(err + ', '.join(self._PRODUCTS.keys()))

            # clean bounds
            bounds = self._cleanBounds(geometry, crs)

            # check start and end times
            if start is None:
                start = self._START
            if end is None:
                end = self._END
            start_str = self._cleanDate(start)
            end_str = self._cleanDate(end)

            # create a task
            task = Task('',
                        variables,
                        filenames=dict(
                            (v, self._filename(bounds, start_str, end_str, v)) for v in variables))

            # check for existing file
            for filename in task.filenames.values():
                logging.info(f'... searching for: {filename}')
            if all(os.path.isfile(filename) for filename in task.filenames.values()):
                if force_download:
                    for filename in task.filenames:
                        try:
                            os.remove(filename)
                        except FileNotFoundError:
                            pass
                else:
                    return self._readData(task)

        if len(task.task_id) == 0:
            # create the task
            assert variables is not None
            task = self._constructRequest(bounds, start_str, end_str, variables)

        if self._download(task):
            return self._readData(task)
        return task

    def wait(self, task, interval=120, tries=100):
        """Blocking -- waits for a task to end."""
        count = 0
        success = False
        res = task
        while count < tries and not success:
            res = self.getDataset(task=res)
            if isinstance(res, Task):
                logging.info('sleeping...')
                time.sleep(interval)
                count += 1
            else:
                success = True
                break

        if success:
            return res
        else:
            raise RuntimeError(f'Unable to get data after {interval*tries} seconds.')
