"""Manager for downloading MODIS products from the NASA Earthdata AppEEARS database."""
from typing import Tuple, Dict, Optional, List, overload

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
import watershed_workflow.warp
from watershed_workflow.crs import CRS
import watershed_workflow.crs

from . import utils as source_utils
from . import filenames
from . import manager_dataset

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



class ManagerMODISAppEEARS(manager_dataset.ManagerDataset):
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

    class Request(manager_dataset.ManagerDataset.Request):
        """MODIS AppEEARS-specific request that includes Task information."""
        def __init__(self,
                     request : manager_dataset.ManagerDataset.Request,
                     task_id : str = "",
                     filenames : Optional[Dict[str, str]] = None,
                     urls : Optional[Dict[str, str]] = None):
            super().copyFromExisting(request)
            self.task_id = task_id
            self.filenames = filenames
            self.urls = urls


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

    def __init__(self, login_token : Optional[str] = None):
        """Create a new manager for MODIS data."""
        
        # Native MODIS properties for base class
        import cftime
        native_resolution = 500.0 * 9.e-6 # in native_crs
        native_start = cftime.datetime(2002, 7, 1, calendar='standard')  
        native_end = cftime.datetime(2021, 1, 1, calendar='standard')
        native_crs = CRS.from_epsg(4269)  # WGS84 Geographic
        valid_variables = ['LAI', 'LULC']
        default_variables = ['LAI', 'LULC']
        
        # Initialize base class with correct parameter order
        super().__init__(
            name='MODIS',
            source='AppEEARS',
            native_resolution=native_resolution,
            native_crs_in=native_crs,
            native_crs_out=native_crs,
            native_start=native_start,
            native_end=native_end,
            valid_variables=valid_variables,
            default_variables=default_variables
        )
        
        # AppEEARS-specific initialization
        self.names = filenames.Names(self.name, 'land_cover', 'MODIS',
                                     'modis_{var}_{start}_{end}_{ymax}x{xmin}_{ymin}x{xmax}.nc')
        self.login_token = login_token
        if not os.path.isdir(self.names.folder_name()):
            os.makedirs(self.names.folder_name())

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
            logging.info('Unable to authenticate at Appeears database:')
            logging.info('Message: {err}')
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

    
    def _constructRequest(self,
                          bounds_ll : Tuple[str,str,str,str],
                          start : str,
                          end : str,
                          variables : List[str]) -> str:
        """Create an AppEEARS request to download the variable from start to
        finish.  Note that this does not do the download -- it only creates
        the request.

        Parameters
        ----------
        start : str
          Start date string in MM-DD-YYYY format
        end : str
          End date string in MM-DD-YYYY format
        variables : list
          List of variables to collect.
        
        Returns
        -------
        str
          Task ID for the AppEEARS request.

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
        r = requests.post(self._TASK_URL,
                          json=task_data,
                          headers={ 'Authorization': f'Bearer {self.login_token}'})
        r.raise_for_status()

        task_id = r.json()['task_id']
        logging.info(f'Requested AppEEARS MODIS dataset on {bounds_ll} yielded task_id {task_id}')
        return task_id

    def _checkStatus(self, request: manager_dataset.ManagerDataset.Request) -> str | bool:
        """Checks and prints the status of the AppEEARS request.

        Returns True, False, or 'UNKNOWN' when the response is not well formed, which seems to happen sometimes...
        """
        if self.login_token is None:
            self.login_token = self._authenticate()

        logging.info(f'Checking status of task: {request.task_id}')
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
                    if entry['task_id'] == request.task_id:
                        logging.info(entry)
                        if 'status' in entry and 'done' == entry['status']:
                            logging.info('... is ready!')
                            return True
                        else:
                            logging.info('... is NOT ready!')
                            return False
            logging.info('... status not found')
            return 'UNKNOWN'

    def _checkBundleURL(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        if self.login_token is None:
            self.login_token = self._authenticate()

        logging.info(f'Checking for bundle of task: {request.task_id}')
        r = requests.get(self._BUNDLE_URL_TEMPLATE.format(request.task_id),
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
            for var in request.variables:
                product = self._PRODUCTS[var]['product']
                found = False
                for entry in r.json()['files']:
                    if entry['file_name'].startswith(product):
                        logging.info(f'... bundle found {entry["file_name"]}')
                        assert (entry['file_name'].endswith('.nc'))
                        request.urls[var] = self._BUNDLE_URL_TEMPLATE.format(
                            request.task_id) + '/' + entry['file_id']
                        found = True
                assert (found)
            return True

    def _download(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Downloads the data for the provided request."""
        if len(request.urls) == 0:
            ready = self._checkBundleURL(request)
        else:
            ready = True

        if ready:
            assert (len(request.filenames) == len(request.urls))
            assert (len(request.variables) == len(request.urls))
            for var in request.variables:
                url = request.urls[var]
                filename = request.filenames[var]
                logging.info("  Downloading: {}".format(url))
                logging.info("      to file: {}".format(filename))
                good = source_utils.download(
                    url, filename, headers={ 'Authorization': f'Bearer {self.login_token}'})
                assert (good)
            return True
        else:
            return False

        
    def _readData(self, request) -> xr.Dataset:
        """Read all files for a request, returning the data as a Dataset."""
        darrays = dict((var, self._readFile(request.filenames[var], var)) for var in request.variables)

        # keep independent times for LAI (which is every 3-6 days) and
        # LULC (which is once a yearish)
        for k,v in darrays.items():
            darrays[k] = darrays[k].rename({'time': f'time_{k}'})

        # Convert to Dataset
        dataset = xr.Dataset(darrays)
        return dataset

    
    def _readFile(self, filename : str, variable : str) -> xr.DataArray:
        """Open the file and get the data -- currently these reads it all, which may not be necessary."""
        with xr.open_dataset(filename) as fid:
            layer = self._PRODUCTS[variable]['layer']
            data = fid[layer]

        data.name = variable
        return data

    
    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request
                        ) -> manager_dataset.ManagerDataset.Request:
        """Request MODIS data from AppEEARS - may not be ready immediately.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
            
        Returns
        -------
        ManagerDataset.Request
            MODIS request object with AppEEARS task information.
        """
        # Geometry is already in native_crs_in (WGS84), get bounds directly
        appeears_bounds = [np.round(b,4) for b in request.geometry.bounds]
        appeears_bounds_str = [f'{b:.4f}' for b in appeears_bounds]
        logging.info(f'Building request for bounds: {appeears_bounds}')
        
        # Convert dates to strings for AppEEARS API
        start_str = request.start.strftime('%m-%d-%Y')
        end_str = request.end.strftime('%m-%d-%Y')
        
        # Create filenames for caching
        filenames = dict((v, self._filename(appeears_bounds_str, start_str, end_str, v)) 
                        for v in request.variables)
        logging.info('... requires files:')
        for fname in filenames.values():
            logging.info(f' ... {fname}')

        
        # Check for existing files
        if all(os.path.isfile(filename) for filename in filenames.values()):
            logging.info('... files exist locally.')
            # Data already exists locally
            modis_request = self.Request(
                request,
                task_id="",  # No remote task needed
                filenames=filenames,
                urls={}
            )
            modis_request.is_ready = True

        else:
            logging.info('... building request.')

            # Need to create AppEEARS request
            task_id = self._constructRequest(appeears_bounds, start_str, end_str, request.variables)
        
            # Create MODIS-specific request with AppEEARS task info
            modis_request = self.Request(
                request,
                task_id=task_id,
                filenames=filenames,
                urls={}  # Will be populated when ready
            )
        
        return modis_request


    def _fetchDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Implementation of abstract method to fetch MODIS data.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing AppEEARS task information.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested MODIS data.
        """
        # If data exists locally, read it directly
        if all(os.path.isfile(filename) for filename in request.filenames.values()):
            return self._readData(request)
        
        # Otherwise, download from AppEEARS
        if self._download(request):
            return self._readData(request)
        else:
            raise RuntimeError(f"Unable to download MODIS data for task {request.task_id}")


    def isReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Check if MODIS data request is ready for download.
        
        Overrides base class to check AppEEARS processing status and bundle availability.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            MODIS request object with AppEEARS task information.
            
        Returns
        -------
        bool
            True if data is ready for download, False otherwise.
        """
        if request.is_ready:
            return True
        
        # Check AppEEARS status
        status = self._checkStatus(request)
        if status != False:  # note this matches True or UNKNOWN
            ready = self._checkBundleURL(request)
            if ready:
                request.is_ready = True
            return ready
        else:
            return False



