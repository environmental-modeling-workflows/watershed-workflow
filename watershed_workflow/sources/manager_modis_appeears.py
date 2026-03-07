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
                     cache_filenames : Optional[Dict[str, str]] = None,
                     urls : Optional[Dict[str, str]] = None):
            super().copyFromExisting(request)
            self.task_id = task_id
            self.cache_filenames = cache_filenames
            self.urls = urls


    _LOGIN_URL = "https://appeears.earthdatacloud.nasa.gov/api/login"  # URL for AppEEARS rest requests
    _TASK_URL = "https://appeears.earthdatacloud.nasa.gov/api/task"
    _STATUS_URL = "https://appeears.earthdatacloud.nasa.gov/api/status/"
    _BUNDLE_URL_TEMPLATE = "https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}"

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

        import cftime
        native_resolution = 500.0 * 9.e-6  # in native_crs (degrees)
        native_start = cftime.datetime(2002, 7, 1, calendar='standard')
        native_end = cftime.datetime(2024, 1, 1, calendar='standard')
        native_crs = CRS.from_epsg(4269)  # WGS84 Geographic
        valid_variables = ['LAI', 'LULC']
        default_variables = ['LAI', 'LULC']

        super().__init__(
            name='MODIS',
            source='AppEEARS',
            native_resolution=native_resolution,
            native_crs_in=native_crs,
            native_crs_out=native_crs,
            native_start=native_start,
            native_end=native_end,
            valid_variables=valid_variables,
            default_variables=default_variables,
            cache_category='land_cover',
            cache_extension='nc',
            has_varname=True,
            short_name='MODIS',
        )

        self.login_token = login_token
        os.makedirs(self._cacheFolder(), exist_ok=True)

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

    def _constructRequest(self,
                          snapped_bounds : tuple,
                          start_year : int,
                          end_year : int,
                          variables : List[str]) -> str:
        """Create an AppEEARS request to download the variable for whole years.

        Parameters
        ----------
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` snapped bounds.
        start_year : int
            Start year (inclusive).
        end_year : int
            End year (inclusive).
        variables : list of str
            Variables to collect.

        Returns
        -------
        str
            Task ID for the AppEEARS request.
        """
        if self.login_token is None:
            self.login_token = self._authenticate()

        xmin, ymin, xmax, ymax = snapped_bounds
        start_str = f'01-01-{start_year}'
        end_str = f'12-31-{end_year}'
        json_vars = [self._PRODUCTS[var] for var in variables]

        task_data = {
            "task_type": "area",
            "task_name": "Area LAI",
            "params": {
                "dates": [{
                    "startDate": start_str,
                    "endDate": end_str
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

        r = requests.post(self._TASK_URL,
                          json=task_data,
                          headers={ 'Authorization': f'Bearer {self.login_token}'})
        r.raise_for_status()

        task_id = r.json()['task_id']
        logging.info(f'Requested AppEEARS MODIS dataset on {snapped_bounds} yielded task_id {task_id}')
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
        """Download the data files for the provided request to cache filenames."""
        os.makedirs(self._cacheFolder(), exist_ok=True)

        if len(request.urls) == 0:
            ready = self._checkBundleURL(request)
        else:
            ready = True

        if ready:
            assert (len(request.cache_filenames) == len(request.urls))
            assert (len(request.variables) == len(request.urls))
            for var in request.variables:
                url = request.urls[var]
                filename = request.cache_filenames[var]
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
        darrays = dict((var, self._readFile(request.cache_filenames[var], var))
                       for var in request.variables)

        # keep independent times for LAI (which is every 3-6 days) and
        # LULC (which is once a yearish)
        for k,v in darrays.items():
            darrays[k] = darrays[k].rename({'time': f'time_{k}'})

        dataset = xr.Dataset(darrays)
        return dataset


    def _readFile(self, filename : str, variable : str) -> xr.DataArray:
        """Open the file and get the data."""
        with xr.open_dataset(filename) as fid:
            layer = self._PRODUCTS[variable]['layer']
            data = fid[layer]

        data.name = variable
        return data


    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        task_id : Optional[str] = None
                        ) -> manager_dataset.ManagerDataset.Request:
        """Request MODIS data from AppEEARS - may not be ready immediately.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
        task_id : str, optional
            Existing AppEEARS task ID to resume, if any.

        Returns
        -------
        ManagerDataset.Request
            MODIS request object with AppEEARS task information.
        """
        # Snap dates to whole years for cache stability
        start_year = request.start.year
        end_year = request.end.year
        snapped_bounds = request.snapped_bounds
        geometry_bounds = request.geometry.bounds

        logging.info(f'Building MODIS request for bounds: {snapped_bounds}, years {start_year}-{end_year}')

        # Build standard cache filenames for each variable
        var_filenames = {}
        for var in request.variables:
            # Check for a spatial+temporal superset in the cache
            superset = self._checkCache(geometry_bounds, snapped_bounds,
                                        var=var, start_year=start_year, end_year=end_year)
            if superset is not None:
                logging.info(f'  Using superset cache for {var}: {superset}')
                var_filenames[var] = superset
            else:
                var_filenames[var] = self._cacheFilename(
                    snapped_bounds, var=var, start_year=start_year, end_year=end_year)

        logging.info('  Cache filenames:')
        for fname in var_filenames.values():
            logging.info(f'    {fname}')

        # Check whether all files are already present
        if all(os.path.isfile(f) for f in var_filenames.values()):
            logging.info('  All files exist locally.')
            modis_request = self.Request(
                request,
                task_id="",
                cache_filenames=var_filenames,
                urls={}
            )
            modis_request.is_ready = True
        else:
            logging.info('  Building AppEEARS request.')
            if task_id is None:
                task_id = self._constructRequest(
                    snapped_bounds, start_year, end_year, request.variables)

            modis_request = self.Request(
                request,
                task_id=task_id,
                cache_filenames=var_filenames,
                urls={}
            )

        return modis_request


    def _fetchDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Fetch MODIS data from cache or AppEEARS.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing AppEEARS task information.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested MODIS data.
        """
        if all(os.path.isfile(f) for f in request.cache_filenames.values()):
            return self._readData(request)

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

        status = self._checkStatus(request)
        if status != False:  # note this matches True or UNKNOWN
            ready = self._checkBundleURL(request)
            if ready:
                request.is_ready = True
            return ready
        else:
            return False
