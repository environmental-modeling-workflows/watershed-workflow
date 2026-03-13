"""Manager for downloading MODIS products from the NASA Earthdata AppEEARS database."""
from typing import Tuple, Dict, Optional, List

import os, sys
import logging
import netrc
import requests
import time
import cftime, datetime
import shapely
import numpy as np
import rasterio.transform
import xarray as xr

import watershed_workflow.utils.warp
from watershed_workflow.crs import CRS
import watershed_workflow.crs

from . import utils as source_utils
from . import manager_dataset
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import CacheInfo, _snapBounds


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


_CACHE_INFO = CacheInfo(
    category='land_cover',
    subcategory='modis_appeears',
    name='modis_appeears',
    snap_resolution=0.01,
    is_temporal=True,
)


@cached_dataset_manager(_CACHE_INFO)
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

    _LOGIN_URL = "https://appeears.earthdatacloud.nasa.gov/api/login"
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

    def __init__(self, login_token: Optional[str] = None):
        """Create a new manager for MODIS data."""

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
        )

        self.login_token = login_token

    def isComplete(self, dir: str, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True if all per-variable NetCDF files exist in the cache directory.

        Parameters
        ----------
        dir : str
            Absolute path to a candidate cache directory.
        request : ManagerDataset.Request
            The request being fulfilled.

        Returns
        -------
        bool
            True if ``{var}.nc`` exists for every requested variable.
        """
        for var in request.variables:
            if not os.path.isfile(os.path.join(dir, f'{var}.nc')):
                return False
        return True

    def _authenticate(self,
                      username: Optional[str] = None,
                      password: Optional[str] = None) -> str | None:
        """Authenticate to the AppEEARS API using NASA Earthdata credentials.

        Parameters
        ----------
        username : str, optional
            NASA Earthdata username.  If not provided, read from ``~/.netrc``
            (``machine urs.earthdata.nasa.gov``).
        password : str, optional
            NASA Earthdata password.  If not provided, read from ``~/.netrc``.
        """
        if username is None or password is None:
            try:
                creds = netrc.netrc().authenticators('urs.earthdata.nasa.gov')
            except FileNotFoundError:
                creds = None
            if creds is None:
                raise ValueError(
                    'NASA Earthdata credentials not found. Add an entry to ~/.netrc:\n\n'
                    '    machine urs.earthdata.nasa.gov login <username> password <password>\n\n'
                    'Register for a free account at https://urs.earthdata.nasa.gov'
                )
            username, _, password = creds

        try:
            lr = requests.post(self._LOGIN_URL, auth=(username, password))
            lr.raise_for_status()
            return lr.json()['token']
        except Exception as err:
            logging.info(f'Unable to authenticate at AppEEARS database: {err}')
            return None

    def _constructRequest(self,
                          snapped_bounds: tuple,
                          start_year: int,
                          end_year: int,
                          variables: List[str]) -> str:
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
        """Check and print the status of the AppEEARS request.

        Returns True, False, or 'UNKNOWN' when the response is not well formed.
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
        """Check for the AppEEARS bundle and populate download URLs on the request."""
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
            if len(r.json()) == 0:
                logging.info('... bundle not found')
                return False

            for var in request.variables:
                product = self._PRODUCTS[var]['product']
                found = False
                for entry in r.json()['files']:
                    if entry['file_name'].startswith(product):
                        logging.info(f'... bundle found {entry["file_name"]}')
                        assert (entry['file_name'].endswith('.nc'))
                        request._urls[var] = self._BUNDLE_URL_TEMPLATE.format(
                            request.task_id) + '/' + entry['file_id']
                        found = True
                assert (found)
            return True

    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        task_id: Optional[str] = None
                        ) -> manager_dataset.ManagerDataset.Request:
        """Submit the AppEEARS task if not already submitted.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
        task_id : str, optional
            Existing AppEEARS task ID to resume, if any.

        Returns
        -------
        ManagerDataset.Request
            The same request with ``task_id`` and ``_urls`` attached.
        """
        start_year = request.start.year
        end_year = request.end.year

        snapped_bounds = _snapBounds(request.geometry.bounds, _CACHE_INFO.snap_resolution)
        logging.info(f'Building MODIS request for bounds: {snapped_bounds}, years {start_year}-{end_year}')

        if task_id is None:
            task_id = self._constructRequest(
                snapped_bounds, start_year, end_year, request.variables)

        request.task_id = task_id
        request._urls = {}
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Poll AppEEARS to check whether the task is complete and bundle is available.

        Parameters
        ----------
        request : ManagerDataset.Request
            MODIS request object with AppEEARS task information.

        Returns
        -------
        bool
            True if data is ready for download, False otherwise.
        """
        status = self._checkStatus(request)
        if status != False:  # note: matches True or 'UNKNOWN'
            ready = self._checkBundleURL(request)
            return ready
        return False

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Download AppEEARS result files to the cache directory.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``task_id`` and ``_urls`` populated.
            Files are written to ``request._download_path/{var}.nc``.
        """
        assert len(request._urls) == len(request.variables), \
            "Bundle URLs must be populated before calling _downloadDataset"

        for var in request.variables:
            url = request._urls[var]
            filename = os.path.join(request._download_path, f'{var}.nc')
            if os.path.isfile(filename):
                logging.info(f'  Using existing: {filename}')
                continue
            logging.info("  Downloading: {}".format(url))
            logging.info("      to file: {}".format(filename))
            good = source_utils.download(
                url, filename, headers={ 'Authorization': f'Bearer {self.login_token}'})
            assert good

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Open cached NetCDF files and merge into a Dataset.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested MODIS data.
        """
        darrays = {}
        for var in request.variables:
            filename = os.path.join(request._download_path, f'{var}.nc')
            da = self._readFile(filename, var)
            da = da.rename({'time': f'time_{var}'})
            darrays[var] = da

        return xr.Dataset(darrays)

    def _readFile(self, filename: str, variable: str) -> xr.DataArray:
        """Open the file and get the data."""
        with xr.open_dataset(filename) as fid:
            layer = self._PRODUCTS[variable]['layer']
            data = fid[layer]

        data.name = variable
        return data
