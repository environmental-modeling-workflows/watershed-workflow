"""Manager for interacting with HydroFrame HF-Hydrodata CONUS2 subsurface datasets.

Provides access to CONUS2 subsurface structure and properties via the
``hf_hydrodata`` package from the HydroFrame project.  The primary product is
the ``pf_indicator`` 3-D integer grid (x, y, z layers) which encodes
hydrogeologic units.  Associated per-cell properties (porosity, permeability,
van Genuchten parameters, etc.) are also available as 3-D gridded fields.

Registration
------------
An HydroFrame account and a short-lived 4-digit PIN are required.  Generate a
PIN at https://hydrogen.princeton.edu/pin, then store credentials in
``~/.netrc``::

    machine hydrogen.princeton.edu
    login user@example.com
    password 1234

The PIN expires after some number of days of non-use and must be reacquired.
Once stored in ``~/.netrc``, registration happens automatically on first use.
"""
from typing import List, Optional

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd

import netrc
import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset
from .manager import ManagerAttributes
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import snapBounds


# CONUS2 Lambert Conformal Conic CRS (spherical Earth, r=6370000 m).
# Parameters extracted from the hf_hydrodata data model grid table.
_CONUS2_CRS = CRS.from_proj4(
    '+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40.0000076294444 +lon_0=-97 '
    '+x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs'
)


def registerPin(email: str, pin: str) -> None:
    """Register a HydroFrame PIN for hf_hydrodata access.

    Parameters
    ----------
    email : str
        HydroFrame account email address.
    pin : str
        4-digit PIN from https://hydrogen.princeton.edu/pin.

    Notes
    -----
    Credentials are read automatically from ``~/.netrc`` (machine
    ``hydrogen.princeton.edu``) on first use.  Call this function only to
    force re-registration (e.g. after acquiring a new PIN).
    """
    import hf_hydrodata as hf
    hf.register_api_pin(email, pin)


@cached_dataset_manager
class ManagerHFHydrodata(manager_dataset.ManagerDataset):
    """HydroFrame HF-Hydrodata CONUS2 subsurface dataset manager.

    Provides access to the CONUS2 1-km gridded subsurface structure and
    property fields from the HydroFrame project via the ``hf_hydrodata``
    package [HFHydro]_.

    The primary field is ``pf_indicator``, a 3-D integer grid (z, y, x) where
    each integer maps to a hydrogeologic unit.  Associated per-cell property
    fields (porosity, permeability, van Genuchten parameters, etc.) are also
    available as static 3-D grids.

    Default variables: ``pf_indicator``, ``pf_flowbarrier``.

    All fields are on the CONUS2 Lambert Conformal Conic 1-km grid.

    .. [HFHydro] https://hf-hydrodata.readthedocs.io/
    """

    # Valid variables in the conus2_domain dataset that this manager exposes.
    VALID_VARIABLES = [
        'pf_indicator',
        'pf_flowbarrier',
        'porosity',
        'permeability_x',
        'permeability_y',
        'permeability_z',
        'vg_alpha',
        'vg_n',
        'sres',
        'ssat',
        'specific_storage',
        'slope_x',
        'slope_y',
        'mannings',
        'pme',
        'ss_pressure_head',
        'elevation',
    ]
    DEFAULT_VARIABLES = ['pf_indicator']

    def __init__(self, force_download: bool = False):
        """Initialize the HF-Hydrodata CONUS2 manager.

        Parameters
        ----------
        force_download : bool, optional
            If ``True``, re-download data even when a cached file already exists.
        """
        attrs = ManagerAttributes(
            category='soil_structure',
            product='ParFlow CONUS2',
            source='HydroFrame HF-Hydrodata',
            description='CONUS2 1-km gridded subsurface structure and property fields from HydroFrame.',
            product_short='parflow_conus2',
            source_short='hydroframe_hf_hydrodata',
            url='https://hf-hydrodata.readthedocs.io/',
            license=None,
            citation='Maxwell et al.',
            native_crs_in=CRS.from_epsg(4326),  # lat/lon — matches API latlng_bounds
            native_crs_out=_CONUS2_CRS,          # CONUS2 LCC output
            native_resolution=0.009,             # ~1 km in degrees (lat/lon)
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
        )
        super().__init__(attrs)
        self.force_download = force_download

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

    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        ) -> manager_dataset.ManagerDataset.Request:
        """Register PIN with hf_hydrodata from ~/.netrc if not already registered.

        Parameters
        ----------
        request : ManagerDataset.Request
            Pre-processed request with geometry and variables.

        Returns
        -------
        ManagerDataset.Request
            The same request, unchanged.
        """
        pin_path = os.path.expanduser('~/.hydrodata/pin.json')
        if not os.path.exists(pin_path):
            try:
                creds = netrc.netrc().authenticators('hydrogen.princeton.edu')
            except (FileNotFoundError, netrc.NetrcParseError):
                creds = None
            if creds is None:
                raise ValueError(
                    "HFHydrodata credentials not found.  Add an entry to ~/.netrc:\n\n"
                    "    machine hydrogen.princeton.edu\n"
                    "    login user@example.com\n"
                    "    password 1234\n\n"
                    "Or call watershed_workflow.sources.manager_hf_hydrodata.registerPin(email, pin)."
                )
            email, _, pin = creds
            import hf_hydrodata as hf
            hf.register_api_pin(email, pin)
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — HF-Hydrodata downloads are synchronous."""
        return True

    def _downloadDataset(self,
                         request: manager_dataset.ManagerDataset.Request,
                         ) -> None:
        """Download each requested variable to the cache directory.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set. Files are written to
            ``request._download_path/{var}.nc`` for each variable.
        """
        snapped_bounds = snapBounds(request.geometry.bounds, self.native_resolution)
        for var in request.variables:
            fname = os.path.join(request._download_path, f'{var}.nc')
            self._download(var, snapped_bounds, fname)

    def _download(self, var: str, snapped_bounds: tuple, filename: str) -> None:
        """Download one variable from hf_hydrodata and save as a NetCDF file.

        Parameters
        ----------
        var : str
            Variable name from the ``conus2_domain`` dataset.
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` in WGS84 lat/lon (EPSG:4326).
        filename : str
            Destination NetCDF file path.
        """
        import hf_hydrodata as hf

        if os.path.exists(filename) and not self.force_download:
            logging.info(f'  Using existing: {filename}')
            return

        logging.info(f'  Downloading {var} from HF-Hydrodata conus2_domain')
        logging.info(f'    snapped_bounds (lon_min, lat_min, lon_max, lat_max): {snapped_bounds}')

        lon_min, lat_min, lon_max, lat_max = snapped_bounds
        # hf_hydrodata latlng_bounds: [lat_min, lon_min, lat_max, lon_max]
        options = {
            'dataset': 'conus2_domain',
            'variable': var,
            'latlng_bounds': [lat_min, lon_min, lat_max, lon_max],
        }

        data = hf.get_gridded_data(options)   # numpy ndarray, shape (z, y, x) or (y, x)
        logging.info(f'    downloaded array shape: {data.shape}')

        # Compute CONUS2 LCC coordinates for the downloaded subset.
        # from_latlon(grid, lat_min, lon_min, lat_max, lon_max) returns
        # [x_left, y_bottom, x_right, y_top] in grid-index units (0 = SW corner).
        grid_bounds = hf.from_latlon('conus2', lat_min, lon_min, lat_max, lon_max)
        x_left_idx, y_bottom_idx, x_right_idx, y_top_idx = grid_bounds

        # Grid indices are 0-based; data array is (ny, nx) or (nz, ny, nx).
        if data.ndim == 3:
            nz, ny, nx = data.shape
        else:
            ny, nx = data.shape
            nz = None

        # Centre-of-cell LCC coordinates in metres.
        # Grid index i → LCC x = origin_x + (i + 0.5) * 1000 m
        # CONUS2 grid origin in LCC metres: (-2208000.30881173, -1668999.65483222).
        _res = 1000.0  # CONUS2 grid resolution in metres
        x_origin = -2208000.30881173
        y_origin = -1668999.65483222
        x_start = x_origin + (int(x_left_idx) + 0.5) * _res
        y_start = y_origin + (int(y_bottom_idx) + 0.5) * _res
        x_coords = x_start + np.arange(nx) * _res
        y_coords = y_start + np.arange(ny) * _res

        # Preserve integer dtype for categorical fields so that spatial
        # resampling uses 'nearest' rather than interpolating indicator IDs.
        out_dtype = np.int32 if var in self._INTEGER_VARS else np.float32

        # Build xarray Dataset
        if nz is not None:
            z_coords = np.arange(nz, dtype=np.int32)
            ds = xr.Dataset(
                {var: xr.DataArray(
                    data.astype(out_dtype),
                    dims=['z', 'y', 'x'],
                    coords={'z': z_coords, 'y': y_coords, 'x': x_coords},
                )},
            )
        else:
            ds = xr.Dataset(
                {var: xr.DataArray(
                    data.astype(out_dtype),
                    dims=['y', 'x'],
                    coords={'y': y_coords, 'x': x_coords},
                )},
            )

        ds = ds.rio.write_crs(self.native_crs_out)
        ds.attrs['source'] = self.source
        ds.attrs['dataset'] = 'conus2_domain'
        ds.attrs['variable'] = var

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ds.to_netcdf(filename)
        logging.info(f'    written to: {filename}')

    # Categorical (indicator) variables that must be kept as integers so that
    # spatial resampling uses 'nearest' rather than interpolating IDs.
    _INTEGER_VARS = {'pf_indicator', 'pf_flowbarrier'}

    def _loadDataset(self,
                     request: manager_dataset.ManagerDataset.Request,
                     ) -> xr.Dataset:
        """Open cached NetCDF files and merge into a single Dataset.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set.

        Returns
        -------
        xr.Dataset
            Merged dataset containing all requested variables.
        """
        datasets = []
        for var in request.variables:
            path = os.path.join(request._download_path, f'{var}.nc')
            datasets.append(xr.open_dataset(path))
        ds = xr.merge(datasets, compat='override')

        # Re-cast categorical variables to int32.  Older cached files may have
        # stored them as float32; ensure we always load with the correct dtype
        # so that _spatialResamplingMethod selects 'nearest' for these fields.
        for var in self._INTEGER_VARS:
            if var in ds:
                ds[var] = ds[var].astype(np.int32)
        return ds

    def getIndicatorTable(self) -> pd.DataFrame:
        """Return the CONUS2 subsurface indicator property lookup table.

        Reads a bundled CSV file mapping each integer indicator value to its
        hydrogeologic unit properties.

        Returns
        -------
        table : pd.DataFrame
            Table indexed by ``indicator`` (int) with columns: ``porosity``,
            ``permeability_x``, ``permeability_y``, ``permeability_z``,
            ``vg_alpha``, ``vg_n``, ``sres``, ``specific_storage``.

        Notes
        -----
        The bundled table contains representative values from published CONUS2
        parameterizations.  For simulation use, verify values against the
        per-cell property grids available via ``getDataset`` (variables
        ``porosity``, ``permeability_x``, etc.).
        """
        data_file = os.path.join(
            os.path.dirname(__file__), '..', 'data',
            'conus2_indicator_properties.csv',
        )
        return pd.read_csv(data_file, comment='#', index_col='indicator')
