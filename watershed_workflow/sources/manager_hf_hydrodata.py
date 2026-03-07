"""Manager for interacting with HydroFrame HF-Hydrodata CONUS2 subsurface datasets.

Provides access to CONUS2 subsurface structure and properties via the
``hf_hydrodata`` package from the HydroFrame project.  The primary product is
the ``pf_indicator`` 3-D integer grid (x, y, z layers) which encodes
hydrogeologic units.  Associated per-cell properties (porosity, permeability,
van Genuchten parameters, etc.) are also available as 3-D gridded fields.

Registration
------------
An HydroFrame account and a short-lived 4-digit PIN are required.  Generate a
PIN at https://hydrogen.princeton.edu/pin, then register once per session::

    import watershed_workflow.sources.manager_hf_hydrodata as hfhd
    hfhd.register_pin('user@example.com', '1234')

Or add to ``~/.watershed_workflowrc``::

    [HFHydrodata]
    email = user@example.com
    pin = 1234

The PIN expires after some number of days of non-use and must be reacquired.
"""
from typing import List, Optional

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd

import watershed_workflow.config
import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset


# CONUS2 Lambert Conformal Conic CRS (spherical Earth, r=6370000 m).
# Parameters extracted from the hf_hydrodata data model grid table.
_CONUS2_CRS = CRS.from_proj4(
    '+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40.0000076294444 +lon_0=-97 '
    '+x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs'
)


def register_pin(email: str, pin: str) -> None:
    """Register a HydroFrame PIN for hf_hydrodata access.

    Parameters
    ----------
    email : str
        HydroFrame account email address.
    pin : str
        4-digit PIN from https://hydrogen.princeton.edu/pin.

    Notes
    -----
    The PIN expires after 2 days of non-use and must be regenerated at the
    HydroFrame PIN page.  Call this function again with the new PIN after
    regeneration.
    """
    import hf_hydrodata as hf
    hf.register_api_pin(email, pin)


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
        super().__init__(
            name='HydroFrame HF-Hydrodata CONUS2 Subsurface',
            source='https://hf-hydrodata.readthedocs.io/',
            native_resolution=0.009,           # ~1 km in degrees (lat/lon)
            native_crs_in=CRS.from_epsg(4326), # lat/lon — matches API latlng_bounds
            native_crs_out=_CONUS2_CRS,        # CONUS2 LCC output
            native_start=None,                 # static dataset
            native_end=None,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
            cache_category='soil_structure',
            cache_extension='nc',
            has_varname=True,     # one cache file per variable
            short_name='CONUS2',
        )
        self.force_download = force_download
        os.makedirs(self._cacheFolder(), exist_ok=True)

    def _prerequestDataset(self) -> None:
        """Register PIN with hf_hydrodata if credentials are configured."""
        email = watershed_workflow.config.rcParams['HFHydrodata']['email']
        pin = watershed_workflow.config.rcParams['HFHydrodata']['pin']
        if email == 'NOT_PROVIDED' or pin == 'NOT_PROVIDED':
            raise ValueError(
                "HFHydrodata credentials not set.  Add an [HFHydrodata] section "
                "with 'email' and 'pin' keys to ~/.watershed_workflowrc, or call "
                "watershed_workflow.sources.manager_hf_hydrodata.register_pin(email, pin) "
                "before requesting data."
            )
        import hf_hydrodata as hf
        hf.register_api_pin(email, pin)

    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        ) -> manager_dataset.ManagerDataset.Request:
        """Download each requested variable to the cache if not already present.

        Parameters
        ----------
        request : ManagerDataset.Request
            Pre-processed request with geometry, snapped bounds, and variables.

        Returns
        -------
        ManagerDataset.Request
            The same request with ``is_ready`` set to ``True``.
        """
        for var in request.variables:
            fname = self._cacheFilename(request.snapped_bounds, var=var)
            cached = self._checkCache(request.geometry.bounds,
                                      request.snapped_bounds, var=var)
            if cached is None or self.force_download:
                self._download(var, request.snapped_bounds, fname)
        request.is_ready = True
        return request

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

        # Build xarray Dataset
        if nz is not None:
            z_coords = np.arange(nz, dtype=np.int32)
            ds = xr.Dataset(
                {var: xr.DataArray(
                    data.astype(np.float32),
                    dims=['z', 'y', 'x'],
                    coords={'z': z_coords, 'y': y_coords, 'x': x_coords},
                )},
            )
        else:
            ds = xr.Dataset(
                {var: xr.DataArray(
                    data.astype(np.float32),
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

    def _fetchDataset(self,
                      request: manager_dataset.ManagerDataset.Request,
                      ) -> xr.Dataset:
        """Open cached NetCDF files and merge into a single Dataset.

        Parameters
        ----------
        request : ManagerDataset.Request
            Ready request with snapped bounds and variable list.

        Returns
        -------
        xr.Dataset
            Merged dataset containing all requested variables.
        """
        datasets = []
        for var in request.variables:
            fname = self._cacheFilename(request.snapped_bounds, var=var)
            cached = self._checkCache(request.geometry.bounds,
                                      request.snapped_bounds, var=var)
            path = cached if cached is not None else fname
            datasets.append(xr.open_dataset(path))
        return xr.merge(datasets, compat='override')

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
