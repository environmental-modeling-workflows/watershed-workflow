"""Manager for downloading POLARIS (30 m CONUS) soil hydraulic properties.

POLARIS provides 30-m resolution, spatially continuous, probabilistic maps of
soil hydraulic properties for the contiguous United States.  Data are
physically modelled float32 GeoTIFFs served via plain HTTP from
hydrology.cee.duke.edu.  No authentication is required.

The URL pattern for a single tile is::

    http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/
        {property}/{stat}/{depth}/lat{south}{north}_lon{west}{east}.tif

where tiles are 1 x 1 degree and the filename encodes both edges, e.g.
``lat3536_lon-84-83`` covers 35N-36N, 84W-83W.

Available properties
--------------------
Texture : clay, sand, silt, om
Hydraulic : ksat (log10 cm/hr), alpha (log10 cm^-1), n, theta_r, theta_s
Other : bd, ph, hb

Units note: ``ksat`` and ``alpha`` are stored as log10 values in
``log10(cm/hr)`` and ``log10(cm^-1)`` respectively.  All other variables are
in their conventional units (fractions, g/cm^3, or dimensionless).

Stats : mean, p5, p50, p95

Depths : 0_5, 5_15, 15_30, 30_60, 60_100, 100_200 (cm)

Data license: CC BY-NC 4.0.

References: [Chaney2019]_

.. [Chaney2019] Chaney, N.W., et al. (2019). POLARIS soil properties: 30-m
   probabilistic maps of soil properties over the contiguous United States.
   *Water Resources Research*, 55, 2916-2938.
   https://doi.org/10.1029/2018WR022797
"""
import io
import math
import os
import logging

import numpy as np
import requests
import rasterio
import rasterio.io
import rasterio.merge
import rasterio.transform
import xarray as xr

import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import CacheInfo, _snapBounds


_CACHE_INFO = CacheInfo(
    category='soil_structure',
    subcategory='polaris',
    name='polaris',
    snap_resolution=0.1,
)


_BASE_URL = (
    'http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0'
    '/{property}/{stat}/{depth}/lat{south}{north}_lon{west}{east}.tif'
)

# Six GlobalSoilMap-aligned depth intervals and their centre depths [m].
_DEPTH_LABELS = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']
_DEPTH_LABEL_PRETTY = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
_DEPTH_CENTRES = [0.025, 0.10, 0.225, 0.45, 0.80, 1.50]

_VALID_STATS = {'mean', 'p5', 'p50', 'p95'}

_VALID_VARIABLES = [
    'clay', 'sand', 'silt', 'om',
    'ksat', 'alpha', 'n', 'theta_r', 'theta_s',
    'bd', 'ph', 'hb',
]

_DEFAULT_VARIABLES = ['theta_s', 'theta_r', 'alpha', 'n', 'ksat']

_UNITS = {
    'clay':    '%',
    'sand':    '%',
    'silt':    '%',
    'om':      'g/kg',
    'ksat':    'log10(cm/hr)',
    'alpha':   'log10(cm^-1)',
    'n':       '-',
    'theta_r': 'cm^3/cm^3',
    'theta_s': 'cm^3/cm^3',
    'bd':      'g/cm^3',
    'ph':      '-',
    'hb':      'cm',
}

_LONG_NAMES = {
    'clay':    'Clay content',
    'sand':    'Sand content',
    'silt':    'Silt content',
    'om':      'Organic matter content',
    'ksat':    'Saturated hydraulic conductivity (log10 scale)',
    'alpha':   'van Genuchten alpha (log10 scale)',
    'n':       'van Genuchten n',
    'theta_r': 'Residual volumetric water content',
    'theta_s': 'Saturated volumetric water content (porosity)',
    'bd':      'Bulk density',
    'ph':      'Soil pH',
    'hb':      'Bubbling pressure',
}


@cached_dataset_manager(_CACHE_INFO)
class ManagerPOLARIS(manager_dataset.ManagerDataset):
    """POLARIS 30-m CONUS soil hydraulic properties manager.

    Downloads soil properties from the POLARIS v1.0 dataset [Chaney2019]_ served
    by Duke University.  Data are returned as a 3-D ``xr.Dataset`` with dimensions
    ``(depth, y, x)`` where ``depth`` holds the centre depth in metres for
    each of the six soil layers (0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm).

    A separate cache file is written for each ``{property}_{stat}`` combination,
    e.g. ``POLARIS_theta_s_mean_...nc``.  All six depth layers are stacked into
    that single file.

    POLARIS covers the contiguous United States (CONUS) only.

    Available variables
    -------------------
    clay, sand, silt, om, ksat, alpha, n, theta_r, theta_s, bd, ph, hb

    Default variables: ``theta_s``, ``theta_r``, ``alpha``, ``n``, ``ksat``
    (the van Genuchten parameter set plus saturated hydraulic conductivity).

    Notes
    -----
    ``ksat`` is stored as log10(cm/hr) and ``alpha`` as log10(cm^-1) -- these
    are the native POLARIS units, preserved as-is in the cache.

    .. [Chaney2019] Chaney, N.W., et al. (2019). POLARIS soil properties: 30-m
       probabilistic maps of soil properties over the contiguous United States.
       *Water Resources Research*, 55, 2916-2938.
       https://doi.org/10.1029/2018WR022797
    """

    VALID_VARIABLES = _VALID_VARIABLES
    DEFAULT_VARIABLES = _DEFAULT_VARIABLES

    def __init__(self, stat: str = 'mean', force_download: bool = False):
        """Initialize the POLARIS manager.

        Parameters
        ----------
        stat : str, optional
            Statistical summary to retrieve.  One of ``'mean'``, ``'p5'``,
            ``'p50'``, ``'p95'``.  Default ``'mean'``.
        force_download : bool, optional
            Re-download data even when a valid cached file already exists.
        """
        if stat not in _VALID_STATS:
            raise ValueError(
                f"Invalid stat '{stat}'. Valid stats: {', '.join(sorted(_VALID_STATS))}"
            )
        super().__init__(
            name='POLARIS',
            source='http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0',
            native_resolution=0.000278,   # ~30 m in degrees
            native_crs_in=CRS.from_epsg(4326),
            native_crs_out=CRS.from_epsg(4326),
            native_start=None,
            native_end=None,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
        )
        self.stat = stat
        self.force_download = force_download

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

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
            True if ``{var}_{stat}.nc`` exists for every requested variable.
        """
        for var in request.variables:
            if not os.path.isfile(os.path.join(dir, f'{var}_{self.stat}.nc')):
                return False
        return True

    def _requestDataset(
        self,
        request: manager_dataset.ManagerDataset.Request,
    ) -> manager_dataset.ManagerDataset.Request:
        """Return the request unchanged — no async step."""
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — POLARIS HTTP server is synchronous."""
        return True

    def _downloadDataset(
        self,
        request: manager_dataset.ManagerDataset.Request,
    ) -> None:
        """Download each requested variable to the cache directory."""
        snapped_bounds = _snapBounds(request.geometry.bounds, _CACHE_INFO.snap_resolution)
        for var in request.variables:
            fname = os.path.join(request._download_path, f'{var}_{self.stat}.nc')
            self._download(var, snapped_bounds, fname)

    def _loadDataset(
        self,
        request: manager_dataset.ManagerDataset.Request,
    ) -> xr.Dataset:
        """Open cached NetCDF files and merge into a single dataset.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set.

        Returns
        -------
        xr.Dataset
            Merged dataset with all requested variables, each shaped
            ``(depth, y, x)``.
        """
        datasets = []
        for var in request.variables:
            path = os.path.join(request._download_path, f'{var}_{self.stat}.nc')
            datasets.append(xr.open_dataset(path))
        return xr.merge(datasets, compat='override')

    # ------------------------------------------------------------------
    # Download internals
    # ------------------------------------------------------------------

    def _download(self, var: str, snapped_bounds: tuple, filename: str) -> None:
        """Download all depth layers for one variable and save as NetCDF.

        Parameters
        ----------
        var : str
            POLARIS property name (e.g. ``'theta_s'``).
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` in WGS84 (EPSG:4326).
        filename : str
            Destination NetCDF path.
        """
        if os.path.exists(filename) and not self.force_download:
            logging.info(f'  Using existing: {filename}')
            return

        xmin, ymin, xmax, ymax = snapped_bounds

        # Determine which 1-degree tiles cover the bounding box.
        tile_wests  = list(range(math.floor(xmin), math.ceil(xmax)))
        tile_souths = list(range(math.floor(ymin), math.ceil(ymax)))
        n_tiles = len(tile_wests) * len(tile_souths)

        logging.info(
            f'  POLARIS: downloading {var}/{self.stat} '
            f'over {n_tiles} tile(s)'
        )

        layers = []
        final_transform = None
        final_height = None
        final_width = None

        for depth_label in _DEPTH_LABELS:
            tile_arrays = []
            tile_transforms = []
            tile_profile = {}

            for tile_south in tile_souths:
                tile_north = tile_south + 1
                for tile_west in tile_wests:
                    tile_east = tile_west + 1
                    raw_data = self._downloadTile(
                        var, self.stat, depth_label,
                        tile_south, tile_north, tile_west, tile_east,
                    )
                    if raw_data is None:
                        continue
                    with rasterio.open(io.BytesIO(raw_data)) as src:
                        arr = src.read(1).astype(np.float32)
                        nodata = src.nodata
                        transform = src.transform
                        tile_profile = dict(src.profile)
                    if nodata is not None:
                        arr[arr == nodata] = np.nan
                    tile_arrays.append(arr)
                    tile_transforms.append(transform)

            if not tile_arrays:
                raise RuntimeError(
                    f'POLARIS: no tiles downloaded for {var}/{self.stat}/{depth_label} '
                    f'bounds {snapped_bounds}'
                )

            if len(tile_arrays) == 1:
                merged_arr = tile_arrays[0]
                merged_transform = tile_transforms[0]
            else:
                merged_arr, merged_transform = self._mosaicTiles(
                    tile_arrays, tile_transforms, tile_profile
                )

            final_transform = merged_transform
            final_height, final_width = merged_arr.shape
            layers.append(merged_arr)

        # Build coordinate arrays from the affine transform.
        xs = final_transform.c + final_transform.a * (np.arange(final_width) + 0.5)
        ys = final_transform.f + final_transform.e * (np.arange(final_height) + 0.5)

        depth_coords = np.array(_DEPTH_CENTRES, dtype=np.float32)
        data = np.stack(layers, axis=0)   # (depth, y, x)

        ds = xr.Dataset({
            var: xr.DataArray(
                data,
                dims=['depth', 'y', 'x'],
                coords={'depth': depth_coords, 'y': ys, 'x': xs},
                attrs={
                    'long_name': _LONG_NAMES[var],
                    'units': _UNITS[var],
                    'stat': self.stat,
                    'depth_labels': ', '.join(_DEPTH_LABEL_PRETTY),
                },
            )
        })
        ds = ds.rio.write_crs(self.native_crs_out)
        ds.attrs['source'] = self.source
        ds.attrs['variable'] = var
        ds.attrs['stat'] = self.stat

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ds.to_netcdf(filename)
        logging.info(f'    Written to: {filename}')

    def _downloadTile(
        self,
        prop: str,
        stat: str,
        depth: str,
        south: int,
        north: int,
        west: int,
        east: int,
    ) -> 'bytes | None':
        """Download a single POLARIS 1-degree tile and return raw bytes.

        Returns ``None`` if the tile does not exist (HTTP 404).
        """
        south_str = f'{south:02d}'
        north_str = f'{north:02d}'
        west_str  = str(west)
        east_str  = str(east)

        url = _BASE_URL.format(
            property=prop,
            stat=stat,
            depth=depth,
            south=south_str,
            north=north_str,
            west=west_str,
            east=east_str,
        )
        logging.info(f'      GET {url}')
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            logging.warning(f'      POLARIS tile not found (404): {url}')
            return None
        resp.raise_for_status()
        return resp.content

    def _mosaicTiles(
        self,
        arrays: list,
        transforms: list,
        profile: dict,
    ) -> tuple:
        """Merge multiple tile arrays into a single mosaic.

        Parameters
        ----------
        arrays : list of np.ndarray
            2-D float32 arrays (one per tile).
        transforms : list of Affine
            Affine transforms corresponding to each tile.
        profile : dict
            Rasterio profile used as a template for in-memory datasets.

        Returns
        -------
        tuple of (np.ndarray, Affine)
            The merged 2-D array and its affine transform.
        """
        datasets = []
        mem_files = []
        for arr, tf in zip(arrays, transforms):
            h, w = arr.shape
            mem_profile = profile.copy()
            mem_profile.update(
                driver='GTiff',
                height=h,
                width=w,
                count=1,
                dtype='float32',
                transform=tf,
                nodata=np.nan,
            )
            mf = rasterio.io.MemoryFile()
            mem_files.append(mf)
            with mf.open(**mem_profile) as dst:
                dst.write(arr[np.newaxis, :, :])
            datasets.append(mf.open())

        merged, merged_transform = rasterio.merge.merge(
            datasets, nodata=np.nan, method='first'
        )
        for ds in datasets:
            ds.close()
        for mf in mem_files:
            mf.close()

        return merged[0], merged_transform

