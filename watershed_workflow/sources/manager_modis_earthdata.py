"""Manager for downloading MODIS products synchronously via NASA earthaccess.

Uses two download strategies depending on product availability:

- **OPeNDAP** (``pydap`` + earthaccess session): for products that expose a
  cloud OPeNDAP endpoint in their granule metadata (currently MCD12Q1 LULC).
  Only the requested spatial subset is transferred — very efficient.

- **HDF4 full-tile download** (``pyhdf`` + ``earthaccess.download``): for
  products that do not yet have a cloud OPeNDAP endpoint (currently MCD15A3H
  LAI).  Granules are processed one at a time so the staging area never
  exceeds one tile (~8 MB) at a time.

Authentication
--------------
Uses ``earthaccess.login(strategy='netrc')``, which reads ``~/.netrc``.
Register at https://urs.earthdata.nasa.gov and add::

    machine urs.earthdata.nasa.gov login <user> password <pass>

Products
--------
- LAI:  MCD15A3H v061, 4-day composite, subdataset ``Lai_500m``, scale 0.1
        — HDF4 full-tile download (no OPeNDAP yet)
- LULC: MCD12Q1  v061, annual,           subdataset ``LC_Type1``,  scale 1.0
        — OPeNDAP spatial subset

Coordinate system
-----------------
CMR bounding-box queries are issued in WGS84 (EPSG:4326), so
``native_crs_in`` is WGS84.  All downloaded data is in the MODIS sinusoidal
projection; ``native_crs_out`` is sinusoidal.  Spatial clipping and optional
reprojection to ``out_crs`` are handled by the base-class
``_postprocessDataset``.
"""
from __future__ import annotations

import math
import os
import re
import logging
import tempfile
import datetime
from typing import Dict, List, Optional

import numpy as np
import rasterio
import rasterio.io
import rasterio.merge
import rasterio.transform
import pyproj
import xarray as xr
import cftime

import watershed_workflow.crs

from . import manager_dataset
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import CacheInfo, _snapBounds


# ---------------------------------------------------------------------------
# Module-level CRS objects and transformer
# ---------------------------------------------------------------------------

#: WGS84 geographic CRS used for CMR bounding-box queries.
_WGS84_CRS = watershed_workflow.crs.from_epsg(4326)

#: MODIS sinusoidal CRS (sphere, a=b=6 371 007.181 m).
_MODIS_SINU_CRS = watershed_workflow.crs.from_string(
    '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 '
    '+a=6371007.181 +b=6371007.181 +units=m +no_defs'
)

#: Pyproj Transformer from WGS84 geographic to MODIS sinusoidal.
_WGS84_TO_SINU = pyproj.Transformer.from_crs(
    'EPSG:4326', _MODIS_SINU_CRS.to_wkt(), always_xy=True
)


_CACHE_INFO = CacheInfo(
    category='land_cover',
    subcategory='modis_earthdata',
    name='modis_earthdata',
    snap_resolution=0.01,
    is_temporal=True,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Per-variable product metadata.
#: ``opendap`` — True if cloud OPeNDAP is available for this product.
_PRODUCTS: Dict[str, Dict] = {
    'LAI': {
        'product': 'MCD15A3H',
        'version': '061',
        'sds_name': 'Lai_500m',
        'scale_factor': 0.1,
        'fill_value': 255,
        'dtype': np.float32,
        'long_name': 'Leaf Area Index',
        'units': 'm^2/m^2',
        'opendap': False,   # not yet available at LP DAAC cloud
    },
    'LULC': {
        'product': 'MCD12Q1',
        'version': '061',
        'sds_name': 'LC_Type1',
        'scale_factor': 1.0,
        'fill_value': 255,
        'dtype': np.int16,
        'long_name': 'Land Cover Type 1 (IGBP)',
        'units': 'class',
        'opendap': True,    # cloud OPeNDAP available
    },
}

#: Regex that extracts year + day-of-year from an HDF4 granule filename.
#: Example: ``MCD15A3H.A2020001.h11v05.061.2020010040000.hdf``
_HDF_DATE_RE = re.compile(r'\.A(\d{4})(\d{3})\.')

# Colour table (same product as the AppEEARS manager).
_COLORS = {
    -1: ('Unclassified', (0, 0, 0)),
    0:  ('Open Water', (140, 219, 255)),
    1:  ('Evergreen Needleleaf Forests', (38, 115, 0)),
    2:  ('Evergreen Broadleaf Forests', (82, 204, 77)),
    3:  ('Deciduous Needleleaf Forests', (150, 196, 20)),
    4:  ('Deciduous Broadleaf Forests', (122, 250, 166)),
    5:  ('Mixed Forests', (137, 205, 102)),
    6:  ('Closed Shrublands', (215, 158, 158)),
    7:  ('Open Shrublands', (255, 240, 196)),
    8:  ('Woody Savannas', (233, 255, 190)),
    9:  ('Savannas', (255, 216, 20)),
    10: ('Grasslands', (255, 196, 120)),
    11: ('Permanent Wetlands', (0, 132, 168)),
    12: ('Croplands', (255, 255, 115)),
    13: ('Urban and Built up lands', (255, 0, 0)),
    14: ('Cropland Natural Vegetation Mosaics', (168, 168, 0)),
    15: ('Permanent Snow and Ice', (255, 255, 255)),
    16: ('Barren Land', (130, 130, 130)),
    17: ('Water Bodies', (140, 209, 245)),
}

colors: Dict[int, tuple] = {
    k: (v[0], tuple(float(c) / 255.0 for c in v[1]))
    for k, v in _COLORS.items()
}
indices: Dict[str, int] = {pars[0]: idx for idx, pars in colors.items()}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parseHdfDate(hdf_filename: str) -> datetime.date:
    """Extract the acquisition date from a MODIS HDF4 granule filename.

    Parameters
    ----------
    hdf_filename : str
        Bare filename or full path, e.g.
        ``MCD15A3H.A2020001.h11v05.061.2020010040000.hdf``.

    Returns
    -------
    datetime.date
        Acquisition date derived from the embedded year + day-of-year.
    """
    basename = os.path.basename(hdf_filename)
    m = _HDF_DATE_RE.search(basename)
    if m is None:
        raise ValueError(
            f'Cannot parse acquisition date from HDF filename: {basename!r}'
        )
    year = int(m.group(1))
    doy = int(m.group(2))
    return datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)


def _granuleDate(granule) -> datetime.date:
    """Extract acquisition date from an earthaccess DataGranule."""
    begin = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
    # format: '2010-01-01T00:00:00.000Z'
    return datetime.date.fromisoformat(begin[:10])


def _opendapUrl(granule) -> Optional[str]:
    """Return the cloud OPeNDAP URL for a granule, or None if unavailable."""
    for u in granule['umm'].get('RelatedUrls', []):
        if u.get('Subtype') == 'OPENDAP DATA':
            return u['URL']
    return None


def _readHdfSds(hdf_path: str, sds_name: str) -> tuple:
    """Open an HDF4 file and read a named SDS.

    Returns ``(array, transform)`` where ``array`` is the raw 2-D
    ``float64`` pixel values and ``transform`` is the affine mapping from
    pixel space to MODIS sinusoidal metres (upper-left corner origin).
    """
    from pyhdf.SD import SD, SDC  # type: ignore[import]

    hdf = SD(hdf_path, SDC.READ)
    try:
        sds = hdf.select(sds_name)
        array = sds.get().astype(np.float64)
        meta_attr = hdf.attributes().get('StructMetadata.0', '')
        ul_x, ul_y, lr_x, lr_y = _parseStructMetadata(meta_attr)
    finally:
        hdf.end()

    nrows, ncols = array.shape
    pixel_w = (lr_x - ul_x) / ncols
    pixel_h = (ul_y - lr_y) / nrows

    transform = rasterio.transform.Affine(
        pixel_w, 0.0, ul_x,
        0.0, -pixel_h, ul_y,
    )
    return array, transform


def _parseStructMetadata(meta: str) -> tuple:
    """Parse ULCorner and LRCorner from StructMetadata.0 text.

    Returns ``(ul_x, ul_y, lr_x, lr_y)`` in MODIS sinusoidal metres.
    """
    ul_x = float(re.search(r'UpperLeftPointMtrs\s*=\s*\(\s*(-?[\d.eE+\-]+)', meta).group(1))
    ul_y = float(re.search(r'UpperLeftPointMtrs\s*=\s*\([^,]+,\s*(-?[\d.eE+\-]+)', meta).group(1))
    lr_x = float(re.search(r'LowerRightMtrs\s*=\s*\(\s*(-?[\d.eE+\-]+)', meta).group(1))
    lr_y = float(re.search(r'LowerRightMtrs\s*=\s*\([^,]+,\s*(-?[\d.eE+\-]+)', meta).group(1))
    return ul_x, ul_y, lr_x, lr_y


def _sinosBoundsFromWgs84(snapped_bounds_deg: tuple) -> tuple:
    """Convert WGS84 bounding box to MODIS sinusoidal metres.

    Parameters
    ----------
    snapped_bounds_deg : tuple of float
        ``(xmin, ymin, xmax, ymax)`` in WGS84 degrees.

    Returns
    -------
    tuple of float
        ``(xmin, ymin, xmax, ymax)`` in sinusoidal metres.
    """
    xmin_d, ymin_d, xmax_d, ymax_d = snapped_bounds_deg
    corners_x, corners_y = _WGS84_TO_SINU.transform(
        [xmin_d, xmax_d, xmin_d, xmax_d],
        [ymin_d, ymin_d, ymax_d, ymax_d],
    )
    return min(corners_x), min(corners_y), max(corners_x), max(corners_y)


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

@cached_dataset_manager(_CACHE_INFO)
class ManagerMODISEarthdata(manager_dataset.ManagerDataset):
    """MODIS LAI and LULC products via NASA earthaccess (synchronous).

    Uses cloud OPeNDAP (via ``pydap``) for products that support it, and
    direct HDF4 tile download (via ``pyhdf``) for those that do not.
    Currently:

    - **LULC** (MCD12Q1): OPeNDAP — only the requested spatial subset is
      transferred (~KB per granule).
    - **LAI** (MCD15A3H): HDF4 download — full tiles (~8 MB each), but
      processed one at a time to limit temporary disk use.

    Authentication reads ``~/.netrc`` via
    ``earthaccess.login(strategy='netrc')``.  Register at
    https://urs.earthdata.nasa.gov.

    Notes
    -----
    Data are returned in the **MODIS sinusoidal** projection
    (``+proj=sinu +a=6371007.181``).  Pass ``out_crs`` to ``getDataset``
    to reproject; the base class handles spatial clipping and reprojection.

    Each variable gets its own time dimension (``time_LAI``, ``time_LULC``)
    so that LAI (4-day composites) and LULC (annual) can coexist in the same
    ``xr.Dataset`` without broadcasting.
    """

    colors = colors
    indices = indices

    def __init__(self, force_download: bool = False):
        """Initialise the earthaccess MODIS manager.

        Parameters
        ----------
        force_download : bool, optional
            Re-download data even when a valid cached file already exists.
            Default is ``False``.
        """
        native_start = cftime.datetime(2002, 7, 4, calendar='standard')
        native_end = cftime.datetime(2024, 12, 31, calendar='standard')

        #: MODIS sinusoidal CRS (sphere, a=b=6 371 007.181 m).
        native_crs_out = _MODIS_SINU_CRS
        
        #: WGS84 geographic CRS used for CMR bounding-box queries.
        native_crs_in = watershed_workflow.crs.from_epsg(4326)

        # Native pixel spacing of 500-m MODIS products expressed in
        # native_in_crs, **degrees**. Used by the base class for
        # buffering/snapping.
        # 500 m / (pi/180 * 6 371 007.181 m/deg) ≈ 4.49e-3 deg
        native_resolution = 500.0 / (math.pi / 180.0 * 6_371_007.181)
        
        super().__init__(
            name='MODIS',
            source='earthaccess/LP DAAC',
            native_resolution=native_resolution,
            native_crs_in=native_crs_in,
            native_crs_out=native_crs_out,
            native_start=native_start,
            native_end=native_end,
            valid_variables=list(_PRODUCTS.keys()),
            default_variables=list(_PRODUCTS.keys()),
        )
        self.force_download = force_download
        self._session = None   # earthaccess HTTPS session, set on first use

    # ------------------------------------------------------------------
    # ManagerDataset abstract method implementations
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
            True if ``{var}.nc`` exists for every requested variable.
        """
        for var in request.variables:
            if not os.path.isfile(os.path.join(dir, f'{var}.nc')):
                return False
        return True

    def _requestDataset(
        self,
        request: manager_dataset.ManagerDataset.Request,
    ) -> manager_dataset.ManagerDataset.Request:
        """Authenticate and return the request.

        Parameters
        ----------
        request : ManagerDataset.Request
            Pre-processed request carrying geometry, date range, and variable list.

        Returns
        -------
        ManagerDataset.Request
            The same request, unchanged.
        """
        self._ensureAuth()
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — earthaccess downloads are synchronous."""
        return True

    def _downloadDataset(
        self,
        request: manager_dataset.ManagerDataset.Request,
    ) -> None:
        """Download each requested variable to the cache directory.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set. Files are written to
            ``request._download_path/{var}.nc`` for each variable.
        """
        start_year = request.start.year
        end_year = request.end.year
        snapped_bounds = _snapBounds(request.geometry.bounds, _CACHE_INFO.snap_resolution)

        for var in request.variables:
            target = os.path.join(request._download_path, f'{var}.nc')
            if os.path.isfile(target) and not self.force_download:
                logging.info(f'  MODIS earthaccess: using existing file for "{var}": {target}')
                continue
            self._downloadVar(var, snapped_bounds, start_year, end_year, target)

    def _loadDataset(
        self,
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
            Merged dataset with all requested variables.  Each variable has a
            per-variable time dimension (``time_LAI``, ``time_LULC``) so that
            LAI and LULC can coexist without coordinate broadcasting.
        """
        data_arrays: Dict[str, xr.DataArray] = {}
        for var in request.variables:
            path = os.path.join(request._download_path, f'{var}.nc')
            da = xr.open_dataset(path)[var]
            if 'time' in da.dims:
                da = da.rename({'time': f'time_{var}'})
            data_arrays[var] = da

        return xr.Dataset(data_arrays)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def _ensureAuth(self) -> None:
        """Authenticate once and cache the session."""
        import earthaccess  # type: ignore[import]
        if self._session is not None:
            return
        try:
            earthaccess.login(strategy='netrc')
        except Exception as e:
            raise RuntimeError(
                'NASA Earthdata authentication failed. Ensure ~/.netrc contains:\n\n'
                '    machine urs.earthdata.nasa.gov login <username> password <password>\n\n'
                'Register for a free account at https://urs.earthdata.nasa.gov\n'
                f'Original error: {e}'
            ) from e
        self._session = earthaccess.get_requests_https_session()

    # ------------------------------------------------------------------
    # Dispatch: OPeNDAP vs HDF4
    # ------------------------------------------------------------------
    def _downloadVar(
        self,
        var: str,
        snapped_bounds: tuple,
        start_year: int,
        end_year: int,
        cache_path: str,
    ) -> None:
        """Search for granules and download one variable to the cache.

        Dispatches to OPeNDAP or HDF4 based on product availability.

        Parameters
        ----------
        var : str
            Variable name, one of ``'LAI'``, ``'LULC'``.
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` in WGS84 degrees.
        start_year : int
            First year to retrieve (inclusive).
        end_year : int
            Last year to retrieve (inclusive).
        cache_path : str
            Destination NetCDF4 file path.
        """
        import earthaccess  # type: ignore[import]

        product_info = _PRODUCTS[var]
        xmin, ymin, xmax, ymax = snapped_bounds
        start_str = f'{start_year}-01-01'
        end_str = f'{end_year}-12-31'

        logging.info(
            f'  MODIS earthaccess: searching {product_info["product"]} v{product_info["version"]} '
            f'bounds={snapped_bounds} temporal={start_str}/{end_str}'
        )
        granules = earthaccess.search_data(
            short_name=product_info['product'],
            version=product_info['version'],
            bounding_box=(xmin, ymin, xmax, ymax),
            temporal=(start_str, end_str),
            count=-1,
        )
        if not granules:
            raise RuntimeError(
                f'earthaccess: no granules found for {product_info["product"]} '
                f'bounds={snapped_bounds} temporal={start_str}/{end_str}'
            )
        logging.info(f'    found {len(granules)} granule(s)')

        sinu_bounds = _sinosBoundsFromWgs84(snapped_bounds)

        if product_info['opendap']:
            self._downloadViaOpendap(var, granules, sinu_bounds, cache_path)
        else:
            self._downloadViaHdf4(var, granules, sinu_bounds, cache_path)

    # ------------------------------------------------------------------
    # OPeNDAP path (e.g. LULC / MCD12Q1)
    # ------------------------------------------------------------------

    def _downloadViaOpendap(
        self,
        var: str,
        granules: list,
        sinu_bounds: tuple,
        cache_path: str,
    ) -> None:
        """Fetch granules via cloud OPeNDAP, reading only the spatial subset.

        Parameters
        ----------
        var : str
            Variable name.
        granules : list of DataGranule
            earthaccess search results.
        sinu_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` in sinusoidal metres.
        cache_path : str
            Destination NetCDF4 file path.
        """
        product_info = _PRODUCTS[var]
        sds_name = product_info['sds_name']
        fill = product_info['fill_value']
        scale = product_info['scale_factor']
        sinu_xmin, sinu_ymin, sinu_xmax, sinu_ymax = sinu_bounds

        time_slices: List[np.ndarray] = []
        dates_out: List[datetime.date] = []
        final_transform = None

        for granule in granules:
            dap_url = _opendapUrl(granule)
            if dap_url is None:
                raise RuntimeError(
                    f'No OPeNDAP URL found for granule {granule}. '
                    f'Product may have lost cloud OPeNDAP support — set opendap=False in _PRODUCTS.'
                )
            acq_date = _granuleDate(granule)
            logging.info(f'    OPeNDAP: {acq_date} {dap_url}')

            # Use dap2:// prefix to suppress the "unable to determine protocol"
            # warning. DAP4 returns an empty dataset for LP DAAC cloud endpoints.
            dap2_url = dap_url.replace('https://', 'dap2://', 1)
            ds = xr.open_dataset(dap2_url, engine='pydap', session=self._session)
            product = _PRODUCTS[var]['product']

            xdim = ds[f'/{product}/XDim'].values   # ascending sinusoidal metres
            ydim = ds[f'/{product}/YDim'].values   # descending sinusoidal metres
            xdim_name = f'XDim_{product}'
            ydim_name = f'YDim_{product}'

            xi1 = max(0, int(np.searchsorted(xdim, sinu_xmin)) - 1)
            xi2 = min(len(xdim), int(np.searchsorted(xdim, sinu_xmax)) + 1)
            yi1 = max(0, int(np.searchsorted(-ydim, -sinu_ymax)) - 1)
            yi2 = min(len(ydim), int(np.searchsorted(-ydim, -sinu_ymin)) + 1)

            da = ds[f'/{product}/Data_Fields/{sds_name}']
            sliced = da.isel(**{ydim_name: slice(yi1, yi2), xdim_name: slice(xi1, xi2)})
            arr = sliced.values.astype(np.float64)

            arr[arr == fill] = np.nan
            mask = ~np.isnan(arr)
            arr[mask] = arr[mask] * scale

            xs_slice = xdim[xi1:xi2]
            ys_slice = ydim[yi1:yi2]
            pixel_w = xdim[1] - xdim[0]
            pixel_h = ydim[1] - ydim[0]   # negative

            transform = rasterio.transform.Affine(
                pixel_w, 0.0, xs_slice[0] - pixel_w / 2,
                0.0, pixel_h, ys_slice[0] - pixel_h / 2,
            )

            time_slices.append(arr)
            dates_out.append(acq_date)
            final_transform = transform

        if not time_slices:
            raise RuntimeError(f'MODIS earthaccess OPeNDAP: no slices assembled for {var}')

        self._writeCacheFile(var, time_slices, dates_out, final_transform, cache_path)

    # ------------------------------------------------------------------
    # HDF4 path (e.g. LAI / MCD15A3H)
    # ------------------------------------------------------------------

    def _downloadViaHdf4(
        self,
        var: str,
        granules: list,
        sinu_bounds: tuple,
        cache_path: str,
    ) -> None:
        """Download HDF4 tiles one at a time, extract, and accumulate time slices.

        Each tile is downloaded to a temporary file, read, then deleted before
        the next download, so the staging footprint is at most one tile (~8 MB).

        Parameters
        ----------
        var : str
            Variable name.
        granules : list of DataGranule
            earthaccess search results.
        sinu_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` in sinusoidal metres.
        cache_path : str
            Destination NetCDF4 file path.
        """
        import earthaccess  # type: ignore[import]

        # Group by date — multiple tiles on the same date are mosaiced.
        date_groups: Dict[datetime.date, list] = {}
        for g in granules:
            date_groups.setdefault(_granuleDate(g), []).append(g)

        sorted_dates = sorted(date_groups)
        # Flatten into an ordered list of (date, granule) pairs so we can chunk.
        ordered: List[tuple] = []
        for d in sorted_dates:
            for g in date_groups[d]:
                ordered.append((d, g))

        chunk_size = 100
        time_slices: List[np.ndarray] = []
        dates_out: List[datetime.date] = []
        final_transform = None

        for chunk_start in range(0, len(ordered), chunk_size):
            chunk = ordered[chunk_start:chunk_start + chunk_size]
            chunk_granules = [g for _, g in chunk]
            logging.info(
                f'    HDF4: downloading chunk {chunk_start//chunk_size + 1}'
                f'/{(len(ordered) - 1)//chunk_size + 1}'
                f' ({len(chunk)} granule(s))'
            )

            with tempfile.TemporaryDirectory(prefix='modis_earthdata_') as staging_dir:
                local_files = earthaccess.download(chunk_granules, local_path=staging_dir)
                # Map filename → granule date using the HDF filename.
                fname_to_date: Dict[str, datetime.date] = {}
                for (d, _), lf in zip(chunk, local_files):
                    if lf:
                        fname_to_date[str(lf)] = d

                # Group downloaded files by date and mosaic same-date tiles.
                date_files: Dict[datetime.date, List[str]] = {}
                for fpath, d in fname_to_date.items():
                    date_files.setdefault(d, []).append(fpath)

                for acq_date in sorted(date_files):
                    tile_arrays: List[np.ndarray] = []
                    tile_transforms: List[rasterio.transform.Affine] = []
                    for hdf_path in date_files[acq_date]:
                        arr, tf = self._extractHdf4Slice(hdf_path, var, sinu_bounds)
                        if arr is not None:
                            tile_arrays.append(arr)
                            tile_transforms.append(tf)

                    if not tile_arrays:
                        continue

                    if len(tile_arrays) == 1:
                        merged_arr, merged_transform = tile_arrays[0], tile_transforms[0]
                    else:
                        merged_arr, merged_transform = self._mosaicTiles(
                            tile_arrays, tile_transforms
                        )

                    time_slices.append(merged_arr)
                    dates_out.append(acq_date)
                    final_transform = merged_transform
                # TemporaryDirectory context exit deletes all files in the chunk.

        if not time_slices:
            raise RuntimeError(f'MODIS earthaccess HDF4: no slices assembled for {var}')

        self._writeCacheFile(var, time_slices, dates_out, final_transform, cache_path)

    def _extractHdf4Slice(
        self,
        hdf_path: str,
        var: str,
        sinu_bounds: tuple,
    ) -> tuple:
        """Read one HDF4 tile, clip to sinu_bounds, scale, and mask fill values.

        Returns ``(None, None)`` if the tile does not overlap ``sinu_bounds``.
        """
        product_info = _PRODUCTS[var]
        fill = product_info['fill_value']
        scale = product_info['scale_factor']

        raw, transform = _readHdfSds(hdf_path, product_info['sds_name'])

        # Reconstruct pixel-centre coordinates from the affine transform.
        nrows, ncols = raw.shape
        xs = transform.c + transform.a * (np.arange(ncols) + 0.5)
        ys = transform.f + transform.e * (np.arange(nrows) + 0.5)  # descending

        sinu_xmin, sinu_ymin, sinu_xmax, sinu_ymax = sinu_bounds

        # Check overlap
        if xs[-1] < sinu_xmin or xs[0] > sinu_xmax:
            return None, None
        if ys[-1] > sinu_ymax or ys[0] < sinu_ymin:
            return None, None

        # Find column indices (xs is ascending)
        xi1 = max(0, int(np.searchsorted(xs, sinu_xmin)) - 1)
        xi2 = min(ncols, int(np.searchsorted(xs, sinu_xmax)) + 1)
        # Find row indices (ys is descending, so negate)
        yi1 = max(0, int(np.searchsorted(-ys, -sinu_ymax)) - 1)
        yi2 = min(nrows, int(np.searchsorted(-ys, -sinu_ymin)) + 1)

        raw_slice = raw[yi1:yi2, xi1:xi2]
        xs_slice = xs[xi1:xi2]
        ys_slice = ys[yi1:yi2]

        # Rebuild transform for the slice (upper-left pixel corner)
        pixel_w = transform.a
        pixel_h = transform.e  # negative
        slice_transform = rasterio.transform.Affine(
            pixel_w, 0.0, xs_slice[0] - pixel_w / 2,
            0.0, pixel_h, ys_slice[0] - pixel_h / 2,
        )

        scaled = raw_slice.astype(np.float64)
        scaled[raw_slice == fill] = np.nan
        mask = ~np.isnan(scaled)
        scaled[mask] *= scale

        return scaled, slice_transform

    def _mosaicTiles(
        self,
        arrays: List[np.ndarray],
        transforms: List[rasterio.transform.Affine],
    ) -> tuple:
        """Mosaic multiple sinusoidal tiles into one array using rasterio.merge."""
        sinu_rasterio_crs = watershed_workflow.crs.to_rasterio(_MODIS_SINU_CRS)
        mem_files = []
        datasets = []
        for arr, tf in zip(arrays, transforms):
            h, w = arr.shape
            mf = rasterio.io.MemoryFile()
            mem_files.append(mf)
            with mf.open(driver='GTiff', height=h, width=w, count=1,
                         dtype='float64', transform=tf, nodata=np.nan,
                         crs=sinu_rasterio_crs) as dst:
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

    # ------------------------------------------------------------------
    # Shared cache writer
    # ------------------------------------------------------------------

    def _writeCacheFile(
        self,
        var: str,
        time_slices: List[np.ndarray],
        dates: List[datetime.date],
        transform: rasterio.transform.Affine,
        cache_path: str,
    ) -> None:
        """Assemble time slices into an ``xr.Dataset`` and write as NetCDF4.

        Parameters
        ----------
        var : str
            Variable name.
        time_slices : list of np.ndarray
            One 2-D array per acquisition date.
        dates : list of datetime.date
            Acquisition dates corresponding to each slice.
        transform : rasterio.transform.Affine
            Affine transform for the spatial grid (sinusoidal metres,
            upper-left pixel-corner origin).
        cache_path : str
            Destination NetCDF4 file path.
        """
        product_info = _PRODUCTS[var]
        h, w = time_slices[0].shape

        xs = transform.c + transform.a * (np.arange(w) + 0.5)
        ys = transform.f + transform.e * (np.arange(h) + 0.5)

        time_coords = [
            cftime.datetime(d.year, d.month, d.day, calendar='standard')
            for d in dates
        ]

        data_stack = np.stack(time_slices, axis=0).astype(product_info['dtype'])

        da = xr.DataArray(
            data_stack,
            dims=['time', 'y', 'x'],
            coords={'time': time_coords, 'y': ys, 'x': xs},
            attrs={
                'long_name': product_info['long_name'],
                'units': product_info['units'],
                'scale_factor_applied': 1,
            },
        )
        ds = xr.Dataset({var: da})
        ds = ds.rio.write_crs(self.native_crs_out)
        ds.attrs['source'] = self.source
        ds.attrs['variable'] = var

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds.to_netcdf(cache_path)
        logging.info(f'    Written to: {cache_path}')
