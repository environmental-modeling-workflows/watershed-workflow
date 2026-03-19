"""Free functions for managing cached data directories.

Cache directories are laid out as::

    {data_dir}/{category}/{product_short}/{source_short}/
        {source_short}[_{start_year}-{end_year}][_{temporal_resampling}]
            _{xmin:.4f}_{ymin:.4f}_{xmax:.4f}_{ymax:.4f}/

Bounds are snapped outward to a grid aligned with ``attrs.native_resolution``
before being encoded in the directory name, so that small changes in the
input geometry do not produce different cache directories.
"""
import math
import os
import re
from typing import TYPE_CHECKING

import watershed_workflow.utils.config

if TYPE_CHECKING:
    from watershed_workflow.sources.manager import ManagerAttributes


# ------------------------------------------------------------------
# Snapping
# ------------------------------------------------------------------

def snapBounds(bounds: tuple, step: float) -> tuple:
    """Snap bounds outward by up to one step unit.

    Parameters
    ----------
    bounds : tuple of float
        ``(xmin, ymin, xmax, ymax)``.
    step : float
        Grid step used for snapping.

    Returns
    -------
    tuple of float
        Snapped ``(xmin, ymin, xmax, ymax)``.

    Notes
    -----
    Minimums snap down (floor), maximums snap up (ceil), so the box
    always expands.  ``math.floor`` / ``math.ceil`` handle negative
    coordinates (western longitudes, southern latitudes) correctly.
    """
    xmin, ymin, xmax, ymax = bounds
    return (math.floor(xmin / step) * step,
            math.floor(ymin / step) * step,
            math.ceil(xmax  / step) * step,
            math.ceil(ymax  / step) * step)


# ------------------------------------------------------------------
# Cache folder and directory naming
# ------------------------------------------------------------------

def cacheFolder(attrs) -> str:
    """Return the absolute root cache directory for a manager.

    Parameters
    ----------
    attrs : ManagerAttributes
        Manager metadata.

    Returns
    -------
    str
        ``{data_dir}/{category}/{product_short}/{source_short}/``
    """
    data_dir = watershed_workflow.utils.config.rcParams['DEFAULT']['data_directory']
    return os.path.join(data_dir, attrs.category, attrs.product_short, attrs.source_short)


def cacheDirname(attrs,
                 geometry_bounds: tuple,
                 start_year: int | None = None,
                 end_year: int | None = None,
                 temporal_resampling: str | None = None) -> str:
    """Return the canonical cache directory path for a request.

    Parameters
    ----------
    attrs : ManagerAttributes
        Manager metadata.
    geometry_bounds : tuple of float
        ``(xmin, ymin, xmax, ymax)`` — snapped internally before use.
    start_year : int or None, optional
        Start year.  Required when ``attrs.is_temporal`` is ``True``.
    end_year : int or None, optional
        End year.  Required when ``attrs.is_temporal`` is ``True``.
    temporal_resampling : str or None, optional
        Pandas offset alias (e.g. ``'1D'``).  Required when
        ``attrs.is_resampled`` is ``True``; if ``None`` and
        ``attrs.is_resampled`` is ``True``, denotes the native rate.

    Returns
    -------
    str
        Full absolute path to the cache directory.

    Notes
    -----
    Format::

        {folder}/{source_short}[_{start_year}-{end_year}][_{temporal_resampling}]
            _{xmin:.4f}_{ymin:.4f}_{xmax:.4f}_{ymax:.4f}
    """
    step = attrs.native_resolution
    xmin, ymin, xmax, ymax = snapBounds(geometry_bounds, step)
    parts = [attrs.source_short]

    if attrs.is_temporal:
        if start_year is None or end_year is None:
            raise ValueError(
                f'{attrs.source_short}: is_temporal=True but '
                f'start_year or end_year is None'
            )
        parts.append(f'{start_year}-{end_year}')

    if attrs.is_resampled:
        parts.append(temporal_resampling if temporal_resampling is not None else 'native')

    parts.append(f'{xmin:.4f}')
    parts.append(f'{ymin:.4f}')
    parts.append(f'{xmax:.4f}')
    parts.append(f'{ymax:.4f}')
    dirname = '_'.join(parts)
    return os.path.join(cacheFolder(attrs), dirname)


def parseCacheDirname(attrs, dirname: str) -> dict | None:
    """Parse a cache directory name into its components.

    Parameters
    ----------
    attrs : ManagerAttributes
        Manager metadata.
    dirname : str
        Bare directory name (no parent path), e.g.
        ``'ornl_daac_zarr_2020-2020_1D_-76.0000_42.0000_-73.0000_45.0000'``.

    Returns
    -------
    dict or None
        Keys: ``'xmin'``, ``'ymin'``, ``'xmax'``, ``'ymax'``; and
        optionally ``'start_year'``, ``'end_year'`` (int),
        ``'temporal_resampling'`` (str).  Returns ``None`` if
        ``dirname`` does not match the expected pattern.
    """
    float_pat = r'-?\d+\.\d+'
    int_pat   = r'\d+'

    # Build regex from right to left (bounds are always the last four tokens)
    bounds_pat = (rf'(?P<xmin>{float_pat})_(?P<ymin>{float_pat})'
                  rf'_(?P<xmax>{float_pat})_(?P<ymax>{float_pat})$')
    resample_pat = r'(?P<temporal_resampling>[^_]+)_' if attrs.is_resampled else ''
    time_pat = (rf'(?P<start_year>{int_pat})-(?P<end_year>{int_pat})_'
                if attrs.is_temporal else '')

    slug = re.escape(attrs.source_short)
    pattern = rf'^{slug}_{time_pat}{resample_pat}{bounds_pat}'

    m = re.match(pattern, dirname)
    if m is None:
        return None

    result = {k: float(m.group(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')}
    if attrs.is_temporal:
        result['start_year'] = int(m.group('start_year'))
        result['end_year']   = int(m.group('end_year'))
    if attrs.is_resampled:
        result['temporal_resampling'] = m.group('temporal_resampling')
    return result


# ------------------------------------------------------------------
# Superset detection
# ------------------------------------------------------------------

def findCacheDir(attrs,
                 geometry_bounds: tuple,
                 manager,
                 request,
                 start_year: int | None = None,
                 end_year: int | None = None,
                 temporal_resampling: str | None = None) -> str | None:
    """Scan the cache folder for a directory that contains the current request.

    Checks both spatial containment (against ``geometry_bounds``, the
    buffered un-snapped bounds used for clipping) and, when
    ``attrs.is_temporal`` is ``True``, temporal containment (cached year
    range must span the requested year range).  When ``attrs.is_resampled``
    is ``True``, the resampling rate must match exactly.  Finally, calls
    ``manager.isComplete(candidate_dir, request)`` to verify the directory
    contents are valid for the request.

    Parameters
    ----------
    attrs : ManagerAttributes
        Manager metadata.
    geometry_bounds : tuple of float
        ``(xmin, ymin, xmax, ymax)``.
        A candidate directory must spatially contain these bounds.
    manager : ManagerDatasetCached
        The manager whose ``isComplete(dir, request)`` is called to
        validate directory contents.
    request : ManagerDataset.Request
        The request being fulfilled; passed to ``isComplete``.
    start_year : int or None, optional
        Requested start year.  Required when ``attrs.is_temporal`` is ``True``.
    end_year : int or None, optional
        Requested end year.  Required when ``attrs.is_temporal`` is ``True``.
    temporal_resampling : str or None, optional
        Pandas offset alias (e.g. ``'1D'``).  Matched exactly when
        ``attrs.is_resampled`` is ``True``.

    Returns
    -------
    str or None
        Absolute path to a suitable cache directory, or ``None``.
    """
    folder = cacheFolder(attrs)
    if not os.path.isdir(folder):
        return None

    # If the exact target directory already exists and is complete, return it.
    target_path = cacheDirname(attrs, geometry_bounds,
                                start_year=start_year,
                                end_year=end_year,
                                temporal_resampling=temporal_resampling)
    if os.path.isdir(target_path) and manager.isComplete(target_path, request):
        return target_path

    # Otherwise scan for any superset directory that is complete.
    req_xmin, req_ymin, req_xmax, req_ymax = geometry_bounds
    req_resampling = temporal_resampling if temporal_resampling is not None else 'native'
    target_dirname = os.path.basename(target_path)

    for dirname in os.listdir(folder):
        if dirname == target_dirname:
            continue
        dirpath = os.path.join(folder, dirname)
        if not os.path.isdir(dirpath):
            continue
        parsed = parseCacheDirname(attrs, dirname)
        if parsed is None:
            continue

        # Resampling: exact match only
        if attrs.is_resampled and parsed.get('temporal_resampling') != req_resampling:
            continue

        # Spatial superset
        if not (parsed['xmin'] <= req_xmin and parsed['ymin'] <= req_ymin
                and parsed['xmax'] >= req_xmax and parsed['ymax'] >= req_ymax):
            continue

        # Temporal superset
        if attrs.is_temporal:
            if start_year is None or end_year is None:
                continue
            if not (parsed['start_year'] <= start_year
                    and parsed['end_year'] >= end_year):
                continue

        # Contents check
        if manager.isComplete(dirpath, request):
            return dirpath

    return None


# ------------------------------------------------------------------
# Fixed local file paths (for manually-downloaded datasets)
# ------------------------------------------------------------------

def localFilePath(attrs, filename: str) -> str:
    """Return the absolute path for a locally-stored fixed file.

    Used by managers that require manual download (GLHYMPS, PelletierDTB,
    ShangguanDTB) where data is stored at a fixed path rather than at a
    geometry-keyed cache path.

    Parameters
    ----------
    attrs : ManagerAttributes
        Manager metadata.
    filename : str
        Bare filename (e.g. ``'GLHYMPS.shp'``).

    Returns
    -------
    str
        ``{data_dir}/{category}/{product_short}/{source_short}/{filename}``
    """
    return os.path.join(cacheFolder(attrs), filename)
