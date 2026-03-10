"""Shared abstract base class for all data source managers.

Both ManagerDataset (raster/xarray) and ManagerShapes (vector/GeoDataFrame)
inherit from Manager, which owns the common properties and all cache
infrastructure: folder layout, canonical filename generation, and superset
detection (spatial and temporal).
"""

import math
import os
import re
import logging
from typing import Optional, List

from watershed_workflow.crs import CRS
import watershed_workflow.config


class Manager:
    """Base class for all managers.

    Owns the cache infrastructure shared by raster and vector managers:

    - ``_cacheFolder()`` — canonical cache directory
    - ``_cacheFilename(var, start_year, end_year, snapped_bounds)`` — canonical filename
    - ``_parseCacheFilename(fname)`` — inverse of the above
    - ``_checkCache(snapped_bounds, geometry_bounds, var, start_year, end_year)``
      — superset detection

    Subclasses declare a few traits in their ``__init__`` call:

    - ``cache_category`` — top-level folder group, e.g. ``'meteorology'``.
      Pass ``None`` to opt out of the cache system entirely (e.g. managers
      that read a single pre-existing file at a fixed path).
    - ``cache_extension`` — file extension, e.g. ``'nc'``, ``'shp'``.
    - ``has_varname`` — ``True`` if one cache file is written per variable.
    - ``is_temporal`` — ``True`` if the cache filename encodes a year range.
      Derived automatically from whether ``native_start`` is not ``None``
      (for ``ManagerDataset``); shapes managers always pass ``False``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self,
                 name: str,
                 source: str,
                 native_crs_in: CRS | None,
                 native_resolution: float | None,
                 cache_category: str | None,
                 cache_extension: str,
                 has_varname: bool,
                 is_temporal: bool,
                 has_resampling: bool = False,
                 short_name: str | None = None):
        """Initialize shared manager properties.

        Parameters
        ----------
        name : str
            Human-readable name of the manager / dataset.
        source : str
            Data source description or URL.
        native_crs_in : CRS or None
            CRS expected for input geometry.  May be ``None`` when not yet
            known (e.g. ``ManagerShapefile`` discovers it at first open).
        native_resolution : float or None
            Characteristic resolution in ``native_crs_in`` units.  May be
            ``None`` when not yet known.
        cache_category : str or None
            Top-level folder group (e.g. ``'meteorology'``).  ``None`` opts
            out of the standard cache system entirely.
        cache_extension : str
            File extension for cache files, without a leading dot.
        has_varname : bool
            ``True`` when one file is written per variable (e.g. DayMet).
            ``False`` when all variables share one file (e.g. AORC).
        is_temporal : bool
            ``True`` when the cache filename encodes a year range.
        has_resampling : bool, optional
            ``True`` when the cache filename encodes a temporal resampling
            rate (e.g. AORC ``'1D'``).  Default ``False``.
        short_name : str, optional
            Short, filesystem-safe name used as the leaf cache directory and
            filename prefix (e.g. ``'DayMet'``).  When ``None``, falls back
            to a slugified version of ``name``.
        """
        self.name = name
        self.source = source
        self.native_crs_in = native_crs_in
        self.native_resolution = native_resolution
        self.cache_category = cache_category
        self.cache_extension = cache_extension
        self.has_varname = has_varname
        self.is_temporal = is_temporal
        self.has_resampling = has_resampling
        self.short_name = short_name if short_name is not None else self._nameSlug()

    # ------------------------------------------------------------------
    # Hook called before any data access
    # ------------------------------------------------------------------

    def _prerequestDataset(self) -> None:
        """Called before processing any request.

        Override to perform one-time setup that cannot happen at
        construction time (e.g. downloading an index file, reading
        metadata from a file that may not yet exist).
        """
        pass

    # ------------------------------------------------------------------
    # Snapping
    # ------------------------------------------------------------------

    def _snapBounds(self, bounds: tuple) -> tuple:
        """Snap bounds outward by up to one native resolution unit.

        Parameters
        ----------
        bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)``.

        Returns
        -------
        tuple of float
            Snapped ``(xmin, ymin, xmax, ymax)``.

        Notes
        -----
        Minimums snap down (floor), maximums snap up (ceil), so the box
        always expands.  ``math.floor`` / ``math.ceil`` handle negative
        coordinates (western longitudes, southern latitudes) correctly.
        The snap step equals ``self.native_resolution``, so the additional
        margin beyond the ``3 × native_resolution`` buffer is at most
        ``1 × native_resolution`` per edge.
        """
        step = self.native_resolution
        xmin, ymin, xmax, ymax = bounds
        return (math.floor(xmin / step) * step,
                math.floor(ymin / step) * step,
                math.ceil(xmax  / step) * step,
                math.ceil(ymax  / step) * step)

    # ------------------------------------------------------------------
    # Cache folder and filename
    # ------------------------------------------------------------------

    def _nameSlug(self) -> str:
        """Return a filesystem-safe version of self.name."""
        return re.sub(r'[^a-z0-9]+', '_', self.name.lower()).strip('_')

    def _cacheFolder(self) -> str:
        """Return the absolute path of the cache directory.

        Returns
        -------
        str
            ``{data_directory}/{cache_category}/{name_slug}/``

        Notes
        -----
        Returns ``None`` when ``cache_category`` is ``None``.
        """
        if self.cache_category is None:
            return None
        data_dir = watershed_workflow.config.rcParams['DEFAULT']['data_directory']
        return os.path.join(data_dir, self.cache_category, self.short_name)

    def _cacheFilename(self,
                       snapped_bounds: tuple,
                       var: str | None = None,
                       start_year: int | None = None,
                       end_year: int | None = None,
                       temporal_resampling: str | None = None) -> str:
        """Return the canonical cache filename for a request.

        Parameters
        ----------
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` already snapped.
        var : str or None
            Variable name.  Required when ``has_varname`` is ``True``.
        start_year : int or None
            Start year.  Required when ``is_temporal`` is ``True``.
        end_year : int or None
            End year.  Required when ``is_temporal`` is ``True``.
        temporal_resampling : str or None
            Pandas offset alias (e.g. ``'1D'``).  Required when
            ``has_resampling`` is ``True``; if ``None`` and
            ``has_resampling`` is ``True``, denotes the native (hourly) rate.

        Returns
        -------
        str
            Full absolute path to the cache file.

        Notes
        -----
        Format::

            {folder}/{slug}[_{var}][_{start_year}-{end_year}][_{temporal_resampling}]
                _{xmin:.4f}_{ymin:.4f}_{xmax:.4f}_{ymax:.4f}.{ext}
        """
        if self.cache_category is None:
            raise RuntimeError(f'{self.name}: cache_category is None, cannot generate cache filename')
        slug = self.short_name
        parts = [slug]
        if self.has_varname:
            if var is None:
                raise ValueError(f'{self.name}: has_varname=True but var=None')
            parts.append(var)
        if self.is_temporal:
            if start_year is None or end_year is None:
                raise ValueError(f'{self.name}: is_temporal=True but start_year or end_year is None')
            parts.append(f'{start_year}-{end_year}')
        if self.has_resampling:
            parts.append(temporal_resampling if temporal_resampling is not None else 'native')
        xmin, ymin, xmax, ymax = snapped_bounds
        parts.append(f'{xmin:.4f}')
        parts.append(f'{ymin:.4f}')
        parts.append(f'{xmax:.4f}')
        parts.append(f'{ymax:.4f}')
        fname = '_'.join(parts) + '.' + self.cache_extension
        return os.path.join(self._cacheFolder(), fname)

    def _parseCacheFilename(self, fname: str) -> dict | None:
        """Parse a standard cache filename into its components.

        Parameters
        ----------
        fname : str
            Bare filename (no directory), e.g.
            ``'daymet_1km_tmin_2020-2020_-76.0000_42.0000_-73.0000_45.0000.nc'``.

        Returns
        -------
        dict or None
            Keys: ``'xmin'``, ``'ymin'``, ``'xmax'``, ``'ymax'``; and
            optionally ``'start_year'``, ``'end_year'`` (int), ``'var'``
            (str), and ``'temporal_resampling'`` (str).  Returns ``None``
            if ``fname`` does not match the expected pattern.
        """
        # Strip extension
        stem, ext = os.path.splitext(fname)
        if ext.lstrip('.') != self.cache_extension:
            return None

        float_pat = r'-?\d+\.\d+'
        int_pat   = r'\d+'

        # Build regex from right to left (bounds are always the last four tokens)
        # Pattern: slug [_var] [_start-end] [_resampling] _xmin _ymin _xmax _ymax .ext
        bounds_pat = (rf'(?P<xmin>{float_pat})_(?P<ymin>{float_pat})'
                      rf'_(?P<xmax>{float_pat})_(?P<ymax>{float_pat})$')
        if self.has_resampling:
            resample_pat = r'(?P<temporal_resampling>[^_]+)_'
        else:
            resample_pat = ''
        if self.is_temporal:
            time_pat = rf'(?P<start_year>{int_pat})-(?P<end_year>{int_pat})_'
        else:
            time_pat = ''
        if self.has_varname:
            var_pat = r'(?P<var>[^_]+)_'
        else:
            var_pat = ''

        slug = re.escape(self.short_name)
        pattern = rf'^{slug}_{var_pat}{time_pat}{resample_pat}{bounds_pat}'

        m = re.match(pattern, stem)
        if m is None:
            return None

        result = {
            'xmin': float(m.group('xmin')),
            'ymin': float(m.group('ymin')),
            'xmax': float(m.group('xmax')),
            'ymax': float(m.group('ymax')),
        }
        if self.is_temporal:
            result['start_year'] = int(m.group('start_year'))
            result['end_year']   = int(m.group('end_year'))
        if self.has_varname:
            result['var'] = m.group('var')
        if self.has_resampling:
            result['temporal_resampling'] = m.group('temporal_resampling')
        return result

    # ------------------------------------------------------------------
    # Superset detection
    # ------------------------------------------------------------------

    def _checkCache(self,
                    geometry_bounds: tuple,
                    snapped_bounds: tuple,
                    var: str | None = None,
                    start_year: int | None = None,
                    end_year: int | None = None,
                    temporal_resampling: str | None = None) -> str | None:
        """Scan the cache folder for a file that contains the current request.

        Checks both spatial containment (against ``geometry_bounds``, the
        buffered un-snapped bounds used for clipping) and, when
        ``is_temporal`` is ``True``, temporal containment (cached year
        range must span the requested year range).  When ``has_resampling``
        is ``True``, the resampling rate must match exactly.

        Parameters
        ----------
        geometry_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` of the buffered, un-snapped polygon.
            A candidate file must spatially contain these bounds.
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` snapped.  Used to generate the
            exact target filename and short-circuit if it already exists.
        var : str or None
            Variable name for per-variable managers.
        start_year : int or None
            Requested start year.  Required when ``is_temporal`` is ``True``.
        end_year : int or None
            Requested end year.  Required when ``is_temporal`` is ``True``.
        temporal_resampling : str or None
            Pandas offset alias (e.g. ``'1D'``).  Required when
            ``has_resampling`` is ``True``.  Matched exactly — a cached
            ``'1D'`` file will not be reused for a ``'2D'`` request.

        Returns
        -------
        str or None
            Absolute path to a suitable cache file, or ``None``.
        """
        if self.cache_category is None:
            return None
        folder = self._cacheFolder()
        if not os.path.isdir(folder):
            return None

        # If the exact target file already exists, return it immediately.
        target_path = self._cacheFilename(snapped_bounds, var=var,
                                          start_year=start_year, end_year=end_year,
                                          temporal_resampling=temporal_resampling)
        target_fname = os.path.basename(target_path)
        if os.path.exists(target_path):
            return target_path

        req_xmin, req_ymin, req_xmax, req_ymax = geometry_bounds
        req_resampling = temporal_resampling if temporal_resampling is not None else 'native'

        for fname in os.listdir(folder):
            if fname == target_fname:
                continue
            parsed = self._parseCacheFilename(fname)
            if parsed is None:
                continue

            # Variable match
            if self.has_varname and parsed.get('var') != var:
                continue

            # Resampling: exact match only
            if self.has_resampling and parsed.get('temporal_resampling') != req_resampling:
                continue

            # Spatial superset
            if not (parsed['xmin'] <= req_xmin and parsed['ymin'] <= req_ymin
                    and parsed['xmax'] >= req_xmax and parsed['ymax'] >= req_ymax):
                continue

            # Temporal superset
            if self.is_temporal:
                if start_year is None or end_year is None:
                    continue
                if not (parsed['start_year'] <= start_year
                        and parsed['end_year'] >= end_year):
                    continue

            return os.path.join(folder, fname)

        return None
