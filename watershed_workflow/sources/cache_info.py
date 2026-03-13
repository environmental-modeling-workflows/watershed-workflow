"""Class for managing cached data directories for offline use."""
import dataclasses
import math
import os
import re
from typing import Tuple

import watershed_workflow.utils.config


#
# Snapping
#
# Caches are snapped so that there is some coarse-grained resolution
# to what actually gets downloaded.  Small changes in input geometry
# should not change which cache directory is used.
# ------------------------------------------------------------------
def _snapBounds(bounds: tuple, step: float) -> tuple:
    """Snap bounds outward by up to one step unit.

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
    """
    xmin, ymin, xmax, ymax = bounds
    return (math.floor(xmin / step) * step,
            math.floor(ymin / step) * step,
            math.ceil(xmax  / step) * step,
            math.ceil(ymax  / step) * step)


@dataclasses.dataclass
class CacheInfo:
    """A helper class for managing cached data directories.

    Parameters
    ----------
    category : str
        Top-level folder, e.g. ``'climate'``.
    subcategory : str
        Second-level folder, e.g. ``'aorc'``.
    name : str
        Slug used in directory names, e.g. ``'aorc_4km'``.
    snap_resolution : float
        Grid step for snapping bounds outward.
    is_temporal : bool, optional
        Whether the cache directory name encodes a year range.
    is_resampled : bool, optional
        Whether the cache directory name encodes a temporal resampling token.
    """
    category: str
    subcategory: str
    name: str
    snap_resolution: float
    is_temporal: bool = False
    is_resampled: bool = False

    def cacheFolder(self) -> str:
        """Return the absolute path of the cache directory.

        Returns
        -------
        str
            Location of the cache root for this manager.
        """
        data_dir = watershed_workflow.utils.config.rcParams['DEFAULT']['data_directory']
        return os.path.join(data_dir, self.category, self.subcategory, self.name)

    def cacheDirname(self,
                     geometry_bounds: Tuple[float, float, float, float],
                     start_year: int | None = None,
                     end_year: int | None = None,
                     temporal_resampling: str | None = None) -> str:
        """Return the canonical cache directory path for a request.

        Parameters
        ----------
        geometry_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` — snapped internally before use.
        start_year : int or None
            Start year.  Required when ``is_temporal`` is ``True``.
        end_year : int or None
            End year.  Required when ``is_temporal`` is ``True``.
        temporal_resampling : str or None
            Pandas offset alias (e.g. ``'1D'``).  Required when
            ``is_resampled`` is ``True``; if ``None`` and
            ``is_resampled`` is ``True``, denotes the native rate.

        Returns
        -------
        str
            Full absolute path to the cache directory.

        Notes
        -----
        Format::

            {folder}/{name}[_{start_year}-{end_year}][_{temporal_resampling}]
                _{xmin:.4f}_{ymin:.4f}_{xmax:.4f}_{ymax:.4f}
        """
        xmin, ymin, xmax, ymax = _snapBounds(geometry_bounds, self.snap_resolution)
        parts = [self.name]

        if self.is_temporal:
            if start_year is None or end_year is None:
                raise ValueError(f'{self.name}: is_temporal=True but start_year or end_year is None')
            parts.append(f'{start_year}-{end_year}')

        if self.is_resampled:
            parts.append(temporal_resampling if temporal_resampling is not None else 'native')

        parts.append(f'{xmin:.4f}')
        parts.append(f'{ymin:.4f}')
        parts.append(f'{xmax:.4f}')
        parts.append(f'{ymax:.4f}')
        dirname = '_'.join(parts)
        return os.path.join(self.cacheFolder(), dirname)

    def parseCacheDirname(self, dirname: str) -> dict | None:
        """Parse a standard cache directory name into its components.

        Parameters
        ----------
        dirname : str
            Bare directory name (no parent path), e.g.
            ``'aorc_4km_2020-2020_1D_-76.0000_42.0000_-73.0000_45.0000'``.

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
        if self.is_resampled:
            resample_pat = r'(?P<temporal_resampling>[^_]+)_'
        else:
            resample_pat = ''
        if self.is_temporal:
            time_pat = rf'(?P<start_year>{int_pat})-(?P<end_year>{int_pat})_'
        else:
            time_pat = ''

        slug = re.escape(self.name)
        pattern = rf'^{slug}_{time_pat}{resample_pat}{bounds_pat}'

        m = re.match(pattern, dirname)
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
        if self.is_resampled:
            result['temporal_resampling'] = m.group('temporal_resampling')
        return result

    # ------------------------------------------------------------------
    # Superset detection
    # ------------------------------------------------------------------
    def findCacheDir(self,
                     geometry_bounds: tuple,
                     manager,
                     request,
                     start_year: int | None = None,
                     end_year: int | None = None,
                     temporal_resampling: str | None = None) -> str | None:
        """Scan the cache folder for a directory that contains the current request.

        Checks both spatial containment (against ``geometry_bounds``, the
        buffered un-snapped bounds used for clipping) and, when
        ``is_temporal`` is ``True``, temporal containment (cached year
        range must span the requested year range).  When ``is_resampled``
        is ``True``, the resampling rate must match exactly.  Finally,
        calls ``manager.isComplete(candidate_dir, request)`` to verify
        the directory contents are valid for the request.

        Parameters
        ----------
        geometry_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)``
            A candidate directory must spatially contain these bounds.
        manager : ManagerDatasetCached
            The manager whose ``isComplete(dir, request)`` is called to
            validate directory contents.
        request : ManagerDataset.Request
            The request being fulfilled; passed to ``isComplete``.
        start_year : int or None
            Requested start year.  Required when ``is_temporal`` is ``True``.
        end_year : int or None
            Requested end year.  Required when ``is_temporal`` is ``True``.
        temporal_resampling : str or None
            Pandas offset alias (e.g. ``'1D'``).  Matched exactly when
            ``is_resampled`` is ``True``.

        Returns
        -------
        str or None
            Absolute path to a suitable cache directory, or ``None``.
        """
        folder = self.cacheFolder()
        if not os.path.isdir(folder):
            return None

        # If the exact target directory already exists and is complete, return it.
        target_path = self.cacheDirname(geometry_bounds,
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
            parsed = self.parseCacheDirname(dirname)
            if parsed is None:
                continue

            # Resampling: exact match only
            if self.is_resampled and parsed.get('temporal_resampling') != req_resampling:
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

            # Contents check
            if manager.isComplete(dirpath, request):
                return dirpath

        return None
