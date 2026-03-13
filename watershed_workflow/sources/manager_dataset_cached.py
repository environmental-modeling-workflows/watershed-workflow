"""Caching layer for dataset managers.

Provides :class:`ManagerDatasetCached` and two class decorators:

* :func:`cached_dataset_manager` — for managers that write files directly
  to ``request._download_path`` during ``_downloadDataset``.
* :func:`in_memory_cached_manager` — for managers that return data in-memory
  from ``_loadDataset`` and rely on the decorator to persist it.

Apply a decorator to any :class:`ManagerDataset` subclass to add transparent
cache-directory lookup and completeness checking::

    @cached_dataset_manager(CacheInfo(...))
    class ManagerAORC(ManagerDataset):
        def isComplete(self, dir, request): ...
        def _requestDataset(self, request): ...
        def _isServerReady(self, request): ...
        def _downloadDataset(self, request): ...
        def _loadDataset(self, request): ...

The resulting class MRO is::

    CachedCls -> ManagerDatasetCached -> ManagerAORC -> ManagerDataset

so :class:`ManagerDatasetCached` intercepts each of the four abstract methods
via ``super()``, manages the cache directory, and delegates to the concrete
manager only on a cache miss.

Concrete manager contract
-------------------------
``_downloadDataset`` must write its data to ``request._download_path``
(a directory path).  ``_loadDataset`` must open/return data from
``request._download_path``.

``isComplete(dir, request) -> bool`` must return ``True`` if and only if
``dir`` contains a valid, complete cache for the given request (including
all requested variables and the requested time range).
"""

import logging
import os
import xarray as xr

from watershed_workflow.sources.cache_info import CacheInfo
from watershed_workflow.sources.manager_dataset import ManagerDataset


class ManagerDatasetCached(ManagerDataset):
    """Mixin that adds directory-based cache lookup to a :class:`ManagerDataset`.

    Never instantiated directly.  Applied via :func:`cached_dataset_manager`.

    The decorated concrete manager class must implement
    ``isComplete(dir, request) -> bool`` which returns ``True`` if ``dir``
    contains a valid, complete cache for ``request``.

    Parameters
    ----------
    cache_info : CacheInfo
        Metadata describing the cache layout and snap resolution for this manager.
    """

    def __init__(self, cache_info: CacheInfo):
        self._cache_info = cache_info
        # Does not call super().__init__() — the concrete manager's __init__
        # is called first by CachedCls.__init__ (see cached_dataset_manager).


    # ------------------------------------------------------------------
    # Four abstract-method overrides with cache interception
    # ------------------------------------------------------------------
    def _requestDataset(self, request: ManagerDataset.Request) -> ManagerDataset.Request:
        """Check cache directory; set ``request._download_path``; delegate on miss."""
        start_year = request.start.year if request.start is not None else None
        end_year = request.end.year if request.end is not None else None

        found = self._cache_info.findCacheDir(
            request.geometry.bounds,
            manager=self,
            request=request,
            start_year=start_year,
            end_year=end_year,
            temporal_resampling=request.temporal_resampling,
        )

        if found is not None:
            request._cache_hit = True
            request._download_path = found
            logging.info(f'{self.name}: cache hit at {found}')
            return request

        # Cache miss — set _download_path to the target directory and delegate.
        target = self._cache_info.cacheDirname(
            request.geometry.bounds,
            start_year=start_year,
            end_year=end_year,
            temporal_resampling=request.temporal_resampling,
        )
        os.makedirs(target, exist_ok=True)
        request._cache_hit = False
        request._download_path = target
        logging.info(f'{self.name}: cache miss, downloading to {target}')
        return super()._requestDataset(request)

    def _isServerReady(self, request: ManagerDataset.Request) -> bool:
        """Return True immediately on cache hit, else poll the server."""
        if request._cache_hit:
            return True
        return super()._isServerReady(request)

    def _downloadDataset(self, request: ManagerDataset.Request) -> None:
        """No-op on cache hit, else delegate to concrete manager."""
        if request._cache_hit:
            return
        super()._downloadDataset(request)

    def _loadDataset(self, request: ManagerDataset.Request) -> xr.Dataset:
        """Delegate to concrete manager (which reads from ``request._download_path``)."""
        return super()._loadDataset(request)


# ------------------------------------------------------------------
# Class decorators
# ------------------------------------------------------------------
def cached_dataset_manager(cache_info: CacheInfo):
    """Class decorator that adds directory-based caching to a :class:`ManagerDataset` subclass.

    Parameters
    ----------
    cache_info : CacheInfo
        Cache metadata for this manager.

    Returns
    -------
    decorator : callable
        A decorator that takes a :class:`ManagerDataset` subclass and returns
        a new class with :class:`ManagerDatasetCached` injected into its MRO.

    Notes
    -----
    The resulting class MRO is::

        CachedCls -> ManagerDatasetCached -> OriginalCls -> ManagerDataset

    ``ManagerDatasetCached`` intercepts the four abstract methods via
    ``super()``, managing the cache directory before delegating to ``OriginalCls``.

    The decorated class must implement ``isComplete(dir, request) -> bool``.
    """
    def decorator(cls: type) -> type:
        class CachedCls(ManagerDatasetCached, cls):
            def __init__(self, *args, **kwargs):
                cls.__init__(self, *args, **kwargs)
                ManagerDatasetCached.__init__(self, cache_info)
        CachedCls.__name__ = cls.__name__
        CachedCls.__qualname__ = cls.__qualname__
        CachedCls.__module__ = cls.__module__
        return CachedCls
    return decorator


def in_memory_cached_manager(cache_info: CacheInfo):
    """Class decorator for managers that fetch data in-memory during ``_downloadDataset``.

    Like :func:`cached_dataset_manager` but the concrete manager stores its result
    on ``request._dataset`` inside ``_downloadDataset`` rather than writing files.
    The decorator handles persisting ``request._dataset`` to ``data.nc`` on a cache
    miss and reading it back on a cache hit — the concrete ``_loadDataset`` is a
    no-op (just ``return request._dataset``).

    Parameters
    ----------
    cache_info : CacheInfo
        Cache metadata for this manager.

    Returns
    -------
    decorator : callable
        A decorator that takes a :class:`ManagerDataset` subclass and returns
        a new class that caches the in-memory dataset to ``data.nc`` inside
        the cache directory on a cache miss, and re-reads it on a cache hit.

    Notes
    -----
    The resulting class MRO is::

        CachedCls -> ManagerDatasetCached -> OriginalCls -> ManagerDataset

    ``isComplete(dir, request)`` returns ``True`` iff ``data.nc`` exists in
    ``dir``.  Concrete classes may override this if needed.

    Concrete manager contract:
    - ``_downloadDataset(request)``: call the API, store result as ``request._dataset``
    - ``_loadDataset(request)``: return ``request._dataset``
    """
    def decorator(cls: type) -> type:
        class CachedCls(ManagerDatasetCached, cls):
            def __init__(self, *args, **kwargs):
                cls.__init__(self, *args, **kwargs)
                ManagerDatasetCached.__init__(self, cache_info)

            def isComplete(self, dir, request):
                return os.path.exists(os.path.join(dir, 'data.nc'))

            def _downloadDataset(self, request):
                if request._cache_hit:
                    return
                super()._downloadDataset(request)
                path = os.path.join(request._download_path, 'data.nc')
                logging.info(f'{self.name}: writing in-memory cache to {path}')
                request._dataset.to_netcdf(path)

            def _loadDataset(self, request):
                if request._cache_hit:
                    return xr.open_dataset(os.path.join(request._download_path, 'data.nc'))
                return request._dataset

        CachedCls.__name__ = cls.__name__
        CachedCls.__qualname__ = cls.__qualname__
        CachedCls.__module__ = cls.__module__
        return CachedCls
    return decorator
