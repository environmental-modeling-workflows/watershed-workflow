"""Unit tests for Manager cache infrastructure.

Tests _snapBounds, _cacheFilename, _parseCacheFilename, and _checkCache
on the new Manager shared ABC.
"""
import os
import pytest

import watershed_workflow.config
from watershed_workflow.sources.manager import Manager


# ---------------------------------------------------------------------------
# Minimal concrete Manager for testing
# ---------------------------------------------------------------------------

class _FakeManager(Manager):
    """Minimal concrete subclass used for testing."""

    def __init__(self, name='test_manager', native_resolution=1.0,
                 cache_category='test_cat', cache_extension='nc',
                 has_varname=False, is_temporal=False, has_resampling=False):
        super().__init__(
            name=name,
            source='test',
            native_crs_in=None,
            native_resolution=native_resolution,
            cache_category=cache_category,
            cache_extension=cache_extension,
            has_varname=has_varname,
            is_temporal=is_temporal,
            has_resampling=has_resampling,
        )


# ---------------------------------------------------------------------------
# _snapBounds tests
# ---------------------------------------------------------------------------

def test_snapBounds_positive_coords():
    mgr = _FakeManager(native_resolution=1.0)
    bounds = (1.1, 2.2, 3.3, 4.4)
    snapped = mgr._snapBounds(bounds)
    xmin, ymin, xmax, ymax = snapped
    assert xmin <= 1.1
    assert ymin <= 2.2
    assert xmax >= 3.3
    assert ymax >= 4.4
    # Results should be multiples of native_resolution (1.0)
    assert xmin % 1.0 == pytest.approx(0.0)
    assert ymin % 1.0 == pytest.approx(0.0)
    assert xmax % 1.0 == pytest.approx(0.0)
    assert ymax % 1.0 == pytest.approx(0.0)


def test_snapBounds_negative_coords():
    # Western longitudes and southern latitudes (negative values)
    mgr = _FakeManager(native_resolution=1.0)
    bounds = (-75.3, -12.7, -74.1, -11.2)
    snapped = mgr._snapBounds(bounds)
    xmin, ymin, xmax, ymax = snapped
    # mins snap down (more negative), maxes snap up (less negative)
    assert xmin <= -75.3
    assert ymin <= -12.7
    assert xmax >= -74.1
    assert ymax >= -11.2
    assert xmin == pytest.approx(-76.0)
    assert ymin == pytest.approx(-13.0)
    assert xmax == pytest.approx(-74.0)
    assert ymax == pytest.approx(-11.0)


def test_snapBounds_idempotent():
    # A box already on the grid should not change when snapped again
    mgr = _FakeManager(native_resolution=1.0)
    bounds = (-76.0, -13.0, -74.0, -11.0)
    snapped = mgr._snapBounds(bounds)
    assert snapped == pytest.approx(bounds)
    assert mgr._snapBounds(snapped) == pytest.approx(snapped)


# ---------------------------------------------------------------------------
# _cacheFilename / _parseCacheFilename tests
# ---------------------------------------------------------------------------

def test_cacheFilename_no_var_no_time(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=False, is_temporal=False)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds)
    assert fname.endswith('.nc')
    # slug should be in the filename
    assert 'test_manager' in os.path.basename(fname)
    # bounds should appear
    assert '-76.0000' in fname or '-76.0' in fname


def test_cacheFilename_with_var(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=True, is_temporal=False)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds, var='tmin')
    assert 'tmin' in os.path.basename(fname)


def test_cacheFilename_with_time(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=False, is_temporal=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds, start_year=2020, end_year=2022)
    assert '2020-2022' in os.path.basename(fname)


def test_parseCacheFilename_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=True, is_temporal=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds, var='tmin', start_year=2020, end_year=2022)
    parsed = mgr._parseCacheFilename(os.path.basename(fname))
    assert parsed is not None
    assert parsed['xmin'] == pytest.approx(-76.0)
    assert parsed['ymin'] == pytest.approx(42.0)
    assert parsed['xmax'] == pytest.approx(-73.0)
    assert parsed['ymax'] == pytest.approx(45.0)
    assert parsed['var'] == 'tmin'
    assert parsed['start_year'] == 2020
    assert parsed['end_year'] == 2022


def test_parseCacheFilename_wrong_extension(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(cache_extension='nc')
    result = mgr._parseCacheFilename('test_manager_-76.0000_42.0000_-73.0000_45.0000.shp')
    assert result is None


# ---------------------------------------------------------------------------
# _checkCache tests
# ---------------------------------------------------------------------------

def _touch_cache(mgr, bounds, var=None, start_year=None, end_year=None,
                 temporal_resampling=None):
    """Create an empty file at the canonical cache path."""
    path = mgr._cacheFilename(bounds, var=var, start_year=start_year, end_year=end_year,
                              temporal_resampling=temporal_resampling)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()
    return path


def test_checkCache_finds_spatial_superset(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=False, is_temporal=False)

    # cached file covers a larger area
    cached_bounds = (-77.0, 41.0, -72.0, 46.0)
    _touch_cache(mgr, cached_bounds)

    # request is strictly inside the cached bounds
    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-75.5, 42.5, -73.5, 44.5)  # un-snapped, smaller

    result = mgr._checkCache(request_geometry, request_snapped)
    assert result is not None
    assert result.endswith('.nc')


def test_checkCache_no_match_spatial(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=False, is_temporal=False)

    # cached file covers a smaller area than the request
    cached_bounds = (-75.0, 43.0, -74.0, 44.0)
    _touch_cache(mgr, cached_bounds)

    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-76.0, 42.0, -73.0, 45.0)

    result = mgr._checkCache(request_geometry, request_snapped)
    assert result is None


def test_checkCache_wrong_var_excluded(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=True, is_temporal=False)

    # cached file is for 'tmax', not 'tmin'
    cached_bounds = (-77.0, 41.0, -72.0, 46.0)
    _touch_cache(mgr, cached_bounds, var='tmax')

    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-75.5, 42.5, -73.5, 44.5)

    result = mgr._checkCache(request_geometry, request_snapped, var='tmin')
    assert result is None


def test_checkCache_temporal_superset(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=True, is_temporal=True)

    # cached file covers years 2019-2023, request is 2020-2022
    cached_bounds = (-77.0, 41.0, -72.0, 46.0)
    _touch_cache(mgr, cached_bounds, var='tmin', start_year=2019, end_year=2023)

    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-75.5, 42.5, -73.5, 44.5)

    result = mgr._checkCache(request_geometry, request_snapped,
                             var='tmin', start_year=2020, end_year=2022)
    assert result is not None


def test_checkCache_temporal_no_span(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=True, is_temporal=True)

    # cached file covers years 2020-2021 only, request is 2020-2022
    cached_bounds = (-77.0, 41.0, -72.0, 46.0)
    _touch_cache(mgr, cached_bounds, var='tmin', start_year=2020, end_year=2021)

    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-75.5, 42.5, -73.5, 44.5)

    result = mgr._checkCache(request_geometry, request_snapped,
                             var='tmin', start_year=2020, end_year=2022)
    assert result is None


def test_checkCache_no_cache_category(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(cache_category=None)

    # With cache_category=None, _checkCache always returns None
    result = mgr._checkCache((-75.5, 42.5, -73.5, 44.5), (-76.0, 42.0, -73.0, 45.0))
    assert result is None


def test_checkCache_exact_target_returned(tmp_path, monkeypatch):
    """_checkCache short-circuits and returns the exact file if it already exists."""
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(has_varname=False, is_temporal=False)

    snapped = (-76.0, 42.0, -73.0, 45.0)
    path = _touch_cache(mgr, snapped)

    result = mgr._checkCache((-75.5, 42.5, -73.5, 44.5), snapped)
    assert result == path


# ---------------------------------------------------------------------------
# has_resampling tests
# ---------------------------------------------------------------------------

def test_cacheFilename_with_resampling(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(is_temporal=True, has_resampling=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds, start_year=2020, end_year=2022,
                               temporal_resampling='1D')
    assert '1D' in os.path.basename(fname)
    assert '2020-2022' in os.path.basename(fname)


def test_cacheFilename_resampling_native(tmp_path, monkeypatch):
    """None temporal_resampling should produce 'native' in the filename."""
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(is_temporal=True, has_resampling=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds, start_year=2020, end_year=2022,
                               temporal_resampling=None)
    assert 'native' in os.path.basename(fname)


def test_parseCacheFilename_roundtrip_with_resampling(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(is_temporal=True, has_resampling=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    fname = mgr._cacheFilename(bounds, start_year=2020, end_year=2022,
                               temporal_resampling='1D')
    parsed = mgr._parseCacheFilename(os.path.basename(fname))
    assert parsed is not None
    assert parsed['start_year'] == 2020
    assert parsed['end_year'] == 2022
    assert parsed['temporal_resampling'] == '1D'


def test_checkCache_resampling_exact_match(tmp_path, monkeypatch):
    """Cached '1D' file should be reused for a '1D' request."""
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(is_temporal=True, has_resampling=True)

    cached_bounds = (-77.0, 41.0, -72.0, 46.0)
    _touch_cache(mgr, cached_bounds, start_year=2019, end_year=2023,
                 temporal_resampling='1D')

    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-75.5, 42.5, -73.5, 44.5)

    result = mgr._checkCache(request_geometry, request_snapped,
                             start_year=2020, end_year=2022,
                             temporal_resampling='1D')
    assert result is not None


def test_checkCache_resampling_mismatch_excluded(tmp_path, monkeypatch):
    """Cached '1D' file must NOT be reused for a '2D' request."""
    monkeypatch.setitem(
        watershed_workflow.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    mgr = _FakeManager(is_temporal=True, has_resampling=True)

    cached_bounds = (-77.0, 41.0, -72.0, 46.0)
    _touch_cache(mgr, cached_bounds, start_year=2019, end_year=2023,
                 temporal_resampling='1D')

    request_snapped = (-76.0, 42.0, -73.0, 45.0)
    request_geometry = (-75.5, 42.5, -73.5, 44.5)

    result = mgr._checkCache(request_geometry, request_snapped,
                             start_year=2020, end_year=2022,
                             temporal_resampling='2D')
    assert result is None
