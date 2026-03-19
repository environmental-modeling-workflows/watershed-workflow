"""Unit tests for cache_info free functions.

Tests snapBounds, cacheDirname, parseCacheDirname, and findCacheDir.
"""
import os
import pytest

import watershed_workflow.utils.config
from watershed_workflow.sources.cache_info import (
    snapBounds, cacheFolder, cacheDirname, parseCacheDirname, findCacheDir,
)
from watershed_workflow.sources.manager import ManagerAttributes


# ---------------------------------------------------------------------------
# Minimal isComplete implementations for testing findCacheDir
# ---------------------------------------------------------------------------

class _AlwaysComplete:
    """Fake manager whose isComplete always returns True."""
    def isComplete(self, dir, request):
        return True


class _NeverComplete:
    """Fake manager whose isComplete always returns False."""
    def isComplete(self, dir, request):
        return False


class _FakeRequest:
    """Minimal request stub."""
    def __init__(self, variables=None):
        self.variables = variables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attrs(category='climate', product='AORC 4km', source='ORNL DAAC',
                product_short='aorc', source_short='aorc_4km',
                native_resolution=1.0,
                is_temporal=False, is_resampled=False):
    return ManagerAttributes(
        category=category,
        product=product,
        source=source,
        description='test dataset',
        product_short=product_short,
        source_short=source_short,
        native_resolution=native_resolution,
        is_temporal=is_temporal,
        is_resampled=is_resampled,
    )


def _mkdir_cache(attrs, bounds, start_year=None, end_year=None,
                 temporal_resampling=None):
    """Create an empty directory at the canonical cache path."""
    path = cacheDirname(attrs, bounds, start_year=start_year,
                        end_year=end_year,
                        temporal_resampling=temporal_resampling)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# snapBounds tests
# ---------------------------------------------------------------------------

def testsnapBounds_positive_coords():
    bounds = (1.1, 2.2, 3.3, 4.4)
    snapped = snapBounds(bounds, step=1.0)
    xmin, ymin, xmax, ymax = snapped
    assert xmin <= 1.1
    assert ymin <= 2.2
    assert xmax >= 3.3
    assert ymax >= 4.4
    assert xmin % 1.0 == pytest.approx(0.0)
    assert ymin % 1.0 == pytest.approx(0.0)
    assert xmax % 1.0 == pytest.approx(0.0)
    assert ymax % 1.0 == pytest.approx(0.0)


def testsnapBounds_negative_coords():
    bounds = (-75.3, -12.7, -74.1, -11.2)
    snapped = snapBounds(bounds, step=1.0)
    xmin, ymin, xmax, ymax = snapped
    assert xmin <= -75.3
    assert ymin <= -12.7
    assert xmax >= -74.1
    assert ymax >= -11.2
    assert xmin == pytest.approx(-76.0)
    assert ymin == pytest.approx(-13.0)
    assert xmax == pytest.approx(-74.0)
    assert ymax == pytest.approx(-11.0)


def testsnapBounds_idempotent():
    bounds = (-76.0, -13.0, -74.0, -11.0)
    snapped = snapBounds(bounds, step=1.0)
    assert snapped == pytest.approx(bounds)
    assert snapBounds(snapped, step=1.0) == pytest.approx(snapped)


# ---------------------------------------------------------------------------
# cacheDirname tests
# ---------------------------------------------------------------------------

def test_cacheDirname_no_time(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds)
    assert 'aorc_4km' in os.path.basename(d)
    assert '-76.0000' in os.path.basename(d)
    assert '-73.0000' in os.path.basename(d)


def test_cacheDirname_with_time(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(product='DayMet 1km', product_short='daymet',
                        source_short='daymet_1km', is_temporal=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds, start_year=2020, end_year=2022)
    assert '2020-2022' in os.path.basename(d)


def test_cacheDirname_with_resampling(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(is_temporal=True, is_resampled=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds, start_year=2020, end_year=2022,
                     temporal_resampling='1D')
    name = os.path.basename(d)
    assert '1D' in name
    assert '2020-2022' in name


def test_cacheDirname_resampling_none_becomes_native(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(is_temporal=True, is_resampled=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds, start_year=2020, end_year=2022,
                     temporal_resampling=None)
    assert 'native' in os.path.basename(d)


def test_cacheDirname_snaps_bounds(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    # Raw fractional bounds — should be snapped to integers
    d = cacheDirname(attrs, (-75.3, 42.7, -73.1, 44.9))
    name = os.path.basename(d)
    assert '-76.0000' in name
    assert '42.0000' in name
    assert '-73.0000' in name
    assert '45.0000' in name


# ---------------------------------------------------------------------------
# parseCacheDirname tests
# ---------------------------------------------------------------------------

def test_parseCacheDirname_roundtrip_no_time(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds)
    parsed = parseCacheDirname(attrs, os.path.basename(d))
    assert parsed is not None
    assert parsed['xmin'] == pytest.approx(-76.0)
    assert parsed['ymin'] == pytest.approx(42.0)
    assert parsed['xmax'] == pytest.approx(-73.0)
    assert parsed['ymax'] == pytest.approx(45.0)
    assert 'start_year' not in parsed


def test_parseCacheDirname_roundtrip_with_time(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(product='DayMet 1km', product_short='daymet',
                        source_short='daymet_1km', is_temporal=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds, start_year=2020, end_year=2022)
    parsed = parseCacheDirname(attrs, os.path.basename(d))
    assert parsed is not None
    assert parsed['start_year'] == 2020
    assert parsed['end_year'] == 2022


def test_parseCacheDirname_roundtrip_with_resampling(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(is_temporal=True, is_resampled=True)
    bounds = (-76.0, 42.0, -73.0, 45.0)
    d = cacheDirname(attrs, bounds, start_year=2020, end_year=2022,
                     temporal_resampling='1D')
    parsed = parseCacheDirname(attrs, os.path.basename(d))
    assert parsed is not None
    assert parsed['start_year'] == 2020
    assert parsed['end_year'] == 2022
    assert parsed['temporal_resampling'] == '1D'


def test_parseCacheDirname_no_match():
    attrs = _make_attrs()
    result = parseCacheDirname(attrs, 'some_random_directory_name')
    assert result is None


def test_parseCacheDirname_wrong_slug():
    attrs = _make_attrs()
    result = parseCacheDirname(attrs, 'daymet_1km_-76.0000_42.0000_-73.0000_45.0000')
    assert result is None


# ---------------------------------------------------------------------------
# findCacheDir tests
# ---------------------------------------------------------------------------

def test_findCacheDir_exact_hit(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    bounds = (-76.0, 42.0, -73.0, 45.0)
    _mkdir_cache(attrs, bounds)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, bounds, mgr, req)
    assert result is not None
    assert os.path.isdir(result)


def test_findCacheDir_spatial_superset(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    # Cached directory covers larger area
    _mkdir_cache(attrs, (-77.0, 41.0, -72.0, 46.0))
    # Request is strictly inside
    req_bounds = (-75.5, 42.5, -73.5, 44.5)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req)
    assert result is not None


def test_findCacheDir_no_match_spatial(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    # Cached covers smaller area than request
    _mkdir_cache(attrs, (-75.0, 43.0, -74.0, 44.0))
    req_bounds = (-76.0, 42.0, -73.0, 45.0)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req)
    assert result is None


def test_findCacheDir_incomplete_excluded(tmp_path, monkeypatch):
    """isComplete=False on a spatial superset should not be returned."""
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    _mkdir_cache(attrs, (-77.0, 41.0, -72.0, 46.0))
    req_bounds = (-75.5, 42.5, -73.5, 44.5)
    mgr = _NeverComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req)
    assert result is None


def test_findCacheDir_temporal_superset(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(product='DayMet 1km', product_short='daymet',
                        source_short='daymet_1km', is_temporal=True)
    # Cached covers years 2019-2023
    _mkdir_cache(attrs, (-77.0, 41.0, -72.0, 46.0), start_year=2019, end_year=2023)
    req_bounds = (-75.5, 42.5, -73.5, 44.5)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req, start_year=2020, end_year=2022)
    assert result is not None


def test_findCacheDir_temporal_no_span(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(product='DayMet 1km', product_short='daymet',
                        source_short='daymet_1km', is_temporal=True)
    # Cached covers only 2020-2021, request is 2020-2022
    _mkdir_cache(attrs, (-77.0, 41.0, -72.0, 46.0), start_year=2020, end_year=2021)
    req_bounds = (-75.5, 42.5, -73.5, 44.5)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req, start_year=2020, end_year=2022)
    assert result is None


def test_findCacheDir_resampling_exact_match(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(is_temporal=True, is_resampled=True)
    _mkdir_cache(attrs, (-77.0, 41.0, -72.0, 46.0), start_year=2019, end_year=2023,
                 temporal_resampling='1D')
    req_bounds = (-75.5, 42.5, -73.5, 44.5)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req,
                          start_year=2020, end_year=2022, temporal_resampling='1D')
    assert result is not None


def test_findCacheDir_resampling_mismatch(tmp_path, monkeypatch):
    """A '1D' cache must NOT satisfy a '2D' request."""
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs(is_temporal=True, is_resampled=True)
    _mkdir_cache(attrs, (-77.0, 41.0, -72.0, 46.0), start_year=2019, end_year=2023,
                 temporal_resampling='1D')
    req_bounds = (-75.5, 42.5, -73.5, 44.5)
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, req_bounds, mgr, req,
                          start_year=2020, end_year=2022, temporal_resampling='2D')
    assert result is None


def test_findCacheDir_no_cache_folder(tmp_path, monkeypatch):
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'], 'data_directory', str(tmp_path))
    attrs = _make_attrs()
    # Cache folder doesn't exist yet
    mgr = _AlwaysComplete()
    req = _FakeRequest()
    result = findCacheDir(attrs, (-76.0, 42.0, -73.0, 45.0), mgr, req)
    assert result is None
