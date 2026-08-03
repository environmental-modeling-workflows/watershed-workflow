"""Tests for ManagerMODISEarthdata.

Offline tests (no network, no earthaccess auth) cover constructor properties,
cache filename generation, HDF date parsing, variable validation, and the
short-circuit path when a cache hit is detected.

Network tests (marked ``@pytest.mark.network``) exercise live earthaccess
search and full-stack download over the Coweeta watershed.
"""
import datetime
import os
from unittest.mock import patch

import cftime
import numpy as np
import pytest
import shapely.geometry
import xarray as xr

import watershed_workflow.crs
import watershed_workflow.utils.config
from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_modis_earthdata import (
    ManagerMODISEarthdata,
    _parseHdfDate,
    _MODIS_SINU_CRS,
    _WGS84_CRS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr(tmp_path):
    """ManagerMODISEarthdata with data directory pointing at a temp folder."""
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    return ManagerMODISEarthdata()


@pytest.fixture
def coweeta_bounds():
    """Approximate WGS84 bounding box of the Coweeta watershed."""
    # (xmin, ymin, xmax, ymax) in degrees
    return (-83.493, 35.014, -83.407, 35.091)


@pytest.fixture
def coweeta_box(coweeta_bounds):
    """Shapely box for Coweeta in WGS84."""
    xmin, ymin, xmax, ymax = coweeta_bounds
    return shapely.geometry.box(xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------

def test_manager_properties(mgr):
    """Verify product, source, CRS, variable lists, and cache metadata."""
    assert mgr.product == 'MODIS'
    assert mgr.source == 'NASA Earthdata'

    assert watershed_workflow.crs.isEqual(mgr.native_crs_in, _WGS84_CRS)
    assert watershed_workflow.crs.isEqual(mgr.native_crs_out, _MODIS_SINU_CRS)

    assert set(mgr.valid_variables) == {'LAI', 'LULC'}
    assert set(mgr.default_variables) == {'LAI', 'LULC'}

    assert mgr.attrs.category == 'land_cover'
    assert mgr.attrs.is_temporal


def test_cache_dirname_generation(mgr, tmp_path, monkeypatch):
    """cacheDirname produces a path with the expected tokens."""
    from watershed_workflow.sources.cache_info import cacheDirname, snapBounds
    monkeypatch.setitem(watershed_workflow.utils.config.rcParams['DEFAULT'],
                        'data_directory', str(tmp_path))
    bounds = (-83.5, 35.0, -83.4, 35.1)
    dirpath = cacheDirname(mgr.attrs, bounds, start_year=2020, end_year=2020)

    dirname = os.path.basename(dirpath)
    assert '2020-2020' in dirname
    assert 'nasa_earthdata' in dirname
    # Snapped coordinate tokens present — snap before asserting
    xmin, ymin, xmax, ymax = snapBounds(bounds, mgr.native_resolution)
    assert f'{xmin:.4f}' in dirname
    assert f'{ymin:.4f}' in dirname
    assert f'{xmax:.4f}' in dirname
    assert f'{ymax:.4f}' in dirname


def test_parse_hdf_date_lai():
    """_parseHdfDate returns the correct date for a known LAI filename."""
    fname = 'MCD15A3H.A2020001.h11v05.061.2020010040000.hdf'
    result = _parseHdfDate(fname)
    assert result == datetime.date(2020, 1, 1)


def test_parse_hdf_date_leap_year():
    """_parseHdfDate handles day-of-year 366 in a leap year correctly."""
    fname = 'MCD15A3H.A2020366.h11v05.061.2021010040000.hdf'
    result = _parseHdfDate(fname)
    assert result == datetime.date(2020, 12, 31)


def test_parse_hdf_date_lulc():
    """_parseHdfDate works for LULC filenames too."""
    fname = 'MCD12Q1.A2020001.h11v05.061.2021012152954.hdf'
    result = _parseHdfDate(fname)
    assert result == datetime.date(2020, 1, 1)


def test_parse_hdf_date_invalid():
    """_parseHdfDate raises ValueError when the date token is absent."""
    with pytest.raises(ValueError, match='Cannot parse acquisition date'):
        _parseHdfDate('not_a_modis_filename.hdf')


def test_invalid_variable_error(mgr, coweeta_box):
    """getDataset raises ValueError for an unrecognised variable name."""
    with pytest.raises(ValueError, match='Invalid variable'):
        mgr.getDataset(
            coweeta_box, _WGS84_CRS,
            start=2020, end=2020,
            variables=['NDVI'],
        )


def test_default_variables(mgr, coweeta_box):
    """Calling _preprocessParameters without variables uses the default list."""
    request = mgr._preprocessParameters(
        coweeta_box, _WGS84_CRS, 2020, 2020, None, None, None
    )
    assert set(request.variables) == {'LAI', 'LULC'}


def test_download_not_called_when_cached(mgr, coweeta_box, tmp_path, monkeypatch):
    """_downloadVar is never invoked when the cache directory is complete."""
    import shutil
    from watershed_workflow.sources.cache_info import cacheDirname
    monkeypatch.setitem(watershed_workflow.utils.config.rcParams['DEFAULT'],
                        'data_directory', str(tmp_path))

    bounds = coweeta_box.buffer(3 * mgr.native_resolution).bounds
    cache_dir = cacheDirname(mgr.attrs, bounds, start_year=2020, end_year=2020)
    os.makedirs(cache_dir, exist_ok=True)
    # Create stub LAI.nc so isComplete returns True
    xr.Dataset({'LAI': xr.DataArray([1.0])}).to_netcdf(os.path.join(cache_dir, 'LAI.nc'))

    with patch.object(mgr, '_downloadVar') as mock_dl:
        request = mgr.requestDataset(
            coweeta_box, _WGS84_CRS,
            start=2020, end=2020,
            variables=['LAI'],
        )

    assert request._cache_hit
    mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_earthaccess_search_coweeta():
    """earthaccess.search_data finds at least one MCD12Q1 granule over Coweeta."""
    import earthaccess  # type: ignore[import]

    earthaccess.login(strategy='netrc')
    granules = earthaccess.search_data(
        short_name='MCD12Q1',
        version='061',
        bounding_box=(-83.493, 35.014, -83.407, 35.091),
        temporal=('2020-01-01', '2020-12-31'),
        count=10,
    )
    assert len(granules) >= 1


@pytest.mark.network
def test_getDataset_lai_coweeta(coweeta, tmp_path):
    """Full integration: LAI over Coweeta 2020 returns a valid Dataset."""
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerMODISEarthdata()

    ds = mgr.getDataset(
        coweeta.geometry[0], coweeta.crs,
        start=2020, end=2020,
        variables=['LAI'],
    )

    assert isinstance(ds, xr.Dataset)
    assert 'LAI' in ds.data_vars
    assert 'time_LAI' in ds.dims
    # Values should be non-negative and below a physical upper bound (~10 m²/m²)
    valid = ds['LAI'].values[~np.isnan(ds['LAI'].values)]
    assert valid.size > 0
    assert np.all(valid >= 0.0)
    assert np.all(valid <= 10.0)
    # Dataset carries product/source attributes set by _postprocessDataset
    assert ds.attrs.get('product') == 'MODIS'


@pytest.mark.network
def test_getDataset_lulc_coweeta(coweeta, tmp_path):
    """Full integration: LULC over Coweeta 2020 returns valid class integers."""
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerMODISEarthdata()

    ds = mgr.getDataset(
        coweeta.geometry[0], coweeta.crs,
        start=2020, end=2020,
        variables=['LULC'],
    )

    assert isinstance(ds, xr.Dataset)
    assert 'LULC' in ds.data_vars
    assert 'time_LULC' in ds.dims

    valid = ds['LULC'].values[~np.isnan(ds['LULC'].values)].astype(int)
    assert valid.size > 0
    # MODIS LC_Type1 classes are 0-17; fill (255) must have been masked out
    assert np.all(valid >= 0)
    assert np.all(valid <= 17)
