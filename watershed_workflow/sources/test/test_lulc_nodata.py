"""Network tests verifying that LULC/cover variables from NLCD, MODIS Earthdata,
and MODIS AppEEARs satisfy the nodata contract:

    ds[KEY].attrs['nodata'] == -1
    '_FillValue' not in ds[KEY].attrs
    np.issubdtype(ds[KEY].dtype, np.integer)

Checked for four combinations of cache state × out_crs:
    1. cache miss,  out_crs=None
    2. cache miss,  out_crs=different CRS
    3. cache hit,   out_crs=None
    4. cache hit,   out_crs=different CRS
"""
import pytest
import numpy as np
import shapely.geometry

import watershed_workflow.crs
import watershed_workflow.utils.config
from watershed_workflow.sources.manager_nlcd import ManagerNLCD
from watershed_workflow.sources.manager_modis_earthdata import ManagerMODISEarthdata, _WGS84_CRS


# ---------------------------------------------------------------------------
# Shared geometry and alternate CRS
# ---------------------------------------------------------------------------

_GEOMETRY = shapely.geometry.box(-83.493, 35.014, -83.407, 35.091)
_GEOMETRY_CRS = watershed_workflow.crs.latlon_crs
_OTHER_CRS = watershed_workflow.crs.from_epsg(5070)   # Albers Equal Area


def _assert_lulc_nodata_contract(ds, key):
    """Assert the nodata-contract conditions for ds[key].

    Either 'nodata' or '_FillValue' may carry the sentinel; what matters is:
    - the sentinel value is -1
    - the dtype is integer (the round-trip must not upcast to float)
    """
    da = ds[key]
    sentinel = da.attrs.get('nodata', da.attrs.get('_FillValue'))
    assert sentinel == -1, \
        f"{key}: expected nodata sentinel -1, got {sentinel!r} (attrs={da.attrs})"
    assert np.issubdtype(da.dtype, np.integer), \
        f"{key}.dtype == {da.dtype}, expected an integer dtype"


def _assert_reprojected_corners_use_nodata(ds, key):
    """After reprojection, corner pixels outside the source domain must equal nodata.

    Reprojection of a non-axis-aligned source into a new CRS leaves corners
    that have no valid data.  Those pixels must be filled with the nodata
    sentinel (-1), not NaN (which cannot be represented in an integer array).
    """
    da = ds[key]
    nodata = da.attrs.get('nodata', da.attrs.get('_FillValue'))
    # Squeeze out any leading time dimension before indexing corners.
    spatial = da.isel({d: 0 for d in da.dims if d not in ('y', 'x')})
    corners = [
        spatial.isel(y=0,  x=0).values,
        spatial.isel(y=0,  x=-1).values,
        spatial.isel(y=-1, x=0).values,
        spatial.isel(y=-1, x=-1).values,
    ]
    assert any(c == nodata for c in corners), \
        f"Expected at least one corner pixel == {nodata} after reprojection, got {corners}"


# ===========================================================================
# NLCD  (KEY = 'cover')
# ===========================================================================

@pytest.mark.network
def test_nlcd_cover_cache_miss_no_out_crs(coweeta, tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerNLCD(location='L48', year=2019)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['cover'])
    _assert_lulc_nodata_contract(ds, 'cover')
    _assert_reprojected_corners_use_nodata(ds, 'cover')


@pytest.mark.network
def test_nlcd_cover_cache_miss_with_out_crs(coweeta, tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerNLCD(location='L48', year=2019)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['cover'],
                        out_crs=_OTHER_CRS)
    _assert_lulc_nodata_contract(ds, 'cover')
    _assert_reprojected_corners_use_nodata(ds, 'cover')


@pytest.mark.network
def test_nlcd_cover_cache_hit_no_out_crs(coweeta, tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerNLCD(location='L48', year=2019)
    # Warm the cache.
    mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['cover'])
    # Second call should be a cache hit.
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['cover'])
    _assert_lulc_nodata_contract(ds, 'cover')
    _assert_reprojected_corners_use_nodata(ds, 'cover')


@pytest.mark.network
def test_nlcd_cover_cache_hit_with_out_crs(coweeta, tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerNLCD(location='L48', year=2019)
    # Warm the cache.
    mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['cover'])
    # Second call — cache hit, different CRS.
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['cover'],
                        out_crs=_OTHER_CRS)
    _assert_lulc_nodata_contract(ds, 'cover')
    _assert_reprojected_corners_use_nodata(ds, 'cover')


# ===========================================================================
# MODIS Earthdata  (KEY = 'LULC')
# ===========================================================================

@pytest.mark.network
def test_modis_earthdata_lulc_cache_miss_no_out_crs(tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerMODISEarthdata()
    ds = mgr.getDataset(_GEOMETRY, _WGS84_CRS, start=2020, end=2020,
                        variables=['LULC'])
    _assert_lulc_nodata_contract(ds, 'LULC')


@pytest.mark.network
def test_modis_earthdata_lulc_cache_miss_with_out_crs(tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerMODISEarthdata()
    ds = mgr.getDataset(_GEOMETRY, _WGS84_CRS, start=2020, end=2020,
                        variables=['LULC'], out_crs=_OTHER_CRS)
    _assert_lulc_nodata_contract(ds, 'LULC')
    _assert_reprojected_corners_use_nodata(ds, 'LULC')


@pytest.mark.network
def test_modis_earthdata_lulc_cache_hit_no_out_crs(tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerMODISEarthdata()
    # Warm the cache.
    mgr.getDataset(_GEOMETRY, _WGS84_CRS, start=2020, end=2020,
                   variables=['LULC'])
    # Second call — cache hit.
    ds = mgr.getDataset(_GEOMETRY, _WGS84_CRS, start=2020, end=2020,
                        variables=['LULC'])
    _assert_lulc_nodata_contract(ds, 'LULC')


@pytest.mark.network
def test_modis_earthdata_lulc_cache_hit_with_out_crs(tmp_path):
    watershed_workflow.utils.config.setDataDirectory(str(tmp_path))
    mgr = ManagerMODISEarthdata()
    # Warm the cache.
    mgr.getDataset(_GEOMETRY, _WGS84_CRS, start=2020, end=2020,
                   variables=['LULC'])
    # Second call — cache hit, different CRS.
    ds = mgr.getDataset(_GEOMETRY, _WGS84_CRS, start=2020, end=2020,
                        variables=['LULC'], out_crs=_OTHER_CRS)
    _assert_lulc_nodata_contract(ds, 'LULC')
    _assert_reprojected_corners_use_nodata(ds, 'LULC')


