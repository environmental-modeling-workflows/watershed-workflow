"""Not really sure what the best way to test these are.  For now, we
simply try converting things that we think should be the same and
making sure they are the same.
"""

import pytest
import pyproj

import watershed_workflow.crs


def epsg_harness(epsg):
    gold = watershed_workflow.crs.from_epsg(epsg)

    ppcrs2 = watershed_workflow.crs.from_proj(pyproj.crs.CRS('EPSG:{}'.format(epsg)))
    assert (watershed_workflow.crs.isEqual(gold, ppcrs2))
    
    try:
        import fiona
    except ImportError:
        pass
    else:
        fcrs = watershed_workflow.crs.from_fiona(fiona.crs.CRS.from_epsg(epsg))
        assert (watershed_workflow.crs.isEqual(gold, fcrs))

    try:
        import rasterio
    except ImportError:
        pass
    else:
        rcrs = watershed_workflow.crs.from_rasterio(rasterio.crs.CRS.from_string(
            'EPSG:{}'.format(epsg)))
        assert (watershed_workflow.crs.isEqual(gold, rcrs))

    try:
        import cartopy
    except ImportError:
        pass
    else:
        cartopy_crs = cartopy.crs.epsg(epsg)
        ccrs = watershed_workflow.crs.from_cartopy(cartopy_crs)
        assert(watershed_workflow.crs.isEqual(gold, ccrs))


def test_default():
    epsg_harness(5070)


def test_alaska():
    epsg_harness(3338)


def test_latlon():
    epsg_harness(4269)
