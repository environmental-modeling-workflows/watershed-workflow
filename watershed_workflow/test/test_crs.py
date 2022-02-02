"""Not really sure what the best way to test these are.  For now, we
simply try converting things that we think should be the same and
making sure they are the same.
"""

import pytest

import fiona.crs
import rasterio.crs
import pyproj
import cartopy.crs

import workflow.crs

def epsg_harness(epsg, test_cartopy=True):
    gold = workflow.crs.from_epsg(epsg)

    fcrs = workflow.crs.from_fiona(fiona.crs.from_epsg(epsg))
    rcrs = workflow.crs.from_rasterio(rasterio.crs.CRS.from_string('EPSG:{}'.format(epsg)))
    ppcrs2 = workflow.crs.from_proj(pyproj.crs.CRS('EPSG:{}'.format(epsg)))

    # print(f'gold: {gold}')
    # print(f'fiona: {fcrs}')
    # print(f'rasterio: {rcrs}')
    # print(f'proj: {ppcrs2}')
    
    assert(workflow.crs.equal(gold, fcrs))
    assert(workflow.crs.equal(gold, rcrs))
    assert(workflow.crs.equal(gold, ppcrs2))

    if test_cartopy:
        ccrs = workflow.crs.from_cartopy(cartopy.crs.epsg(epsg))
        #assert(workflow.crs.equal(gold, ccrs))

        
    

def test_default():
    epsg_harness(5070)

def test_alaska():
    epsg_harness(3338)

def test_latlon():
    epsg_harness(4269, False)
    
