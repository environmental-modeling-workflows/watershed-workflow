import pytest

import os
from distutils import dir_util
import shapely
import numpy as np
import fiona

import workflow.conf
import workflow.warp
import workflow.sources.manager_nhd
import workflow.sources.utils as sutils

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8_ll = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

#bounds4_crs = np.array(list(workflow.warp.xy(bounds4[0], bounds4[1], workflow.crs.latlon_crs(), workflow.crs.default_crs())) + list(workflow.warp.xy(bounds4[2], bounds4[3], workflow.crs.latlon_crs(), workflow.crs.default_crs())))

#bounds8_crs = np.array(list(workflow.warp.xy(bounds8[0], bounds8[1], workflow.crs.latlon_crs(), workflow.crs.default_crs())) + list(workflow.warp.xy(bounds8[2], bounds8[3], workflow.crs.latlon_crs(), workflow.crs.default_crs())))


@pytest.fixture
def wbd():
    return workflow.sources.manager_nhd.FileManagerWBD()


def test_wbd_url(wbd):
    url = wbd._url('02')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/WBD_02_HU2_GDB.zip' == url)

def test_wbd_url2(wbd):
    url = wbd._url('06')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/WBD_06_HU2_GDB.zip' == url)

def test_wbd_url3(wbd):
    url = wbd._url('14')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/WBD_14_HU2_GDB.zip' == url)

def test_wbd_url4(wbd):
    url = wbd._url('19')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/WBD_19_HU2_GDB.zip' == url)
    
def test_wbd_url_fail(wbd):
    with pytest.raises(ValueError):
        url = wbd._url('99') # this huc is not a real huc

def test_wbd_download(wbd):
    # download
    hfile = wbd._download('02', force=True)
    assert(hfile == wbd.name_manager.file_name('02'))
    
def test_wbd_get(wbd):
    # download
    profile, huc = wbd.get_huc('02040101')
    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8_ll, np.array(bounds), 1.e-6))

# hydro tests
def test_wbd_get_hydro(wbd):
    profile, huc = wbd.get_huc('020401010101')
    bounds = workflow.utils.shply(huc['geometry']).bounds

    with pytest.raises(RuntimeError):
        profile, huc = wbd.get_hydro(bounds, profile['crs'], '020401010101')
        

