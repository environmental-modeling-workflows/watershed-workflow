import pytest

import os
from distutils import dir_util
import shapely
import numpy as np
import fiona

import watershed_workflow.config
import watershed_workflow.warp
import watershed_workflow.sources.manager_nhd
import watershed_workflow.sources.utils as sutils

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8_ll = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

#bounds4_crs = np.array(list(watershed_workflow.warp.xy(bounds4[0], bounds4[1], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())) + list(watershed_workflow.warp.xy(bounds4[2], bounds4[3], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())))

#bounds8_crs = np.array(list(watershed_workflow.warp.xy(bounds8[0], bounds8[1], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())) + list(watershed_workflow.warp.xy(bounds8[2], bounds8[3], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())))


@pytest.fixture
def nhd():
    return watershed_workflow.sources.manager_nhd.FileManagerNHD()


def test_nhd_url(nhd):
    url = nhd._url('02040101')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/HighResolution/GDB/NHD_H_02040101_HU8_GDB.zip' == url)

def test_nhd_url2(nhd):
    url = nhd._url('06010202')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/HighResolution/GDB/NHD_H_06010202_HU8_GDB.zip' == url)

def test_nhd_url3(nhd):
    url = nhd._url('14020001')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/HighResolution/GDB/NHD_H_14020001_HU8_GDB.zip' == url)

def test_nhd_url4(nhd):
    url = nhd._url('19060402')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/HighResolution/GDB/NHD_H_19060402_HU8_GDB.zip' == url)
    
    
def test_nhd_url_fail(nhd):
    with pytest.raises(ValueError):
        url = nhd._url('02010000') # this huc is not a real huc

def test_nhd_url_invalid(nhd):
    with pytest.raises(ValueError):
        h = nhd.get_hucs('0201', 2) # not allowed, on 8 only

def test_nhd_download(nhd):
    # download
    hfile = nhd._download('02040101', force=True)
    assert(hfile == nhd.name_manager.file_name('02040101'))
    
def test_nhd_get(nhd):
    # download
    profile, huc = nhd.get_huc('02040101')
    bounds = watershed_workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8_ll, np.array(bounds), 1.e-6))

# hydro tests
def test_nhd_get_hydro(nhd):
    profile, rivers = nhd.get_hydro('020401010101')
    assert(571 == len(rivers)) # note this is different from NHDPlus
