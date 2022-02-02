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
    return watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()

# having some robustness issues, lets just test a bunch
def test_nhdplus_url1(nhd):
    url = nhd._url('0204')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_0204_HU4_GDB.zip' == url)

def test_nhdplus_url2(nhd):
    url = nhd._url('0601')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_0601_HU4_GDB.zip' == url)

def test_nhdplus_url3(nhd):
    url = nhd._url('1402')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_1402_HU4_GDB.zip' == url)

# def test_nhdplus_url4(nhd):
#     url = nhd._url('1906')
#     assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_1906_HU4_GDB.zip' == url)

def test_nhdplus_url_fail(nhd):
    with pytest.raises(ValueError):
        url = nhd._url('0201') # this huc was removed

def test_nhdplus_url_invalid(nhd):
    with pytest.raises(ValueError):
        h = nhd.get_hucs('02', 2) # not allowed, on 4 only

def test_nhdplus_download(nhd):
    # download
    hfile = nhd._download('0204', force=True)
    assert(hfile == nhd.name_manager.file_name('0204'))

def test_nhdplus2(nhd):
    # download
    profile, hucs = nhd.get_hucs('0204',4)
    bounds = watershed_workflow.utils.shply(hucs[0]['geometry']).bounds
    assert(np.allclose(bounds4_ll, np.array(bounds), 1.e-6))


def test_nhdplus3(nhd):
    # download
    profile, huc = nhd.get_huc('02040101')
    bounds = watershed_workflow.utils.shply(huc['geometry']).bounds
    print(bounds)
    print(bounds8_ll)
    assert(np.allclose(bounds8_ll, np.array(bounds), 1.e-6))


def test_nhdplus4(nhd):
    # download
    profile, huc8s = nhd.get_hucs('0204', 8)

    
# hydro tests
def test_nhdplus10(nhd):
    # download hydrography
    profile, huc = nhd.get_huc('020401010101')
    profile, rivers = nhd.get_hydro('020401010101')
    assert(574 == len(rivers))

    
    

    




