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

#bounds4_crs = np.array(list(workflow.warp.warp_xy(bounds4[0], bounds4[1], workflow.conf.latlon_crs(), workflow.conf.default_crs())) + list(workflow.warp.warp_xy(bounds4[2], bounds4[3], workflow.conf.latlon_crs(), workflow.conf.default_crs())))

#bounds8_crs = np.array(list(workflow.warp.warp_xy(bounds8[0], bounds8[1], workflow.conf.latlon_crs(), workflow.conf.default_crs())) + list(workflow.warp.warp_xy(bounds8[2], bounds8[3], workflow.conf.latlon_crs(), workflow.conf.default_crs())))


@pytest.fixture
def nhd():
    return workflow.sources.manager_nhd.FileManagerNHD()


def test_nhd_url(nhd):
    url = nhd._url('02040101')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/HighResolution/GDB/NHD_H_02040101_HU8_GDB.zip' == url)

def test_nhd_url_fail(nhd):
    with pytest.raises(ValueError):
        url = nhd._url('02010000') # this huc is not a real huc

def test_nhd_url_invalid(nhd):
    with pytest.raises(ValueError):
        h = nhd.get_hucs('0201', 2) # not allowed, on 8 only

def test_nhd_download(nhd):
    # download
    hfile = nhd._download('02040101')
    assert(hfile == nhd.name_manager.file_name('02040101'))
    
def test_nhd_get(nhd):
    # download
    profile, huc = nhd.get_huc('02040101')
    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8_ll, np.array(bounds), 1))

# hydro tests
def test_nhd_get_hydro(nhd):
    profile, huc = nhd.get_huc('020401010101')
    bounds = workflow.utils.shply(huc['geometry']).bounds
    profile, rivers = nhd.get_hydro(bounds, profile['crs'], '020401010101')
    assert(571 == len(rivers)) # note this is different from NHDPlus
