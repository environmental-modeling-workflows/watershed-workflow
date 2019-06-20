import pytest

import os
from distutils import dir_util
import shapely
import numpy as np
import fiona

import workflow.conf
import workflow.warp
import workflow.sources.manager_nhdplus
import workflow.sources.utils as sutils

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8_ll = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

#bounds4_crs = np.array(list(workflow.warp.warp_xy(bounds4[0], bounds4[1], workflow.conf.latlon_crs(), workflow.conf.default_crs())) + list(workflow.warp.warp_xy(bounds4[2], bounds4[3], workflow.conf.latlon_crs(), workflow.conf.default_crs())))

#bounds8_crs = np.array(list(workflow.warp.warp_xy(bounds8[0], bounds8[1], workflow.conf.latlon_crs(), workflow.conf.default_crs())) + list(workflow.warp.warp_xy(bounds8[2], bounds8[3], workflow.conf.latlon_crs(), workflow.conf.default_crs())))


def test_nhdplus():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    with pytest.raises(ValueError):
        url = nhd._url('0201') # this huc was removed
    with pytest.raises(ValueError):
        h = nhd.get_hucs('02', 2) # not allowed, on 4 only

    url = nhd._url('0204')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlus/HU4/HighResolution/GDB/NHDPLUS_H_0204_HU4_GDB.zip' == url)

    # download
    hfile = nhd._download('0204')
    assert(hfile == nhd.names.file_name('0204'))

def test_nhdplus2():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    # download
    profile, hucs = nhd.get_hucs('0204',4)
    bounds = workflow.utils.shply(hucs[0]['geometry']).bounds
    assert(np.allclose(bounds4_ll, np.array(bounds), 1))


def test_nhdplus3():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    # download
    profile, huc = nhd.get_huc('02040101')
    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8_ll, np.array(bounds), 1))


def test_nhdplus4():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    # download
    profile, huc8s = nhd.get_hucs('0204', 8)

    
def test_nhdplus6():
    # test in a different crs
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8_ll, bounds, 1.e-4))

# hydro tests
def test_nhdplus10():
    # download hydrography
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    profile, huc = nhd.get_huc('020401010101')
    bounds = workflow.utils.shply(huc['geometry']).bounds
    

    profile, rivers = nhd.get_hydro(bounds, profile['crs'], '020401010101')
    assert(575 == len(rivers))

    
    

    




