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

bounds4_ll = np.array([-85.19249725424618, 34.903371461448046, -81.26111840201514, 37.24054551094525])
bounds8_ll = np.array([-83.82022857720955, 34.903371461448046, -83.14341380430176, 35.58000945935606])


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

def test_nhdplus_url_fail(nhd):
    with pytest.raises(ValueError):
        url = nhd._url('0201') # this huc was removed

def test_nhdplus_url_invalid(nhd):
    with pytest.raises(ValueError):
        h = nhd.get_hucs('02', 2) # not allowed, on 4 only

def test_nhdplus_download(nhd):
    # download
    hfile = nhd._download('0601', force=True)
    assert(hfile == nhd.name_manager.file_name('0601'))

def test_nhdplus2(nhd):
    # download
    profile, hucs = nhd.get_hucs('0601',4)
    bounds = watershed_workflow.utils.shply(hucs[0]['geometry']).bounds
    assert(np.allclose(bounds4_ll, np.array(bounds), 1.e-6))


def test_nhdplus3(nhd):
    # download
    profile, huc = nhd.get_huc('06010202')
    bounds = watershed_workflow.utils.shply(huc['geometry']).bounds
    print(bounds)
    print(bounds8_ll)
    assert(np.allclose(bounds8_ll, np.array(bounds), 1.e-6))


def test_nhdplus4(nhd):
    # download
    profile, huc8s = nhd.get_hucs('0601', 8)

    
# hydro tests
def test_nhdplus12(nhd):
    # download hydrography
    profile, huc = nhd.get_huc('060102020103')
    profile, rivers = nhd.get_hydro('060102020103')
    assert(202 == len(rivers))

def test_vaa(nhd):
    profile, reaches = nhd.get_hydro('060102020103',
                                    properties=['HydrologicSequence',
                                                'DownstreamMainPathHydroSeq',
                                                'TotalDrainageAreaSqKm',
                                                'CatchmentAreaSqKm'])

    

    




