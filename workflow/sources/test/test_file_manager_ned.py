import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.sources.manager_ned
import workflow.sources.manager_nhd

    
def test_ned1():
    # single file covers it
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('020401010101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED()
    dem_prof, dem = ned.get_raster(huc, workflow.crs.from_fiona(profile['crs']))
    assert((1612, 1606) == dem.shape)

def test_ned2():
    # requires tiles
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED()
    dem_prof, dem = ned.get_raster(huc, workflow.crs.from_fiona(profile['crs']))

    assert((10743, 11169) == dem.shape)


def test_ned3():
    # single file covers it
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('020401010101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED('1 arc-second')
    dem_prof, dem = ned.get_raster(huc, workflow.crs.from_fiona(profile['crs']))
    assert((538, 536) == dem.shape)

def test_ned4():
    # requires tiles
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED('1 arc-second')
    dem_prof, dem = ned.get_raster(huc, workflow.crs.from_fiona(profile['crs']))
    assert((3581, 3723) == dem.shape)

    
