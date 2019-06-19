import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.sources.manager_ned
import workflow.sources.manager_nhdplus

    
def test_ned1():
    # single file covers it
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('020401010101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED()
    dem_prof, dem = ned.get_dem(profile, huc)
    assert(dem.shape == (1396, 1390))

def test_ned2():
    # requires tiles
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED()
    dem_prof, dem = ned.get_dem(profile, huc)

    assert(dem.shape == (10527, 10953))


def test_ned3():
    # single file covers it
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('020401010101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED('1 arc-second')
    dem_prof, dem = ned.get_dem(profile, huc)
    assert(dem.shape == (466,464))

def test_ned4():
    # requires tiles
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    ned = workflow.sources.manager_ned.FileManagerNED('1 arc-second')
    dem_prof, dem = ned.get_dem(profile, huc)
    assert(dem.shape == (3509, 3651))

    
