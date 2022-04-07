import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
import watershed_workflow.sources.manager_ned
import watershed_workflow.sources.manager_nhd

    
# these just take too long    
# def test_ned1():
#     # single file covers it
#     nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
#     profile, huc = nhd.get_huc('020401010101')

#     # get imgs
#     ned = watershed_workflow.sources.manager_ned.FileManagerNED()

#     # force the download here to test the API hasn't changed
#     dem_prof, dem = ned.get_raster(huc, watershed_workflow.crs.from_fiona(profile['crs']), force_download=True)
#     assert((1612, 1606) == dem.shape)

# def test_ned2():
#     # requires tiles
#     nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
#     profile, huc = nhd.get_huc('02040101')

#     # get imgs
#     ned = watershed_workflow.sources.manager_ned.FileManagerNED()
#     dem_prof, dem = ned.get_raster(huc, watershed_workflow.crs.from_fiona(profile['crs']))

#     assert((10743, 11169) == dem.shape)


def test_ned3():
    # single file covers it
    nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('020401010101')

    # get imgs
    ned = watershed_workflow.sources.manager_ned.FileManagerNED('1 arc-second')
    dem_prof, dem = ned.get_raster(huc, watershed_workflow.crs.from_fiona(profile['crs']))
    assert((537, 535) == dem.shape)

def test_ned4():
    # requires tiles
    nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    ned = watershed_workflow.sources.manager_ned.FileManagerNED('1 arc-second')
    dem_prof, dem = ned.get_raster(huc, watershed_workflow.crs.from_fiona(profile['crs']))
    assert((3581, 3723) == dem.shape)

    
