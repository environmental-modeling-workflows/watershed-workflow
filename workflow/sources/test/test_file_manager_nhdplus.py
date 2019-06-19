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


bounds4 = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8 = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

bounds4_crs = np.array(list(workflow.warp.warp_xy(bounds4[0], bounds4[1], workflow.conf.latlon_crs(), workflow.conf.default_crs())) + list(workflow.warp.warp_xy(bounds4[2], bounds4[3], workflow.conf.latlon_crs(), workflow.conf.default_crs())))

bounds8_crs = np.array(list(workflow.warp.warp_xy(bounds8[0], bounds8[1], workflow.conf.latlon_crs(), workflow.conf.default_crs())) + list(workflow.warp.warp_xy(bounds8[2], bounds8[3], workflow.conf.latlon_crs(), workflow.conf.default_crs())))


@pytest.fixture
def datadir(tmpdir, request):
    """Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    folder = os.path.join(*os.path.split(filename)[:-1])
    print('COPYING: ', os.path.join(folder, 'testfiles'))
    dir_util.copy_tree(os.path.join(folder, 'testfiles'), str(tmpdir))
    return tmpdir

def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]

    workflow.warp.warp_shape(shp, profile['crs'], workflow.conf.latlon_crs())
    profile['crs'] = workflow.conf.latlon_crs()
    return profile,workflow.utils.shply(shp['geometry'])


def test_find_raises(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    profile, shp = get_fiona(testshpfile)

    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    with pytest.raises(ValueError):
        sutils.find_huc(shp, profile['crs'], '06', nhd)

def test_find12(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    profile, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == sutils.find_huc(shp, profile['crs'], '0601', nhd))

def test_find12_exact(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    profile, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == sutils.find_huc(shp, profile['crs'], '060102020103', nhd))

def test_find12_raises(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    profile, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        sutils.find_huc(shp, profile['crs'], '060101080204', nhd)

def test_find8(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    assert('06010202' == sutils.find_huc(shp, profile['crs'], '0601', nhd))

def test_find8_exact(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    assert('06010202' == sutils.find_huc(shp, profile['crs'], '06010202', nhd))

def test_find8_raises(datadir):
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    with pytest.raises(RuntimeError):
        sutils.find_huc(shp, profile['crs'], '0204', nhd)
        


def test_nhdplus():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    with pytest.raises(ValueError):
        url = nhd.url('0201') # this huc was removed
    with pytest.raises(ValueError):
        h = nhd.get_huc('02') # not allowed, on 4 only

    url = nhd.url('0204')
    assert('https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlus/HU4/HighResolution/GDB/NHDPLUS_H_0204_HU4_GDB.zip' == url)

    # download
    hfile = nhd.download('0204')
    assert(hfile == nhd.names.file_name('0204'))

def test_nhdplus2():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    # download
    profile, huc = nhd.get_huc('0204')
    bounds = workflow.utils.shply(huc['geometry']).bounds

    assert(np.allclose(bounds4_crs, np.array(bounds), 1))


def test_nhdplus3():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    # download
    profile, huc = nhd.get_huc('02040101')
    assert(workflow.conf.default_crs() == profile['crs'])
    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8_crs, np.array(bounds), 1))


def test_nhdplus4():
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()

    # download
    profile, huc8s = nhd.get_hucs('0204', 8)
    assert(workflow.conf.default_crs() == profile['crs'])
    
def test_nhdplus6():
    # test in a different crs
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101', crs=workflow.conf.latlon_crs())
    assert(workflow.conf.latlon_crs() == profile['crs'])

    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds8, bounds, 1.e-4))

def test_nhdplus7():
    # test in a different crs
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('0204', crs=workflow.conf.latlon_crs())
    assert(workflow.conf.latlon_crs() == profile['crs'])

    bounds = workflow.utils.shply(huc['geometry']).bounds
    assert(np.allclose(bounds4, bounds, 1.e-4))

# hydro tests
def test_nhdplus10():
    # download hydrography
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, rivers = nhd.get_hydro('020401010101')
    assert(workflow.conf.default_crs() == profile['crs'])
    assert(575 == len(rivers))


def test_nhdplus11():
    # download hydrography
    nhd = workflow.sources.manager_nhdplus.FileManagerNHDPlus()
    profile, rivers = nhd.get_hydro('020401010101', crs=workflow.conf.latlon_crs())
    assert(workflow.conf.latlon_crs() == profile['crs'])
    assert(575 == len(rivers))
    
    

    




