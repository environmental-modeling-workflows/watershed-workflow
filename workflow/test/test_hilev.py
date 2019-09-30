import os
from distutils import dir_util
import pytest
import fiona
import shapely.geometry
import numpy as np
import workflow.crs
import workflow.hilev
import workflow.source_list

@pytest.fixture
def datadir(tmpdir, request):
    """Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]

    workflow.warp.warp_shape(shp, workflow.crs.from_fiona(profile['crs']), workflow.crs.latlon_crs())
    shply = workflow.utils.shply(shp['geometry'])
    assert(type(shply) == shapely.geometry.Polygon)
    return workflow.crs.latlon_crs(), shply


@pytest.fixture
def sources():
    sources = dict()
    #sources['HUC08'] = workflow.files.NHDFileManager()
    sources['HUC'] = workflow.source_list.FileManagerNHDPlus()
    sources['DEM'] = workflow.source_list.FileManagerNED()
    return sources


def test_find_raises(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)

    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    with pytest.raises(ValueError):
        workflow.hilev.find_huc(nhd, shp, crs, '06')

def test_find12(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == workflow.hilev.find_huc(nhd, shp, crs, '0601'))

def test_find12_exact(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == workflow.hilev.find_huc(nhd, shp, crs, '060102020103'))

def test_find12_raises(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        workflow.hilev.find_huc(nhd, shp, crs, '060101080204')

def test_find8(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert('06010202' == workflow.hilev.find_huc(nhd, shp, crs, '0601'))

def test_find8_exact(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert('06010202' == workflow.hilev.find_huc(nhd, shp, crs, '06010202'))

def test_find8_raises(datadir):
    nhd = workflow.source_list.FileManagerNHDPlus()

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    with pytest.raises(RuntimeError):
        workflow.hilev.find_huc(nhd, shp, crs, '0204')


