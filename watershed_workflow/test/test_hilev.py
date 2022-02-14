import os
from distutils import dir_util
import pytest
import fiona
import shapely.geometry
import numpy as np
import watershed_workflow.crs
import watershed_workflow.hilev

from source_fixtures import datadir, sources

def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]

    crs = watershed_workflow.crs.latlon_crs()
    watershed_workflow.warp.shape(shp, watershed_workflow.crs.from_fiona(profile['crs']), crs)
    shply = watershed_workflow.utils.shply(shp)
    assert(type(shply) == shapely.geometry.Polygon)
    return crs, shply


def test_find_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)

    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    with pytest.raises(ValueError):
        watershed_workflow.hilev.find_huc(nhd, shp, crs, '06')

def test_find12(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '0601'))

def test_find12_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '060102020103'))

def test_find12_raises(datadir, sources):
    """This throws because the shape is not in this huc"""
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        watershed_workflow.hilev.find_huc(nhd, shp, crs, '060101080204')

def test_find8(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert('06010202' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '0601'))

def test_find8_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert('06010202' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '06010202'))

def test_find8_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('copper_creek.shp')
    crs, shp = get_fiona(testshpfile)
    with pytest.raises(RuntimeError):
        watershed_workflow.hilev.find_huc(nhd, shp, crs, '0601')

def test_river_tree_properties(sources):
    crs = watershed_workflow.crs.default_crs()
    nhd = sources['HUC']
    _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
    _, reaches = watershed_workflow.get_reaches(nhd, '060102020103',None,crs, merge=False)

    rivers1 = watershed_workflow.simplify_and_prune(cc, reaches, filter=True, simplify=50, cut_intersections=False, ignore_small_rivers=2)
    rivers2 = watershed_workflow.simplify_and_prune(cc, reaches, filter=True, simplify=50, cut_intersections=False, ignore_small_rivers=2,
                                          prune_by_area_fraction=0.03)
import os
from distutils import dir_util
import pytest
import fiona
import shapely.geometry
import numpy as np
import watershed_workflow.crs
import watershed_workflow.hilev

from source_fixtures import datadir, sources

def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]

    crs = watershed_workflow.crs.latlon_crs()
    watershed_workflow.warp.shape(shp, watershed_workflow.crs.from_fiona(profile['crs']), crs)
    shply = watershed_workflow.utils.shply(shp)
    assert(type(shply) == shapely.geometry.Polygon)
    return crs, shply


def test_find_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)

    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    with pytest.raises(ValueError):
        watershed_workflow.hilev.find_huc(nhd, shp, crs, '06')

def test_find12(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '0601'))

def test_find12_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    assert('060102020103' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '060102020103'))

def test_find12_raises(datadir, sources):
    """This throws because the shape is not in this huc"""
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area/np.pi)
    shp = shp.buffer(-.001*radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        watershed_workflow.hilev.find_huc(nhd, shp, crs, '060101080204')

def test_find8(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert('06010202' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '0601'))

def test_find8_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert('06010202' == watershed_workflow.hilev.find_huc(nhd, shp, crs, '06010202'))

def test_find8_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('copper_creek.shp')
    crs, shp = get_fiona(testshpfile)
    with pytest.raises(RuntimeError):
        watershed_workflow.hilev.find_huc(nhd, shp, crs, '0601')

def test_river_tree_properties(sources):
    crs = watershed_workflow.crs.default_crs()
    nhd = sources['HUC']
    _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
    _, reaches = watershed_workflow.get_reaches(nhd, '060102020103',None,crs, merge=False)

    rivers1 = watershed_workflow.simplify_and_prune(cc, reaches, filter=True, simplify=50, cut_intersections=False, ignore_small_rivers=2)
    rivers2 = watershed_workflow.simplify_and_prune(cc, reaches, filter=True, simplify=50, cut_intersections=False, ignore_small_rivers=2,
                                          prune_by_area_fraction=0.03)