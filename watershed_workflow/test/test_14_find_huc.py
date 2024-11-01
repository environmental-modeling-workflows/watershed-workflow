import pytest
import geopandas
import numpy as np
import watershed_workflow.crs
import watershed_workflow

from source_fixtures import datadir, sources


def get_shapes(filename):
    gdf = geopandas.read_file(filename)
    gdf.to_crs(watershed_workflow.crs.latlon_crs)
    return gdf


def test_find_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)

    radius = np.sqrt(float(shp.area[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    with pytest.raises(ValueError):
        watershed_workflow.findHUC(nhd, shp.geometry[0], gdf.crs, '06')


def test_find12(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)
    radius = np.sqrt(float(shp.area[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    assert ('060102020103' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '0601'))


def test_find12_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)
    radius = np.sqrt(float(shp.area[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    assert ('060102020103' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '060102020103'))


def test_find12_raises(datadir, sources):
    """This throws because the shape is not in this huc"""
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)
    radius = np.sqrt(float(shp.area[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '060101080204')


def test_find8(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    shp = get_shapes(testshpfile)
    assert ('06010202' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '0601'))


def test_find8_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    shp = get_shapes(testshpfile)
    assert ('06010202' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '06010202'))


def test_find8_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('copper_creek.shp')
    shp = get_shapes(testshpfile)
    with pytest.raises(RuntimeError):
        watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '0601')


