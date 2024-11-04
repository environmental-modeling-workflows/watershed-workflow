import pytest
import geopandas
import math
import numpy as np
import watershed_workflow.crs
from watershed_workflow.sources.manager_nhd import FileManagerWBD

from source_fixtures import datadir


def get_shapes(filename):
    gdf = geopandas.read_file(filename)
    gdf = gdf.to_crs(watershed_workflow.crs.default_crs)
    return gdf


def test_find12(datadir):
    nhd = FileManagerWBD()

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)
    radius = math.sqrt(float(shp.area.iloc[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    assert ('060102020103' == watershed_workflow.findHUC(nhd, shp.geometry.iloc[0], shp.crs, '0601'))


def test_find12_exact(datadir):
    nhd = FileManagerWBD()

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)
    radius = np.sqrt(float(shp.area[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    assert ('060102020103' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '060102020103'))


def test_find12_raises(datadir):
    """This throws because the shape is not in this huc"""
    nhd = FileManagerWBD()

    testshpfile = datadir.join('test_shapefile.shp')
    shp = get_shapes(testshpfile)
    radius = np.sqrt(float(shp.area[0]) / np.pi)
    shp = shp.buffer(-.001 * radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '060101080204')


def test_find8(datadir):
    nhd = FileManagerWBD()

    testshpfile = datadir.join('test_polygon.shp')
    shp = get_shapes(testshpfile)
    assert ('06010202' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '0601'))


def test_find8_exact(datadir):
    nhd = FileManagerWBD()

    testshpfile = datadir.join('test_polygon.shp')
    shp = get_shapes(testshpfile)
    assert ('06010202' == watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '06010202'))


def test_find8_raises(datadir):
    nhd = FileManagerWBD()

    testshpfile = datadir.join('copper_creek.shp')
    shp = get_shapes(testshpfile)
    with pytest.raises(RuntimeError):
        watershed_workflow.findHUC(nhd, shp.geometry[0], shp.crs, '0601')


