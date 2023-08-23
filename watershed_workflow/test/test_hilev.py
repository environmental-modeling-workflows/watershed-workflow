import os
from distutils import dir_util
import pytest
import fiona
import shapely.geometry
import numpy as np
import watershed_workflow.crs
import watershed_workflow

from source_fixtures import datadir, sources, sources_download


def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]

    crs = watershed_workflow.crs.latlon_crs()
    watershed_workflow.warp.shape(shp, watershed_workflow.crs.from_fiona(profile['crs']), crs)
    shply = watershed_workflow.utils.create_shply(shp)
    assert (type(shply) == shapely.geometry.Polygon)
    return crs, shply


def test_find_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)

    radius = np.sqrt(shp.area / np.pi)
    shp = shp.buffer(-.001 * radius)
    with pytest.raises(ValueError):
        watershed_workflow.find_huc(nhd, shp, crs, '06')


def test_find12(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area / np.pi)
    shp = shp.buffer(-.001 * radius)
    print(shp.area)
    assert ('060102020103' == watershed_workflow.find_huc(nhd, shp, crs, '0601'))


def test_find12_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area / np.pi)
    shp = shp.buffer(-.001 * radius)
    print(shp.area)
    assert ('060102020103' == watershed_workflow.find_huc(nhd, shp, crs, '060102020103'))


def test_find12_raises(datadir, sources):
    """This throws because the shape is not in this huc"""
    nhd = sources['HUC']

    testshpfile = datadir.join('test_shapefile.shp')
    crs, shp = get_fiona(testshpfile)
    radius = np.sqrt(shp.area / np.pi)
    shp = shp.buffer(-.001 * radius)
    print(shp.area)
    with pytest.raises(RuntimeError):
        watershed_workflow.find_huc(nhd, shp, crs, '060101080204')


def test_find8(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert ('06010202' == watershed_workflow.find_huc(nhd, shp, crs, '0601'))


def test_find8_exact(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('test_polygon.shp')
    crs, shp = get_fiona(testshpfile)
    assert ('06010202' == watershed_workflow.find_huc(nhd, shp, crs, '06010202'))


def test_find8_raises(datadir, sources):
    nhd = sources['HUC']

    testshpfile = datadir.join('copper_creek.shp')
    crs, shp = get_fiona(testshpfile)
    with pytest.raises(RuntimeError):
        watershed_workflow.find_huc(nhd, shp, crs, '0601')


def test_river_tree_properties(sources_download):
    crs = watershed_workflow.crs.default_crs()
    nhd = sources_download['hydrography']
    _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
    _, reaches = watershed_workflow.get_reaches(nhd,
                                                '060102020103',
                                                cc.exterior(),
                                                crs,
                                                crs,
                                                properties=True)

    rivers = watershed_workflow.construct_rivers(reaches, method='hydroseq')
    assert (len(rivers) == 1)
    assert (rivers[0].is_consistent())
    assert (len(rivers[0]) == 97)


def test_river_tree_properties_prune(sources_download):
    crs = watershed_workflow.crs.default_crs()
    nhd = sources_download['hydrography']
    _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
    _, reaches = watershed_workflow.get_reaches(nhd,
                                                '060102020103',
                                                cc.exterior(),
                                                crs,
                                                crs,
                                                properties=True)

    rivers = watershed_workflow.construct_rivers(reaches,
                                                 method='hydroseq',
                                                 prune_by_area=0.03*cc.exterior().area*1.e-6)
    assert (len(rivers) == 1)
    assert (rivers[0].is_consistent())
    assert (len(rivers[0]) == 50)


def test_river_tree_geometry(sources):
    crs = watershed_workflow.crs.default_crs()
    nhd = sources['HUC']
    _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
    _, reaches = watershed_workflow.get_reaches(nhd,
                                                '060102020103',
                                                cc.exterior(),
                                                crs,
                                                crs,
                                                properties=False)

    rivers = watershed_workflow.construct_rivers(reaches)
    assert (len(rivers) == 1)
    assert (rivers[0].is_consistent())
    assert (len(rivers[0]) == 98)
