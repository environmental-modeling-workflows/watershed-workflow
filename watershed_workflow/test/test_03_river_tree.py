import pytest
import itertools

import shapely.geometry

from watershed_workflow.test.shapes import *
import watershed_workflow.river_tree

import geopandas

def assert_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    assert (len(l1) == len(l2))
    for a, b in zip(l1, l2):
        assert (a == b)


def is_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    if (len(l1) != len(l2)):
        return False
    for a, b in zip(l1, l2):
        if (a != b):
            return False
    return True


def test_list_same():
    a = list(shapely.geometry.MultiLineString([[(0, 0), (1, 0)], [(10, 0), (10, 1)]]).geoms)
    assert_list_same(a, a)


def test_tree_gen():
    df = geopandas.GeoDataFrame({'index':[0,1,2,3],
                                 'geometry': [shapely.geometry.LineString([(0,0), (0,1)]),
                                              shapely.geometry.LineString([(0,1), (1,0)]),
                                              shapely.geometry.LineString([(1,0), (1,1)]),
                                              shapely.geometry.LineString([(1,0), (2,0)])]
                                 }).set_index('index')
    t = watershed_workflow.river_tree.River(0,df)
    n0 = t.addChild(1)
    n0.addChild(2)
    n0.addChild(3)
    leaves = list(t.leaf_nodes)
    assert (len(leaves) == 2)
    assert (leaves[0].index == 2)
    assert (leaves[1].index == 3)


def test_factory_empty():
    trees = watershed_workflow.river_tree.River.constructRiversByGeometry(list())
    assert (len(trees) == 0)


def test_factory_y(y):
    trees = watershed_workflow.river_tree.River.constructRiversByGeometry(y)
    assert (len(trees) == 1)
    assert_list_same([n.linestring for n in trees[0].preOrder()], y.geometry)


def test_factory_y2(y_with_extension):
    trees = watershed_workflow.river_tree.River.constructRiversByGeometry(y_with_extension)
    assert (len(trees) == 1)
    assert_list_same([n.linestring for n in trees[0].preOrder()], y_with_extension.geometry)


def test_factory_ys(two_ys):
    trees = watershed_workflow.river_tree.River.constructRiversByGeometry(two_ys)
    assert (len(trees) == 2)
    assert_list_same(itertools.chain([n.linestring for n in trees[0].preOrder()],
                                     [n.linestring for n in trees[1].preOrder()]), two_ys.geometry)


def test_factory_dfs():
    points = [[(0, 0), (1, 0)], [(1, 0), (2, 0)]]
    ml = list(shapely.geometry.MultiLineString(points).geoms)
    df = geopandas.GeoDataFrame({'index' : range(len(ml)),
                                 'geometry' : ml}).set_index('index')

    trees = watershed_workflow.river_tree.River.constructRiversByGeometry(df)
    riverlist = [n.linestring for n in trees[0].preOrder()]
    assert (riverlist[0] == ml[1])
    assert (riverlist[1] == ml[0])


def test_factory_two_ys_props(two_ys):
    """Creates a river using the mocked HydroSeq data"""
    trees = watershed_workflow.river_tree.River.constructRiversByHydroseq(two_ys)
    assert (len(trees) == 2)
    assert_list_same(itertools.chain([n.linestring for n in trees[0].preOrder()],
                                     [n.linestring for n in trees[1].preOrder()]), two_ys.geometry)


def test_factory_braided_geometry(braided_stream):
    """Creates a river using the geometry for a braided system"""
    # randomize the order a bit
    new_order = [3, 5, 1, 0, 2, 4]
    braided_stream['index'] = new_order
    braided_stream_rand = braided_stream.sort_index()
    
    trees = watershed_workflow.river_tree.River.constructRiversByGeometry(braided_stream_rand)

    assert (len(trees) == 1)
    assert (type(trees[0]) is watershed_workflow.river_tree.River)
    assert (len(trees[0]) == 6)

    valid_orderings = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5, 3], [0, 4, 5, 1, 2, 3],
                       [0, 4, 5, 3, 1, 2]]

    assert (any(
        is_list_same([n.linestring for n in trees[0].preOrder()],
                     [braided_stream.geometry[i] for i in ordering])
        for ordering in valid_orderings))


def test_factory_braided_hydroseq(braided_stream):
    """Creates a river using the geometry for a braided system"""
    # randomize the order a bit
    new_order = [3, 5, 1, 0, 2, 4]
    braided_stream['index'] = new_order
    braided_stream_rand = braided_stream.sort_index()

    trees = watershed_workflow.river_tree.River.constructRiversByHydroseq(braided_stream_rand)

    assert (len(trees) == 1)
    assert (type(trees[0]) is watershed_workflow.river_tree.River)
    assert (len(trees[0]) == 6)
    assert_list_same([n.linestring for n in trees[0].preOrder()],
                      braided_stream.geometry)


def test_merge():
    s2 = shapely.geometry.LineString([(2, 0), (1, 0)])
    s1 = shapely.geometry.LineString([(1, 0), (0, 0)])
    df = geopandas.GeoDataFrame({'index' : [2,1],
                                 'geometry' : [s2,s1]}).set_index('index')

    n2 = watershed_workflow.river_tree.River(2, df)
    n1 = watershed_workflow.river_tree.River(1, df, [n2,])
    assert n1.isContinuous()
    assert n1.linestring.length == 1
    assert n2.parent is n1
    assert n2 in n1.preOrder()

    n2.merge()
    assert n2 not in n1.preOrder()
    del n2
    assert (n1.linestring.length == 2)
    assert (len(n1.children) == 0)


def test_prune():
    s2 = shapely.geometry.LineString([(2, 0), (1, 0)])
    s1 = shapely.geometry.LineString([(1, 0), (0, 0)])
    df = geopandas.GeoDataFrame({'index' : [2,1],
                                 'geometry' : [s2,s1]}).set_index('index')

    n2 = watershed_workflow.river_tree.River(2, df)
    n1 = watershed_workflow.river_tree.River(1, df, [n2,])
    assert n1.isContinuous()
    assert n1.linestring.length == 1
    assert n2.parent is n1
    assert n2 in n1.preOrder()

    n2.prune()
    assert n2 not in n1.preOrder()
    del n2
    assert (n1.linestring.length == 1)
    assert (len(n1.children) == 0)
    
