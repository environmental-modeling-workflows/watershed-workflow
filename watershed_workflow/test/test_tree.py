import pytest
import itertools

import shapely.geometry

from watershed_workflow.test.shapes import *
import watershed_workflow.river_tree

def assert_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    assert(len(l1) == len(l2))
    for a,b in zip(l1,l2):
        assert(a == b)

def is_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    if (len(l1) != len(l2)):
        return False
    for a,b in zip(l1,l2):
        if (a != b):
            return False
    return True
        
def test_list_same():
    a = list(shapely.geometry.MultiLineString([[(0,0), (1,0)], [(10,0), (10,1)]]).geoms)
    assert_list_same(a, a)

def test_tree_gen():
    t = watershed_workflow.river_tree.River()
    n0 = t.addChild(0)
    n0.addChild(1)
    n0.addChild(2)
    leaves = list(t.leaves())
    assert(len(leaves) == 2)
    assert(leaves[0] == 1)
    assert(leaves[1] == 2)

 
def test_factory_empty():
    trees = watershed_workflow.river_tree.River.construct_rivers_by_geometry(list())
    assert(len(trees) == 0)
    
def test_factory_y(y):
    trees = watershed_workflow.river_tree.River.construct_rivers_by_geometry(y)
    assert(len(trees) == 1)
    assert_list_same(trees[0].dfs(), y)


def test_factory_y2(y_with_extension):
    trees = watershed_workflow.river_tree.River.construct_rivers_by_geometry(y_with_extension)
    assert(len(trees) == 1)
    assert_list_same(trees[0].dfs(), y_with_extension)


def test_factory_ys(two_ys):
    trees = watershed_workflow.river_tree.River.construct_rivers_by_geometry(two_ys)
    assert(len(trees) == 2)
    assert_list_same(itertools.chain(trees[0].dfs(),trees[1].dfs()), two_ys)


def test_factory_dfs():
    points = [[(0,0), (1,0)],
              [(1,0), (2,0)]]
    ml = list(shapely.geometry.MultiLineString(points).geoms)
    trees = watershed_workflow.river_tree.River.construct_rivers_by_geometry(ml)
    riverlist = list(trees[0].dfs())
    assert(riverlist[0] == ml[1])
    assert(riverlist[1] == ml[0])

    
def test_factory_two_ys_props(two_ys):
    """Creates a river using the mocked HydroSeq data"""
    trees = watershed_workflow.river_tree.River.construct_rivers_by_hydroseq(two_ys)
    assert(len(trees) == 2)
    assert_list_same(itertools.chain(trees[0].dfs(),trees[1].dfs()), two_ys)


def test_factory_braided_geometry(braided_stream):
    """Creates a river using the geometry for a braided system"""
    # randomize the order a bit
    new_order = [3,5,1,0,2,4]
    braided_stream_rand = [braided_stream[i] for i in new_order]
    trees = watershed_workflow.river_tree.River.construct_rivers_by_geometry(braided_stream_rand)

    assert(len(trees) == 1)
    assert(type(trees[0]) is watershed_workflow.river_tree.River)
    assert(len(trees[0]) == 6)

    valid_orderings = [[0,1,2,3,4,5],
                       [0,1,2,4,5,3],
                       [0,4,5,1,2,3],
                       [0,4,5,3,1,2]]
    
    assert(any(is_list_same(trees[0].dfs(), [braided_stream[i] for i in ordering]) for ordering in valid_orderings))


def test_factory_braided_hydroseq(braided_stream):
    """Creates a river using the geometry for a braided system"""
    # randomize the order a bit
    new_order = [3,5,1,0,2,4]
    braided_stream_rand = [braided_stream[i] for i in new_order]
    trees = watershed_workflow.river_tree.River.construct_rivers_by_hydroseq(braided_stream_rand)

    assert(len(trees) == 1)
    assert(type(trees[0]) is watershed_workflow.river_tree.River)
    assert(len(trees[0]) == 6)
    assert_list_same(trees[0].dfs(), braided_stream)
    

    
