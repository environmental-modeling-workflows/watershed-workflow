import pytest
import itertools

import shapely.geometry

from watershed_workflow.test.shapes import *
import watershed_workflow.river_tree

def assert_list_same(l1, l2):
    for a,b in zip(l1,l2):
        assert(a == b)

def test_list_same():
    a = shapely.geometry.MultiLineString([[(0,0), (1,0)], [(10,0), (10,1)]])
    assert_list_same(a, a)

def test_tree_gen():
    t = watershed_workflow.river_tree.RiverTree()
    n0 = t.addChild(0)
    n0.addChild(1)
    n0.addChild(2)
    leaves = list(t.leaves())
    assert(len(leaves) == 2)
    assert(leaves[0] == 1)
    assert(leaves[1] == 2)


def test_y(y):
    trees = watershed_workflow.river_tree.make_trees(y)
    assert(len(trees) == 1)
    assert_list_same(list(trees[0].dfs()), y)


def test_y2(y_with_extension):
    trees = watershed_workflow.river_tree.make_trees(y_with_extension)
    assert(len(trees) == 1)
    dfs = list(trees[0].dfs())
    assert_list_same(dfs, y_with_extension)


def test_ys(two_ys):
    trees = watershed_workflow.river_tree.make_trees(two_ys)
    assert(len(trees) == 2)
    assert_list_same(itertools.chain(trees[0].dfs(),trees[1].dfs()), two_ys)

def test_ywj(y_with_junction):
    trees = watershed_workflow.river_tree.make_trees(y_with_junction)
    assert(len(trees) is 1)
    tree = trees[0]
    segs = list(tree.dfs())
    assert(len(segs) is 4)

    newseg0 = shapely.geometry.LineString([(0.5,0), (0,0)])
    newseg1 = shapely.geometry.LineString([(1,0), (0.5,0)])
    assert(segs[0] == newseg0)
    assert((segs[1] == newseg1 and segs[2] == y_with_junction[2]) or
           (segs[2] == newseg1 and segs[1] == y_with_junction[2]))
    assert(segs[3] == y_with_junction[1])
    
def test_dfs():
    points = [[(0,0), (1,0)],
              [(1,0), (2,0)]]
    ml = shapely.geometry.MultiLineString(points)
    trees = watershed_workflow.river_tree.make_trees(ml)
    riverlist = list(trees[0].dfs())
    assert(riverlist[0] == ml[1])
    assert(riverlist[1] == ml[0])

def test_to_list(y_with_junction):
    forest = watershed_workflow.river_tree.make_trees(y_with_junction)
    treelist = watershed_workflow.river_tree.tree_to_list(forest[0])
    forestlist = watershed_workflow.river_tree.forest_to_list(forest)
    
    
    
    

    
