import pytest
import numpy as np
import math
import shapely
import geopandas
from matplotlib import pyplot as plt

from watershed_workflow.river_mesh import *
from watershed_workflow.test.shapes import *

plot = False

#
# Tests for geometry helpers
#
def test_line_shifts():
    p1 = (0.,0)
    p2 = (2.,1)
    l1 = computeLine(p1, p2)
    l1s = translateLinePerpendicular(l1, 0.6)

    p2s = projectOne(p1, p2, 0.6)
    p1s = projectOne(p2, p1, -0.6)
    l2 = computeLine(p1s, p2s)

    assert np.allclose(l2, l1s)



def test_intersections():
    p1 = (0,0)
    p2 = (1,0)
    p3 = (1,1)

    i1 = findIntersection(computeLine(p1,p2), computeLine(p2,p3))
    assert np.allclose(i1, (1,0))

    p4 = (2,3)
    p5 = (1,4)
    i2 = findIntersection(computeLine(p1,p3), computeLine(p4,p5))
    assert np.allclose(i2, (2.5,2.5))


def test_projectOne():
    p = projectOne((0.,1), (0,0), 1)
    assert np.allclose(p, (-1,0))

    p = projectOne((0.,1), (1,1), 1)
    assert np.allclose(p, (1,0))

    p = projectOne((3.,4), (4,3), np.sqrt(2))
    assert np.allclose(p, (3,2))

def test_projectTwo_parallel():
    p = projectTwo((0.,0), (1.,0), (2.,0), 1, 1)
    assert np.allclose(p, (1.,-1))

def test_projectTwo_parallel_mixedwidth():
    p = projectTwo((0.,0), (1.,0), (2.,0), 0.5, 1.5)
    assert np.allclose(p, (1.,-1))


def test_projectTwo():
    p = projectTwo((0.,0), (1.,0), (1.,1), 1., 1.)
    assert np.allclose(p, (2., -1))

def test_projectTwo_mixed():
    p = projectTwo((0.,0), (1.,0), (1.,1), 2., 1.)
    assert np.allclose(p, (2., -2))
    
def test_projectTwo_jagged():
    p = projectTwo((0,0), (2,1), (4,0), 1, 1)
    new_p = (2, 1 - 1./np.cos(np.arctan(0.5)))
    assert np.allclose(p, new_p)
    


#
# Tests for toy reaches
#
_plot = True
_assert_plot = False
def plot(river, coords, elems, hucs = None, force = False):
    if _plot or force:
        fig, ax = plt.subplots(1,1)
        river.plot(ax=ax, color='b', marker='x')
        ax.scatter(coords[:,0], coords[:,1], marker='o', color='g')
        to_df(coords, elems).plot(ax=ax, color='r', )

        if hucs is not None:
            df_huc_ls = geopandas.GeoDataFrame(geometry=list(hucs.linestrings.values()))
            df_huc_ls.plot(color='k', ax=ax)
        plt.show()
        assert not _assert_plot

        
def to_df(coords, elems):
    return geopandas.GeoDataFrame(geometry=[shapely.geometry.Polygon(coords[e]).exterior for e in elems])

def test_single_reach_vert():
    r1 = shapely.geometry.LineString([(0,3), (0,2), (0,1), (0,0)])
    reaches = geopandas.GeoDataFrame(geometry=[r1])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')

    def computeWidth(x): return 0.5

    coords, elems = watershed_workflow.river_mesh.createRiverMesh(rivers[0], computeWidth)
    assert len(elems) == 3
    assert len(coords) == 7
    assert len(elems[0]) == 3
    assert len(elems[1]) == 4
    assert len(elems[2]) == 4

    print(coords)
    gold_coords = np.array(
        [[-0.25,  0.  ],
         [-0.25,  1.  ],
         [-0.25,  2.  ],
         [ 0.,    3.  ],
         [ 0.25,  2.  ],
         [ 0.25,  1.  ],
         [ 0.25,  0.  ]])
    assert np.allclose(gold_coords, coords, 1.e-4)
    plot(rivers[0], coords, elems)


def test_single_reach_horiz():
    r1 = shapely.geometry.LineString([(0,0), (2,0), (3,0), (6,0)])
    reaches = geopandas.GeoDataFrame(geometry=[r1])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')

    def computeWidth(x): return 0.5

    coords, elems = watershed_workflow.river_mesh.createRiverMesh(rivers[0], computeWidth)
    assert len(elems) == 3
    assert len(coords) == 7
    assert len(elems[0]) == 3
    assert len(elems[1]) == 4
    assert len(elems[2]) == 4

    print(coords)

    gold_coords = np.array(
        [[ 6.,   -0.25],
         [ 3.,   -0.25],
         [ 2.,   -0.25],
         [ 0.,    0.  ],
         [ 2.,    0.25],
         [ 3.,    0.25],
         [ 6.,    0.25]])
    assert np.allclose(gold_coords, coords, 1.e-4)
    plot(rivers[0], coords, elems)

    
def test_single_reach_diag():
    r1 = shapely.geometry.LineString([(0,0), (2,2), (3,3), (6,6)])
    reaches = geopandas.GeoDataFrame(geometry=[r1])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')

    def computeWidth(x): return 0.5

    coords, elems = watershed_workflow.river_mesh.createRiverMesh(rivers[0], computeWidth)
    assert len(elems) == 3
    assert len(coords) == 7
    assert len(elems[0]) == 3
    assert len(elems[1]) == 4
    assert len(elems[2]) == 4

    print(coords)

    gold_coords = np.array(
        [[6.1767767, 5.8232233],
         [3.1767767, 2.8232233],
         [2.1767767, 1.8232233],
         [0.,        0.       ],
         [1.8232233, 2.1767767],
         [2.8232233, 3.1767767],
         [5.8232233, 6.1767767]]
    )
    assert np.allclose(gold_coords, coords, 1.e-4)
    plot(rivers[0], coords, elems)
    

def test_single_reach_jagged():
    r1 = shapely.geometry.LineString([(0,0), (2,1), (3,3), (5,5)])
    reaches = geopandas.GeoDataFrame(geometry=[r1])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')

    def computeWidth(x): return np.sqrt(2.0)/2

    coords, elems = watershed_workflow.river_mesh.createRiverMesh(rivers[0], computeWidth)
    assert len(elems) == 3
    assert len(coords) == 7
    assert len(elems[0]) == 3
    assert len(elems[1]) == 4
    assert len(elems[2]) == 4

    print(coords)

    gold_coords = np.array(
        [[5.25,       4.75      ],
         [3.29056942, 2.79056942],
         [2.26352314, 0.73647686],
         [0.,         0.        ],
         [1.73647686, 1.26352314],
         [2.70943058, 3.20943058],
         [4.75,       5.25      ]]
    )
    assert np.allclose(gold_coords, coords, 1.e-4)
    plot(rivers[0], coords, elems)


def test_two_coplanar_reaches():
    r1 = shapely.geometry.LineString([(0,0), (2,0), (3,0), (5,0)])
    r2 = shapely.geometry.LineString([(-10, 0), (-5,0), (-4,0), (-1,0), (0,0)])
    reaches = geopandas.GeoDataFrame(geometry=[r1, r2])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')

    def computeWidth(x): return 1

    coords, elems = watershed_workflow.river_mesh.createRiverMesh(rivers[0], computeWidth)
    assert len(elems) == 7
    assert len(coords) == 15
    assert len(elems[0]) == 3
    assert all(len(e) == 4 for e in elems[1:])

    print(coords)

    gold_coords = np.array(
        [[5,-0.5],
         [3,-0.5],
         [2,-0.5],
         [0,-0.5],
         [-1,-0.5],
         [-4,-0.5],
         [-5,-0.5],
         [-10,0],
         [-5,0.5],
         [-4,0.5],
         [-1,0.5],
         [0,0.5],
         [2,0.5],
         [3,0.5],
         [5,0.5]], 'd')

    print(gold_coords)
    assert np.allclose(gold_coords, coords, 1.e-4)
    plot(rivers[0], coords, elems)


def test_two_kinked_reaches():
    r1 = shapely.geometry.LineString([(0,0), (2,1), (4,2), (6,3)])
    r2 = shapely.geometry.LineString([(-6,3), (-4, 2), (-2, 1), (0,0)])
    reaches = geopandas.GeoDataFrame(geometry=[r1, r2])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')

    def computeWidth(x): return 1

    coords, elems = watershed_workflow.river_mesh.createRiverMesh(rivers[0], computeWidth)
    assert len(elems) == 6
    assert len(coords) == 13
    assert len(elems[0]) == 3
    assert all(len(e) == 4 for e in elems[1:])

    print(coords)
    gold_coords = np.array(
        [[ 6.2236068,   2.5527864 ],
         [ 4.2236068,   1.5527864 ],
         [ 2.2236068,   0.5527864 ],
         [ 0.,         -0.55901699],
         [-2.2236068,   0.5527864 ],
         [-4.2236068,   1.5527864 ],
         [-6.,          3.        ],
         [-3.7763932,   2.4472136 ],
         [-1.7763932,   1.4472136 ],
         [-0.,          0.55901699],
         [ 1.7763932,   1.4472136 ],
         [ 3.7763932,   2.4472136 ],
         [ 5.7763932,   3.4472136 ]])
    assert np.allclose(gold_coords, coords, 1.e-4)
    plot(rivers[0], coords, elems)


def assert_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    assert (len(l1) == len(l2))
    for a, b in zip(l1, l2):
        assert (a == b)

        
@pytest.fixture
def not_as_simple_y():
    reach1 = shapely.geometry.LineString([(4, 10), (4.01, 5.0), (4, 0)])
    reach2 = shapely.geometry.LineString([(1, 19), (2, 15.01), (4, 10)])
    reach3 = shapely.geometry.LineString([(8, 19), (6, 15.01), (4, 10)])
    reaches = geopandas.GeoDataFrame(geometry=[reach1, reach2, reach3])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')
    return rivers[0]


def test_y(not_as_simple_y):
    """A three-reach system with a junction."""
    def computeWidth(a): return 1
    coords, elems = createRiverMesh(not_as_simple_y, computeWidth)

    assert (13 == len(coords))
    assert np.allclose((1.,19), coords[4])
    assert np.allclose((4.,11.348612713124119), coords[6])

    assert (6 == len(elems))
    assert (3 == len(elems[0])) # headwater elem tri on left branch
    assert (4 == len(elems[1])) # midstream elem quad on left branch
    assert (3 == len(elems[2])) # headwater elem tri on right
    assert (4 == len(elems[3])) # midstream elem quad on right
    assert (5 == len(elems[4])) # junction
    assert (4 == len(elems[5])) # downstream outlet

    assert_list_same([1, 2, 6, 10, 11], elems[4])
    assert_list_same([7, 8, 9], elems[2])
    assert_list_same([0, 1, 11, 12], elems[-1])

    plot(not_as_simple_y, coords, elems)


# def test_ADD_TEST_FOR_VARIABLE_WIDTH():
#     assert False
    

    

#
# Adapt HUCs to river corridor
# 
def test_huc_to_corridor(watershed_rivers1):
    hucs, rivers = watershed_rivers1
    watershed_workflow.simplify(hucs, rivers, 1)

    coords, elems, hole_points = createRiverMesh(rivers[0], lambda a : 1)
    adjustHUCsToRiverMesh(hucs, rivers[0], coords)

    plot(rivers[0], coords, elems, hucs)
    
