"""
'tests' for triangulate

These aren't actually tests -- they just exercise the capability
and make sure it runs and does something.  Can plot for debugging.
Not sure how to test them..."""

import pytest
import shapely
import geopandas
from matplotlib import pyplot as plt

import watershed_workflow.triangulation
import watershed_workflow.hydrography
import watershed_workflow.split_hucs
import watershed_workflow.plot


@pytest.fixture
def hucs_rivers():
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    b2 = [(10, -5), (20, -5), (20, 5), (10, 5)]
    b3 = [(0, 5), (10, 5), (20, 5), (20, 10), (0, 10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))
    df = geopandas.GeoDataFrame(geometry=tb)
    hucs = watershed_workflow.split_hucs.SplitHUCs(df)

    rs = geopandas.GeoDataFrame(geometry=[
        shapely.geometry.LineString([(5., 0.), (10., 5), ]),
        shapely.geometry.LineString([(15., 0.), (10., 5), ]),
        shapely.geometry.LineString([(10., 5.), (10, 10)]),
    ])
    rivers = watershed_workflow.river_tree.createRivers(rs)
    watershed_workflow.simplify(hucs, rivers, 1)
    return hucs, rivers


def test_triangulate_nofunc(hucs_rivers):
    hucs, rivers = hucs_rivers
    points, tris = watershed_workflow.triangulation.triangulate(hucs, tol=0.01)

    # fig, ax = plt.subplots(1,1)
    # ax.triplot(points[:,0], points[:,1], tris)
    # hucs.plot(color='r', ax=ax)
    # for river in rivers:
    #     river.plot(color='b', ax=ax)
    # plt.show()
    # assert(False)
    

def test_triangulate_max_area(hucs_rivers):
    hucs, rivers = hucs_rivers
    func = watershed_workflow.triangulation.refineByMaxArea(1.)
    points, tris = watershed_workflow.triangulation.triangulate(hucs, refinement_func=func, tol=0.01)

    # fig, ax = plt.subplots(1,1)
    # ax.triplot(points[:,0], points[:,1], tris)
    # hucs.plot(color='r', ax=ax)
    # for river in rivers:
    #     river.plot(color='b', ax=ax)
    # plt.show()
    # assert(False)


def test_triangulate_distance(hucs_rivers):
    hucs, rivers = hucs_rivers
    func = watershed_workflow.triangulation.refineByRiverDistance(1., 0.5, 4, 2, rivers)
    points, tris = watershed_workflow.triangulation.triangulate(hucs, refinement_func=func, tol=0.01)

    # fig, ax = plt.subplots(1,1)
    # ax.triplot(points[:,0], points[:,1], tris)
    # hucs.plot(color='r', ax=ax)
    # for river in rivers:
    #     river.plot(color='b', ax=ax)
    # plt.show()
    # assert(False)


def test_triangulate_internal_boundary(hucs_rivers):
    hucs, rivers = hucs_rivers
    func = watershed_workflow.triangulation.refineByRiverDistance(1., 0.5, 4, 2, rivers)

    points, tris = watershed_workflow.triangulation.triangulate(hucs,
                                                                internal_boundaries=[r.linestring for river in rivers for r in river],
                                                                refinement_func=func, tol=0.01)

    fig, ax = plt.subplots(1,1)
    ax.triplot(points[:,0], points[:,1], tris)
    hucs.plot(color='r', ax=ax)
    for river in rivers:
        river.plot(color='b', ax=ax)
    plt.show()
    assert(False)
    
