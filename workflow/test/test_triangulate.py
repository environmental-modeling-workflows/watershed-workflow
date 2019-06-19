"""
'tests' for triangulate

These aren't actually tests -- they just exercise the capability
and make sure it runs and does something.  Can plot for debugging.
Not sure how to test them..."""

import pytest
import shapely
from matplotlib import pyplot as plt

import workflow.triangulate
import workflow.hydrography
import workflow.split_hucs
import workflow.plot

@pytest.fixture
def hucs_rivers():
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5)]
    b3 = [(0, 5), (10,5), (20,5), (20,10), (0,10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))
    hucs = workflow.split_hucs.SplitHUCs(tb)

    rs = [shapely.geometry.LineString([(5.,0.), (10.,5),]),
          shapely.geometry.LineString([(15.,0.), (10.,5),]),
          shapely.geometry.LineString([(10.,5.), (10,10)]),
    ]
    rivers = workflow.hydrography.make_global_tree(rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    return hucs, rivers

def test_triangulate_nofunc(hucs_rivers):
    hucs,rivers = hucs_rivers

    points, tris = workflow.triangulate.triangulate(hucs, rivers)
    # workflow.plot.triangulation(points,tris)
    # workflow.plot.hucs(hucs,'r')
    # workflow.plot.rivers(rivers,'b')
    # plt.show()

def test_triangulate_max_area(hucs_rivers):
    hucs,rivers = hucs_rivers
    func = workflow.triangulate.refine_from_max_area(1.)
    points, tris = workflow.triangulate.triangulate(hucs, rivers, refinement_func=func)
    # workflow.plot.triangulation(points,tris)
    # workflow.plot.hucs(hucs,'r')
    # workflow.plot.rivers(rivers,'b')
    # plt.show()

def test_triangulate_distnace(hucs_rivers):    
    hucs,rivers = hucs_rivers
    func = workflow.triangulate.refine_from_river_distance(1., 0.5, 4, 2, rivers)
    points, tris = workflow.triangulate.triangulate(hucs, rivers, refinement_func=func)
    # workflow.plot.triangulation(points,tris)
    # workflow.plot.hucs(hucs,'r')
    # workflow.plot.rivers(rivers,'b')
    # plt.show()
