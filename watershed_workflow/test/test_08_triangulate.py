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
from watershed_workflow.test.shapes import *

_plot = False
_assert_plot = False
def plot(hucs, rivers, points, tris, force = False):
    if _plot or force:
        fig, ax = plt.subplots(1,1)
        ax.triplot(points[:,0], points[:,1], tris)
        hucs.plot(color='r', ax=ax)
        for river in rivers:
            river.plot(color='b', ax=ax)
        plt.show()
        assert not _assert_plot


def test_triangulate_nofunc(watershed_rivers1):
    hucs, rivers = watershed_rivers1
    watershed_workflow.simplify(hucs, rivers, 1)
    
    points, tris = watershed_workflow.triangulation.triangulate(hucs, tol=0.01)
    plot(hucs, rivers, points, tris)
    

def test_triangulate_max_area(watershed_rivers1):
    hucs, rivers = watershed_rivers1
    watershed_workflow.simplify(hucs, rivers, 1)

    func = watershed_workflow.triangulation.refineByMaxArea(1.)
    points, tris = watershed_workflow.triangulation.triangulate(hucs, refinement_func=func, tol=0.01)
    plot(hucs, rivers, points, tris)


def test_triangulate_distance(watershed_rivers1):
    hucs, rivers = watershed_rivers1
    watershed_workflow.simplify(hucs, rivers, 1)

    func = watershed_workflow.triangulation.refineByRiverDistance(1., 0.5, 4, 2, rivers)
    points, tris = watershed_workflow.triangulation.triangulate(hucs, refinement_func=func, tol=0.01)
    plot(hucs, rivers, points, tris)


def test_triangulate_internal_boundary(watershed_rivers1):
    hucs, rivers = watershed_rivers1
    watershed_workflow.simplify(hucs, rivers, 1)

    func = watershed_workflow.triangulation.refineByRiverDistance(1., 0.5, 4, 2, rivers)
    points, tris = watershed_workflow.triangulation.triangulate(hucs,
               internal_boundaries=[r.linestring for river in rivers for r in river],
               refinement_func=func, tol=0.01)
    plot(hucs, rivers, points, tris)
    
