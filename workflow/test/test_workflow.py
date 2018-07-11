import pytest

import numpy as np
import shapely.geometry
import workflow.utils
import workflow.hucs
import workflow.hydrography

from workflow.test.shapes import *

def data(poly_hucs,river_segs):
    hucs = workflow.hucs.HUCs(poly_hucs)
    rivers = workflow.hydrography.make_global_tree(river_segs)
    return hucs,rivers


def test_snap1():
    # snap a river endpoint onto a huc
    b1 = [(0, -5), (10,-5), (10, 0.001), (10,5), (0,5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5.,0.), (9.99,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 5)
    assert(any(workflow.utils.close((10.,0.), c) for c in poly0.boundary.coords[:-1]))
    assert(len(rivers) is 1)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,0.), c) for c in r.coords))

def test_snap2():
    # generic intersection with no movement only additions
    tb = two_boxes()
    rs = [shapely.geometry.LineString([(5.,0.), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 5)
    assert(any(workflow.utils.close((10.,0.), c) for c in poly0.boundary.coords[:-1]))
    assert(len(rivers) is 2)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,0.), c) for c in r.coords))

def test_snap3():
    # move a river point onto the huc boundary
    tb = two_boxes()
    rs = [shapely.geometry.LineString([(5.,0.), (10.001,0), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 5)
    assert(any(workflow.utils.close((10.,0.), c) for c in poly0.boundary.coords[:-1]))
    assert(len(rivers) is 2)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,0.), c) for c in r.coords))

def test_snap4():
    # move a huc point onto the river
    b1 = [(0, -5), (10,-5), (10, 0.001), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5), (10, 0.001)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    rs = [shapely.geometry.LineString([(5.,0.), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 5)
    assert(any(workflow.utils.close((10.,0.), c) for c in poly0.boundary.coords[:-1]))
    assert(len(rivers) is 2)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,0.), c) for c in r.coords))

def test_snap5():
    # move both
    b1 = [(0, -5), (10,-5), (10, 0.001), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5), (10, 0.001)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    rs = [shapely.geometry.LineString([(5.,0.), (10.001,0), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 5)
    assert(any(workflow.utils.close((10.,0.), c) for c in poly0.boundary.coords[:-1]))
    assert(len(rivers) is 2)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,0.), c) for c in r.coords))
        
def test_snap6():
    # three boxes, intersection at the junction, do nothing
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5)]
    b3 = [(0, 5), (10,5), (20,5), (20,10), (0,10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))

    rs = [shapely.geometry.LineString([(5.,0.), (10.,5),]),
          shapely.geometry.LineString([(15.,0.), (10.,5),]),
          shapely.geometry.LineString([(10.,5.), (10,10)]),
    ]

    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)

    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 4)
    poly1 = hucs.polygon(1)
    assert(len(poly1.boundary.coords[:-1]) is 4)
    poly2 = hucs.polygon(2)
    assert(len(poly2.boundary.coords[:-1]) is 6)
    assert(any(workflow.utils.close((10.,10.), c) for c in poly2.boundary.coords[:-1]))
    assert(any(workflow.utils.close((10.,5.), c) for c in poly2.boundary.coords[:-1]))

    assert(len(rivers) is 3)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,5.), c) for c in r.coords))


def test_snap7():
    # three boxes, intersection at the junction, do nothing
    b1 = [(0, -5), (10,-5), (10.001,5.001), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10.001,5.001)]
    b3 = [(0, 5), (10.001,5.001), (20,5), (20,10), (0,10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))

    rs = [shapely.geometry.LineString([(5.,0.), (10.,5),]),
          shapely.geometry.LineString([(15.,0.), (10.,5),]),
          shapely.geometry.LineString([(10.,5.), (10,10)]),]

    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)

    poly0 = hucs.polygon(0)
    assert(len(poly0.boundary.coords[:-1]) is 4)
    poly1 = hucs.polygon(1)
    assert(len(poly1.boundary.coords[:-1]) is 4)
    poly2 = hucs.polygon(2)
    assert(len(poly2.boundary.coords[:-1]) is 6)
    assert(any(workflow.utils.close((10.,10.), c) for c in poly2.boundary.coords[:-1]))
    assert(any(workflow.utils.close((10.,5.), c) for c in poly2.boundary.coords[:-1]))

    assert(len(rivers) is 3)
    for r in rivers:
        assert(len(r.coords) is 2)
        assert(any(workflow.utils.close((10.,5.), c) for c in r.coords))
        
