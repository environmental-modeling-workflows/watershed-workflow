import pytest
import shapely.geometry

from watershed_workflow.test.shapes import *

import watershed_workflow.utils
import watershed_workflow.split_hucs
import watershed_workflow.hydrography as hydro
import watershed_workflow.river_tree
import watershed_workflow.plot


def data(poly_hucs, river_segs):
    if isinstance(poly_hucs, list):
        poly_hucs = geopandas.GeoDataFrame(geometry=poly_hucs)
    hucs = watershed_workflow.split_hucs.SplitHUCs(poly_hucs)

    if isinstance(river_segs, list):
        river_segs = geopandas.GeoDataFrame(geometry=river_segs)
    rivers = watershed_workflow.river_tree.createRivers(river_segs, method='geometry')
    for tree in rivers:
        assert (tree.isConsistent())
    return hucs, rivers


def check_twoboxes(hucs):
    assert len(hucs) == 2
    poly0 = hucs.computePolygon(0)
    assert (watershed_workflow.utils.isClose(
        poly0, shapely.geometry.Polygon([(0, -5), (10, -5), (10, 0), (10, 5), (0, 5)])))
    poly1 = hucs.computePolygon(1)
    assert (watershed_workflow.utils.isClose(
        poly1, shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5), (10, 0)])))


def hilevSnap(hucs, rivers, tol):
    """This helper function simply does all three hydro operations."""
    hydro.snapHUCsJunctions(hucs, rivers, 3*tol)
    for river in rivers:
        hydro.snapReachEndpoints(hucs, river, tol)
    watershed_workflow.hydrography.cutAndSnapCrossings(hucs, rivers, tol)

#
# test0:
# one box, one reach with outlet on the boundary
#   one box:  [0,10] x [-5,5]
#   one trunk reach: [5,0] -> [10,0]
#
def check0(hucs, rivers):
    assert len(hucs) == 1
    poly0 = hucs.computePolygon(0)
    print(poly0.boundary.coords[:])

    # close to the polygon
    assert (watershed_workflow.utils.isClose(
        poly0, shapely.geometry.Polygon([(0, -5), (10, -5), (10, 0), (10, 5), (0, 5)])),
            1.e-3)

    riverlist = [r.linestring for r in rivers[0]]
    assert len(riverlist) == 1
    print(riverlist[0].coords[:])

    # close to the river
    for tree in rivers:
        assert tree.isConsistent()
    assert watershed_workflow.utils.isClose(riverlist[0],
                                            shapely.geometry.LineString([(5, 0), (10, 0)]),
                                            1.e-3)

    # exact at the endpoint
    assert any(watershed_workflow.utils.isClose(riverlist[0].coords[-1], c) for c in poly0.boundary.coords)
    

def test_snap0():
    """null-op, all data are already correct"""
    b1 = [(0, -5), (10, -5), (10, 0.), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (10., 0.)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0a():
    """add reach endpoint to huc"""
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (10., 0)]), ]
    hucs, rivers = data(tb, rs)
    assert (len(rivers) == 1)
    assert (len(rivers[0]) == 1)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0b():
    """snap reach endpoint to huc"""
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (9.999, 0)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0c():
    """snap reach endpoint to huc (from right)"""
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (10.001, 0)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0d():
    """cut, not snap"""
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (10.2, 0)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0e():
    """snap a river endpoint onto a huc, near point on river, point on huc"""
    b1 = [(0, -5), (10, -5), (10, 0.), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (10.001, 0.)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0h():
    """cut a river at huc exterior boundary, point already exists (no duplicate point)"""
    b1 = [(0, -5), (10, -5), (10, 0.), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (15, 0.)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

def test_snap0j():
    """cut a river at huc exterior boundary with an extra external point"""
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5., 0.), (15, 0.), (16, 0.)]), ]
    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check0(hucs, rivers)

#
# test1:
# two boxes: split a reach
#   two boxes:  [0,10] x [-5,5] & [10,20] x [-5,5]
#   two reaches: [5,0] -> [10,0] & [10,0] -> [15,0]
#
def check1(hucs, rivers, extra_point=None):
    check_twoboxes(hucs)
    
    riverlist = [r.linestring for r in rivers[0]]
    if extra_point == 'reach right':
        assert len(riverlist) == 3
    else:
        assert len(riverlist) == 2

    # note preorder!
    i = 0
    if extra_point == 'right':
        assert watershed_workflow.utils.isClose(riverlist[i],
                                           shapely.geometry.LineString([(10, 0), (10.001,0), (15, 0)]))
    elif extra_point == 'reach right':
        assert watershed_workflow.utils.isClose(riverlist[i],
                                           shapely.geometry.LineString([(10.2, 0), (15, 0)]))
        i += 1
        assert watershed_workflow.utils.isClose(riverlist[i],
                                           shapely.geometry.LineString([(10, 0), (10.2,0)]))
    else:
        assert watershed_workflow.utils.isClose(riverlist[i],
                                           shapely.geometry.LineString([(10, 0), (15, 0)]))

    i += 1
    if extra_point == 'left':
        assert watershed_workflow.utils.isClose(riverlist[i],
                                                shapely.geometry.LineString([(5, 0), (9.999,0), (10, 0)]))
    else:
        assert watershed_workflow.utils.isClose(riverlist[i],
                                                shapely.geometry.LineString([(5, 0), (10, 0)]))


def test_snap1a(two_boxes):
    """generic intersection with no movement only additions"""
    rs = [shapely.geometry.LineString([(5., 0.), (15, 0)]), ]
    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check1(hucs, rivers, None)

def test_snap1b(two_boxes):
    """generic intersection with nearby point"""
    rs = [shapely.geometry.LineString([(5., 0.), (9.999, 0.), (15, 0)]), ]
    hucs, rivers = data(two_boxes, rs)
    hydro.cutAndSnapCrossings(hucs, rivers, 0.1)
    check1(hucs, rivers, 'left')

def test_snap1c(two_boxes):
    """generic intersection with nearby reach endpoints"""
    rs = [shapely.geometry.LineString([(5., 0.), (9.999, 0.)]),
          shapely.geometry.LineString([(9.999, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    for river in rivers:
        hydro.snapReachEndpoints(hucs, river, 0.1)
    check1(hucs, rivers)

def test_snap1d(two_boxes):
    """generic intersection with nearby point to the right"""
    rs = [shapely.geometry.LineString([(5., 0.), (10.001, 0.), (15, 0)]), ]
    hucs, rivers = data(two_boxes, rs)
    hydro.cutAndSnapCrossings(hucs, rivers, 0.1)
    check1(hucs, rivers, 'right')

def test_snap1e(two_boxes):
    """generic intersection with nearby reach endpoints to the right"""
    rs = [shapely.geometry.LineString([(5., 0.), (10.001, 0.)]),
          shapely.geometry.LineString([(10.001, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    for river in rivers:
        hydro.snapReachEndpoints(hucs, river, 0.1)
    check1(hucs, rivers)

def test_snap1f(two_boxes):
    """an extra reach is added to the right because the upstream reach gets cut and the downstream portion is above threshold"""
    rs = [shapely.geometry.LineString([(5., 0.), (10.2, 0.)]),
          shapely.geometry.LineString([(10.2, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    hydro.cutAndSnapCrossings(hucs, rivers, 0.1)
    check1(hucs, rivers, 'reach right')
    
    
#
# test2:
# two boxes, junction at the internal boundary
#   two boxes:  [0,10] x [-5,5] & [10,20] x [-5,5]
#   three reaches: [5,2.5] -> [10,0],
#                  [5,-2.5] -> [10,0],
#                & [10,0] -> [15,0]
#
def check2(hucs, rivers, extra_point=None):
    check_twoboxes(hucs)

    riverlist = [r.linestring for r in rivers[0]]
    assert len(riverlist) == 3

    # note preorder
    # - trunk
    assert watershed_workflow.utils.isClose(riverlist[0],
                                            shapely.geometry.LineString([(10, 0), (15, 0)]))

    # - paddlers left
    assert watershed_workflow.utils.isClose(riverlist[1],
                                            shapely.geometry.LineString([(5,-2.5), (10,0)]))

    # - paddlers right
    assert watershed_workflow.utils.isClose(riverlist[2],
                                            shapely.geometry.LineString([(5,2.5), (10,0)]))


def test_snap2a(two_boxes):
    """just need to insert the point on the box"""
    rs = [shapely.geometry.LineString([(5., 2.5), (10., 0.)]),
          shapely.geometry.LineString([(5., -2.5), (10., 0.)]),
          shapely.geometry.LineString([(10., 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check2(hucs, rivers, None)


def test_snap2b(two_boxes):
    """insert point on box, move reach endpoints from right side"""
    rs = [shapely.geometry.LineString([(5., 2.5), (10.001, 0.)]),
          shapely.geometry.LineString([(5., -2.5), (10.001, 0.)]),
          shapely.geometry.LineString([(10.001, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check2(hucs, rivers, None)


def test_snap2c(two_boxes):
    """insert point on box, move reach endpoints from left side"""
    rs = [shapely.geometry.LineString([(5., 2.5), (9.999, 0.)]),
          shapely.geometry.LineString([(5., -2.5), (9.999, 0.)]),
          shapely.geometry.LineString([(9.999, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check2(hucs, rivers, None)


def test_snap2d(two_boxes):
    """insert point on box, move reach endpoints from right, delete extra points"""
    rs = [shapely.geometry.LineString([(5., 2.5), (10.0005, 0.0005), (10.001, 0.)]),
          shapely.geometry.LineString([(5., -2.5), (10.0005, -0.0005), (10.001, 0.)]),
          shapely.geometry.LineString([(10.001, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check2(hucs, rivers, None)


def test_snap2e(two_boxes):
    """insert point on box, move reach endpoints from leftside, delete extra points"""
    rs = [shapely.geometry.LineString([(5., 2.5), (9.999, 0.)]),
          shapely.geometry.LineString([(5., -2.5), (9.999, 0.)]),
          shapely.geometry.LineString([(9.999, 0.), (9.9995, 0.), (15, 0.)]),]
    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check2(hucs, rivers, None)


#
# test3:
# two boxes, 3-way junction at the internal boundary
#   two boxes:  [0,10] x [-5,5] & [10,20] x [-5,5]
#   three reaches: [5,2.5] -> [10,0],
#                  [5,-2.5] -> [10,0],
#                  [5, 0] -> [10,0],
#                & [10,0] -> [15,0]
#
def check3(hucs, rivers, extra_point=None):
    check_twoboxes(hucs)

    riverlist = [r.linestring for r in rivers[0]]
    assert len(riverlist) == 4

    # note preorder
    # - trunk
    assert watershed_workflow.utils.isClose(riverlist[0],
                                            shapely.geometry.LineString([(10, 0), (15, 0)]))

    # - paddlers left
    assert watershed_workflow.utils.isClose(riverlist[1],
                                            shapely.geometry.LineString([(5,-2.5), (10,0)]))

    # - paddlers center
    assert watershed_workflow.utils.isClose(riverlist[2],
                                            shapely.geometry.LineString([(5,0), (10,0)]))

    # - paddlers right
    assert watershed_workflow.utils.isClose(riverlist[3],
                                            shapely.geometry.LineString([(5,2.5), (10,0)]))


def test_snap3a(two_boxes):
    """add point on interior boundary"""
    rs = [shapely.geometry.LineString([(5., 2.5), (10., 0.)]),
          shapely.geometry.LineString([(5., -2.5), (10., 0.)]),
          shapely.geometry.LineString([(5., 0.), (10., 0.)]),
          shapely.geometry.LineString([(10., 0.), (15, 0.)]),]

    hucs, rivers = data(two_boxes, rs)
    hilevSnap(hucs, rivers, 0.1)
    check3(hucs, rivers, None)


#
# NOTE: these actually should be caught by mergeShortReaches(), not
# cutAndSnapCrossings()
#
# def test_snap3b(two_boxes):
#     """Remove a short between-junction segment to the left"""
#     rs = [shapely.geometry.LineString([(5., 2.5), (9.95, 0.)]),
#           shapely.geometry.LineString([(5., 0.), (9.95, 0.)]),
#           shapely.geometry.LineString([(9.95, 0.), (10, 0.)]),
#           shapely.geometry.LineString([(5., -2.5), (10., 0.)]),
#           shapely.geometry.LineString([(10., 0.), (15, 0.)]),]

#     hucs, rivers = data(two_boxes, rs)
#     hilevSnap(hucs, rivers, 0.1)
#     check3(hucs, rivers, None)


# def test_snap3c(two_boxes):
#     """Remove a short between-junction segment, to the right"""
#     rs = [shapely.geometry.LineString([(5., 2.5), (10., 0.)]),
#           shapely.geometry.LineString([(5., 0.), (10., 0.)]),
#           shapely.geometry.LineString([(10., 0.), (10.05, 0.)]),
#           shapely.geometry.LineString([(5., -2.5), (10.05, 0.)]),
#           shapely.geometry.LineString([(10.05, 0.), (15, 0.)]),]

#     hucs, rivers = data(two_boxes, rs)
#     hilevSnap(hucs, rivers, 0.1)
#     check3(hucs, rivers, None)
    

    
#
# test6:
# three boxes, 2-way junction at the triple point of the boxes
#   three boxes:  [0,10] x [-5,5],
#                 [10,20] x [-5,5]
#                 [00,20] x [5,10]
#   three reaches: [5,0] -> [10,5],
#                  [15,0] -> [10,5],
#                & [10,5] -> [10,10]
#
def check6(hucs, rivers):
    assert (len(hucs) is 3)
    poly0 = hucs.computePolygon(0)
    assert (watershed_workflow.utils.isClose(
        poly0, shapely.geometry.Polygon([(0, -5), (10, -5), (10, 5), (0, 5)])))
    poly1 = hucs.computePolygon(1)
    assert (watershed_workflow.utils.isClose(
        poly1, shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5)])))
    poly2 = hucs.computePolygon(2)
    print(list(poly2.boundary.coords))
    print(
        list(
            shapely.geometry.Polygon([(0, 5), (10, 5), (20, 5), (20, 10), (10, 10),
                                      (0, 10)]).boundary.coords))
    assert (watershed_workflow.utils.isClose(
        poly2, shapely.geometry.Polygon([(0, 5), (10, 5), (20, 5), (20, 10), (10, 10), (0, 10)])))

    assert (len(rivers[0]) is 3)
    riverlist = [r.linestring for r in rivers[0]]
    print(riverlist[0].coords[:])
    assert (watershed_workflow.utils.isClose(riverlist[0],
                                           shapely.geometry.LineString([(10., 5.), (10., 10.)])))
    print(riverlist[1].coords[:])
    assert (watershed_workflow.utils.isClose(riverlist[1],
                                           shapely.geometry.LineString([(15., 0.), (10., 5.)])))
    print(riverlist[2].coords[:])
    assert (watershed_workflow.utils.isClose(riverlist[2],
                                           shapely.geometry.LineString([(5., 0.), (10., 5.)])))

    for tree in rivers:
        assert (tree.isConsistent())


def test_snap6():
    """three boxes, intersection at the junction, null op"""
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    b2 = [(10, -5), (20, -5), (20, 5), (10, 5)]
    b3 = [(0, 5), (10, 5), (20, 5), (20, 10), (0, 10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))

    rs = [
        shapely.geometry.LineString([(5., 0.), (10., 5), ]),
        shapely.geometry.LineString([(15., 0.), (10., 5), ]),
        shapely.geometry.LineString([(10., 5.), (10, 10)]),
    ]

    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check6(hucs, rivers)


def test_snap6a():
    """three boxes, intersection at the junction, move box triple point to intersection"""
    b1 = [(0, -5), (10, -5), (10.001, 5.001), (0, 5)]
    b2 = [(10, -5), (20, -5), (20, 5), (10.001, 5.001)]
    b3 = [(0, 5), (10.001, 5.001), (20, 5), (20, 10), (0, 10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))

    rs = [
        shapely.geometry.LineString([(5., 0.), (10., 5), ]),
        shapely.geometry.LineString([(15., 0.), (10., 5), ]),
        shapely.geometry.LineString([(10., 5.), (10, 10)]),
    ]

    hucs, rivers = data(tb, rs)
    hilevSnap(hucs, rivers, 0.1)
    check6(hucs, rivers)
    
    

