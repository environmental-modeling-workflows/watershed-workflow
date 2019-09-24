import pytest
import numpy as np
import shapely.geometry

import workflow.utils
from workflow.test.shapes import two_boxes


def test_close():
    # points
    p0 = shapely.geometry.Point((0.,0.))
    p1 = shapely.geometry.Point((0.,0.))
    assert(workflow.utils.close(p0,p1))

    p1 = shapely.geometry.Point((0.,0.000001))
    assert(not workflow.utils.close(p0,p1))

    p1 = shapely.geometry.Point((0.,1.e-8))
    assert(workflow.utils.close(p0,p1))

    # point and tuples
    assert(workflow.utils.close(p0,(0,1.e-8)))
    assert(workflow.utils.close((0,0),(0,1.e-8)))

    # fails
    assert(not workflow.utils.close((0,0,0),(0,0)))

    # lineseg
    l0 = shapely.geometry.LineString([(0,0), (1,0), (2,0)])
    l1 = shapely.geometry.LineString([(0,0.001), (1,0), (2,0)])
    l2 = shapely.geometry.LineString([(0,0.001), (2,0)])
    assert(workflow.utils.close(l0,l1,0.01))
    assert(not workflow.utils.close(l0,l1,0.0001))
    assert(not workflow.utils.close(l0,l2,100))

    # polygon
    p0 = shapely.geometry.Polygon([(0,0), (1,0), (1,1), (0,1)])
    p1 = shapely.geometry.Polygon([(1,0.001), (1.001,1), (0,1), (0,0)])
    p2 = shapely.geometry.Polygon([(0,0),(0,1), (1.001,1), (1,0.001)])
    assert(workflow.utils.close(p0,p1, 0.01))
    assert(not workflow.utils.close(p0,p1, 0.0001))
    assert(workflow.utils.close(p0,p2, 0.01))
    assert(not workflow.utils.close(p0,p2, 0.0001))
    


def test_contains():
    coords = np.array([[1.03425,0.0013], [0.0035,1.03523], [-1.09824,0.0033], [0.0012,-1.04856]])
    linecoords = np.array([[0.1394,0.0492], [3.1415, 1.1394]])
    
    def wiggle(coords):
        random = np.random.random((len(coords),2))
        random = 2*(random - .5) * .01
        return coords + random

    contains = []
    for i in range(100):
        newc = wiggle(coords)
        newl = wiggle(linecoords)
        shp = shapely.geometry.Polygon(newc)
        line = shapely.geometry.LineString(newl)
        lines = workflow.utils.cut(line, shp.boundary)

        contains.append(workflow.utils.contains(shp, lines[0]))

    print("Contains % = ", sum(1 for i in contains if i)/100.0)
    assert(all(contains))

    
def test_cut_point_not_there():
    line = shapely.geometry.LineString([(0,0), (1,0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 2)
    l1 = lines[0]
    l2 = lines[1]
    print( l1.coords[:])
    print( l2.coords[:])
    assert(l1 == shapely.geometry.LineString([(0,0), (0.5,0)]))
    assert(l2 == shapely.geometry.LineString([(0.5,0), (1,0)]))

def test_cut_point_there():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 2)
    l1 = lines[0]
    l2 = lines[1]
    print( l1.coords[:])
    print( l2.coords[:])
    assert(l1 == shapely.geometry.LineString([(0,0), (0.5,0)]))
    assert(l2 == shapely.geometry.LineString([(0.5,0), (1,0)]))

def test_cut_point_nearly_there_after():
    line = shapely.geometry.LineString([(0,0), (0.50001, 0), (1,0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
    lines = workflow.utils.cut(line, cut, 0.01)
    assert(len(lines) == 2)
    l1 = lines[0]
    l2 = lines[1]
    print( l1.coords[:])
    print( l2.coords[:])
    assert(l1 == shapely.geometry.LineString([(0,0), (0.5,0)]))
    assert(l2 == shapely.geometry.LineString([(0.5,0), (1,0)]))

def test_cut_point_nearly_there_before():
    line = shapely.geometry.LineString([(0,0), (0.49999, 0), (1,0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1)])
    lines = workflow.utils.cut(line, cut, 0.01)
    assert(len(lines) == 2)
    l1 = lines[0]
    l2 = lines[1]
    print( l1.coords[:])
    print( l2.coords[:])
    assert(l1 == shapely.geometry.LineString([(0,0), (0.5,0)]))
    assert(l2 == shapely.geometry.LineString([(0.5,0), (1,0)]))
    
def test_cut_first_point():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(0, -1), (0, 1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 1)
    print(list(lines[0].coords))
    assert(workflow.utils.close(lines[0], line))

def test_cut_nearly_first_point():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(0.001, -1), (0.001, 1)])
    lines = workflow.utils.cut(line, cut,0.01)
    assert(len(lines) == 1)
    print(list(lines[0].coords))
    assert(workflow.utils.close(lines[0], shapely.geometry.LineString([(0.001,0), (0.5, 0), (1,0)])))
    
def test_cut_last_point():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(1, -1), (1, 1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 1)
    print(list(lines[0].coords))
    assert(workflow.utils.close(lines[0], line))

def test_cut_nearly_last_point():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(0.9999, -1), (0.9999, 1)])
    lines = workflow.utils.cut(line, cut,0.01)
    assert(len(lines) == 1)
    print(list(lines[0].coords))
    assert(workflow.utils.close(lines[0], shapely.geometry.LineString([(0.,0), (0.5, 0), (0.9999,0)])))
    
def test_cut_two_crossings():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0), (1.5,0), (2, 0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1), (1.5,1), (1.5,-1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 3)
    print(list(lines[0].coords))
    assert(lines[0] == shapely.geometry.LineString([(0,0), (0.5, 0)]))
    assert(lines[1] == shapely.geometry.LineString([(0.5,0), (1, 0),(1.5, 0)]))
    assert(lines[2] == shapely.geometry.LineString([(1.5,0), (2, 0)]))
    
def test_cut_two_ways():
    line1 = shapely.geometry.LineString([(-1,0), (1,0)])
    line2 = shapely.geometry.LineString([(0,-1),(0,1)])
    l1_segs = workflow.utils.cut(line1, line2)
    l2_segs = workflow.utils.cut(line2, line1)
    assert(l1_segs[0] == shapely.geometry.LineString([(-1,0), (0, 0)]))
    assert(l1_segs[1] == shapely.geometry.LineString([(0,0), (1, 0)]))
    assert(l2_segs[0] == shapely.geometry.LineString([(0,-1), (0, 0)]))
    assert(l2_segs[1] == shapely.geometry.LineString([(0,0), (0,1)]))
    
    
    

def test_raises():
    line = shapely.geometry.LineString([(0,0), (1,0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.6, -1)])
    with pytest.raises(AssertionError):
        workflow.utils.cut(line, cut)


def test_intersect_point_to_segment():
    from shapely.geometry import Point as P

    # test on the first point
    p0 = workflow.utils.intersect_point_to_segment(P(0,0), P(0,0), P(0,1))
    assert(workflow.utils.close(p0, (0,0)))

    # test on the last point
    p0 = workflow.utils.intersect_point_to_segment(P(0,1), P(0,0), P(0,1))
    assert(workflow.utils.close(p0, (0,1)))
    
    # test on the line
    p0 = workflow.utils.intersect_point_to_segment(P(0,0), P(0,-1), P(0,1))
    assert(workflow.utils.close(p0, (0,0)))

    # test x-perp
    p0 = workflow.utils.intersect_point_to_segment(P(1,0), P(0,-1), P(0,1))
    assert(workflow.utils.close(p0, (0,0)))

    # test diagonal perp
    p0 = workflow.utils.intersect_point_to_segment(P(1,-1), P(-1,-1), P(1,1))
    assert(workflow.utils.close(p0, (0,0)))            

    # test colinear but negative
    p0 = workflow.utils.intersect_point_to_segment(P(-2,-2), P(-1,-1), P(1,1))
    assert(workflow.utils.close(p0, (-1,-1)))            

    # test colinear but positive
    p0 = workflow.utils.intersect_point_to_segment(P(2,2), P(-1,-1), P(1,1))
    assert(workflow.utils.close(p0, (1,1)))            

    # test not colinear but past end
    p0 = workflow.utils.intersect_point_to_segment(P(-3.3,-2.1), P(-1,-1), P(1,1))
    assert(workflow.utils.close(p0, (-1,-1)))            

    # test end but close
    p0 = workflow.utils.intersect_point_to_segment(P(-.9,-1.1), P(-1,-1), P(1,1))
    assert(workflow.utils.close(p0, (-1,-1)))            

    # test throws
    with pytest.raises(AssertionError):
        p0 = workflow.utils.intersect_point_to_segment(P(-.9,-1.1), P(1,1), P(1,1))
        
def test_neighborhood():
    p1 = shapely.geometry.LineString([(0,0), (1,1)])

    p2 = shapely.geometry.LineString([(2,2), (3,3)])
    assert(not workflow.utils.in_neighborhood(p1,p2))

    p2 = shapely.geometry.LineString([(0,3), (3,3)])
    assert(not workflow.utils.in_neighborhood(p1,p2))

    p2 = shapely.geometry.LineString([(3,0), (3,1)])
    assert(not workflow.utils.in_neighborhood(p1,p2))
    
    p2 = shapely.geometry.LineString([(1,0), (0,1)])
    assert(workflow.utils.in_neighborhood(p1,p2))

    p2 = shapely.geometry.LineString([(1,1), (2,1)])
    assert(workflow.utils.in_neighborhood(p1,p2))

    p2 = shapely.geometry.LineString([(1.01,1), (2,1)])
    assert(workflow.utils.in_neighborhood(p1,p2))

    p2 = shapely.geometry.LineString([(1.01,1), (2,1)])
    assert(not workflow.utils.in_neighborhood(p1,p2, 1.e-3))

    # single point
    p3 = shapely.geometry.Point(978563.4249385255, 1512322.6640905372)
    p4 = shapely.geometry.LineString([(977132.6302807415, 1507051.5674243502), (979578.2010028946, 1515834.394320889)])
    assert(workflow.utils.in_neighborhood(p3,p4,0.))

    
    
    
def test_perp():
    l1 = shapely.geometry.LineString([(0,0), (1,0)])
    p = (0.5,0)
    p3 = workflow.utils.find_perp(l1, p)
    assert(abs(p3[0] - 0.5) < 1.e-6)
    assert(abs(p3[1] - 0.) > 1.e-6)

    l1 = shapely.geometry.LineString([(0,0), (0,1)])
    p = (0,0.5)
    p3 = workflow.utils.find_perp(l1, p)
    assert(abs(p3[1] - 0.5) < 1.e-6)
    assert(abs(p3[0] - 0.) > 1.e-6)

    l1 = shapely.geometry.LineString([(0,0), (1,1)])
    p = (0.5,0.5)
    p3 = workflow.utils.find_perp(l1, p)
    assert(abs( (p3[1] - 0.5) - (p3[0] - 0.5) ) < 1.e-6)

    
    
def test_merge0():
    l1 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (2.5,2.5)])
    l2 = shapely.geometry.LineString([(2.5,2.5), (3,3), (4,4), (5,5)])
    l3 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (2.5,2.5)])
    l4 = shapely.geometry.LineString([(2.5,2.5), (3,3), (4,4), (5,5)])

    ml1 = shapely.geometry.MultiLineString([l1,l2])
    ml2 = shapely.geometry.MultiLineString([l3,l4])

    new_ml = workflow.utils.merge(ml1,ml2)
    assert(len(new_ml) == 2)
    assert(workflow.utils.close(new_ml[0], l1))
    assert(workflow.utils.close(new_ml[1], l2))
                                           

def test_merge1():
    l1 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (2.5,2.5)])
    l2 = shapely.geometry.LineString([(2.5,2.5), (3,3), (4,4), (5,5)])
    l3 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (3,3), (3.5,3.5)])
    l4 = shapely.geometry.LineString([(3.5, 3.5), (4,4),(5,5)])

    ml1 = shapely.geometry.MultiLineString([l1,l2])
    ml2 = shapely.geometry.MultiLineString([l3,l4])

    new_ml = workflow.utils.merge(ml1,ml2)
    assert(len(new_ml) == 3)
    assert(workflow.utils.close(new_ml[0], l1))
    assert(workflow.utils.close(new_ml[1], shapely.geometry.LineString([(2.5,2.5),(3,3),(3.5,3.5)])))
    assert(workflow.utils.close(new_ml[2], l4))


def test_merge1():
    l1 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (2.5,2.5)])
    l2 = shapely.geometry.LineString([(2.5,2.5), (3,3), (4,4), (5,5)])
    l3 = shapely.geometry.LineString([(0,0), (1,1), (1.5,1.5)])
    l4 = shapely.geometry.LineString([(1.5,1.5),(2,2),(3,3), (3.5, 3.5)])
    l5 = shapely.geometry.LineString([(3.5, 3.5), (4,4),(5,5)])

    ml1 = shapely.geometry.MultiLineString([l1,l2])
    ml2 = shapely.geometry.MultiLineString([l3,l4,l5])

    new_ml = workflow.utils.merge(ml1,ml2)
    assert(len(new_ml) == 4)
    assert(workflow.utils.close(new_ml[0], l3))
    assert(workflow.utils.close(new_ml[1], shapely.geometry.LineString([(1.5,1.5),(2,2),(2.5,2.5)])))
    assert(workflow.utils.close(new_ml[2], shapely.geometry.LineString([(2.5,2.5),(3,3),(3.5,3.5)])))
    assert(workflow.utils.close(new_ml[3], l5))
    
    

def test_generate_points():
    # with points
    p = shapely.geometry.Point(978563., 1512322.)
    with pytest.raises(TypeError):
        for r in workflow.utils.generate_rings(shapely.geometry.mapping(p)):
            pass
    c1 = list(workflow.utils.generate_coords(shapely.geometry.mapping(p)))
    assert(1 == len(c1))
    assert(np.allclose(np.array([978563., 1512322.]), np.array(c1)))

def test_generate_lines():
    l1 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (2.5,2.5)])
    l1a = np.array([(0,0), (1,1), (2,2), (2.5,2.5)])
    for r in workflow.utils.generate_rings(shapely.geometry.mapping(l1)):
        assert(np.allclose(l1a, np.array(r)))

    for c1,c2 in zip(l1a, workflow.utils.generate_coords(shapely.geometry.mapping(l1))):
        assert(workflow.utils.close(tuple(c1), c2))

def test_generate_multilines():
    l1 = shapely.geometry.LineString([(0,0), (1,1), (2,2), (2.5,2.5)])
    l1a = np.array([(0,0), (1,1), (2,2), (2.5,2.5)])

    l2 = shapely.geometry.LineString([(2.5,2.5), (3,3), (4,4), (5,5)])
    l2a = np.array([(2.5,2.5), (3,3), (4,4), (5,5)])

    ml = shapely.geometry.MultiLineString([l1,l2])
    ringlist = list(workflow.utils.generate_rings(shapely.geometry.mapping(ml)))
    assert(np.allclose(l1a, np.array(ringlist[0])))
    assert(np.allclose(l2a, np.array(ringlist[1])))

    coordlist = np.array(list(workflow.utils.generate_coords(shapely.geometry.mapping(ml))))
    assert(np.allclose(np.concatenate([l1a,l2a]), coordlist))

def test_generate_polygons():
    poly1 = shapely.geometry.Polygon([(0,0), (1,1), (2,2), (2.5,2.5)])
    poly1a = np.array([(0,0), (1,1), (2,2), (2.5,2.5), (0,0)])

    for r in workflow.utils.generate_rings(shapely.geometry.mapping(poly1)):
        assert(np.allclose(poly1a, np.array(r)))

    for c1,c2 in zip(poly1a, workflow.utils.generate_coords(shapely.geometry.mapping(poly1))):
        assert(workflow.utils.close(tuple(c1), c2))

def test_generate_multipolygons():
    poly1 = shapely.geometry.Polygon([(0,0), (1,1), (2,2), (2.5,2.5)])
    poly1a = np.array([(0,0), (1,1), (2,2), (2.5,2.5), (0,0)])
    
    poly2 = shapely.geometry.Polygon([(2.5,2.5), (3,3), (4,4), (5,5)])
    poly2a = np.array([(2.5,2.5), (3,3), (4,4), (5,5), (2.5,2.5)])
    
    mpoly = shapely.geometry.MultiPolygon([poly1, poly2])

    ringlist = list(workflow.utils.generate_rings(shapely.geometry.mapping(mpoly)))
    assert(np.allclose(poly1a, np.array(ringlist[0])))
    assert(np.allclose(poly2a, np.array(ringlist[1])))

    coordlist = np.array(list(workflow.utils.generate_coords(shapely.geometry.mapping(mpoly))))
    assert(np.allclose(np.concatenate([poly1a,poly2a]), coordlist))
    
                                           
    
    
        
    

