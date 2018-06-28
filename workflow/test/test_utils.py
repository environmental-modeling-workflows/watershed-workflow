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

    
def test_point_not_there():
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

def test_point_there():
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

def test_first_point():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(0, -1), (0, 1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 1)
    print(list(lines[0].coords))
    assert(lines[0] == line)

def test_last_point():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0)])
    cut = shapely.geometry.LineString([(1, -1), (1, 1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 1)
    print(list(lines[0].coords))
    assert(lines[0] == line)

def test_two_crossings():
    line = shapely.geometry.LineString([(0,0), (0.5, 0), (1,0), (1.5,0), (2, 0)])
    cut = shapely.geometry.LineString([(0.5, -1), (0.5, 1), (1.5,1), (1.5,-1)])
    lines = workflow.utils.cut(line, cut)
    assert(len(lines) == 3)
    print(list(lines[0].coords))
    assert(lines[0] == shapely.geometry.LineString([(0,0), (0.5, 0)]))
    assert(lines[1] == shapely.geometry.LineString([(0.5,0), (1, 0),(1.5, 0)]))
    assert(lines[2] == shapely.geometry.LineString([(1.5,0), (2, 0)]))
    
def test_two_ways():
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

def test_intersect_and_split(two_boxes):
    boundaries, intersections = workflow.utils.intersect_and_split(two_boxes)
    assert(len(boundaries) is 2)

    for b in boundaries:
        assert type(b) is shapely.geometry.LineString
        assert len(b.coords) == 4

    assert(len(intersections) is 2)
    for i,row in enumerate(intersections):
        assert(len(row) is 2)
        for j,entry in enumerate(row):
            print("At i,j=%d,%d type is %r"%(i,j,type(entry)))
            if i <= j:
                assert entry is None
            else:
                assert type(entry) is shapely.geometry.LineString
                assert len(entry.coords) is 2
                workflow.utils.close(entry.coords[0], (10,-5))
                workflow.utils.close(entry.coords[1], (10,5))
        
    
