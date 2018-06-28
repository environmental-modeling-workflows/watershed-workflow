"""A series of tests to see how shapely works."""

import numpy as np
import shapely.geometry
import workflow.utils

def test_intersection_intersects():
    """Does the intersection of two shapes always intersect those shapes?  SURPRISE"""
    shp = shapely.geometry.Polygon([(1.03425,0.0013), (0.0035,1.03523), (-1.09824,0.0033), (0.0012,-1.04856)])
    line = shapely.geometry.LineString([(0.1394,0.0492), (3.1415, 1.1394)])

    # check if a linestring intersecting a linestring intersects the line
    p = shp.boundary.intersection(line)

    # Lesson learned:
    # The intersection of two lines does not always intersect either line!
    # Nonrobustness of point geometry!
    assert(not shp.boundary.intersects(p))
    assert(not line.intersects(p))

    # same as
    assert(not shp.boundary.contains(p))

    # is this obviously safe?  NO!
    assert(shp.contains(p))


def test_simplify():
    """Does simplify ever move the end points of a linestring?  YAY!"""
    coords = np.array([(-.001, -.001), (0,0), (100,0), (100.001,.001)])

    def wiggle(coords):
        random = np.random.random((len(coords),2))
        random = 2*(random - .5) * .001
        return coords + random


    good = []
    for i in range(100):
        newc = wiggle(coords)
        ls = shapely.geometry.LineString(newc)
        ls_s = ls.simplify(.01)
        mygood = ((len(ls_s.coords) is 2) and 
                  workflow.utils.close(ls_s.coords[0], ls.coords[0], 1.e-10) and
                  workflow.utils.close(ls_s.coords[-1], ls.coords[-1], 1.e-10))
        good.append(mygood)

    print("Good % = ", sum(1 for i in good if i)/100.0)
    assert(all(good))

