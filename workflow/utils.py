"""Shapely-only utilities not provided by shapely"""
import logging
import subprocess
import numpy as np
import shapely.geometry
import shapely.ops
import shapely.affinity

_tol = 1.e-7
def close(s1, s2, tol=_tol):
    """Are two shapes topologically equivalent and gemoetrically close"""
    if type(s1) is shapely.geometry.Point:
        return close(s1.coords[0], s2, tol)
    if type(s2) is shapely.geometry.Point:
        return close(s1, s2.coords[0], tol)
    if type(s1) != type(s2):
        return False
    if type(s1) is tuple:
        if len(s1) != len(s2):
            return False
        return sum((p1 - p2)**2 for p1,p2 in zip(s1,s2)) < tol**2
    elif type(s1) is shapely.geometry.LineString:
        if len(s1.coords) != len(s2.coords):
            return False
        print("allclose?", tol, list(s1.coords), list(s2.coords), np.allclose(np.array(s1.coords), np.array(s2.coords), tol, tol))
        return np.allclose(np.array(s1.coords), np.array(s2.coords), tol, tol)
    elif type(s1) is shapely.geometry.MultiLineString:
        if len(s1) != len(s2):
            return False
        good2 = [False,]*len(s2)
        for l1 in s1:
            found = False
            for i,l2 in enumerate(s2):
                if not good2[i]:
                    if close(l1,l2,tol):
                        good2[i] = True
                        found = True
                        break
            if not found:
                return False
        return True
    elif type(s1) is shapely.geometry.Polygon:
        if len(s1.boundary.coords) != len(s2.boundary.coords):
            return False
        ls1 = s1.boundary.coords[:-1]
        ls2 = np.array(s2.boundary.coords[:-1])
        ls2f = np.flipud(ls2)
        return any(np.allclose(ls1, np.roll(ls2, i, 0), tol, tol) for i in range(len(ls2))) or \
            any(np.allclose(ls1, np.roll(ls2f, i, 0), tol, tol) for i in range(len(ls2)))
    else:
        raise NotImplementedError("Not implemented for type '%r'"%type(s1))

def contains(s1, s2, tol=_tol):
    """A contains algorithm that deals with close/roundoff issues"""
    return s1.buffer(tol,2).contains(s2)


def cut(line, cutline, tol=1.e-5):
    """Cuts a line at all intersections with cutline."""

    def plot():
        from matplotlib import pyplot as plt
        plt.plot(cutline.xy[0], cutline.xy[1], 'k-x', linewidth=3)
        plt.plot(line.xy[0], line.xy[1], 'g-+', linewidth=3)

    assert(type(line) is shapely.geometry.LineString)
    assert(type(cutline) is shapely.geometry.LineString)
    assert(line.intersects(cutline))

    segs = []
    coords = list(line.coords)

    segcoords = [coords[0],]
    i = 0
    while i < len(coords)-1:
        seg = shapely.geometry.LineString(coords[i:i + 2])
        #logging.debug("Intersecting seg %d"%i)
        point = seg.intersection(cutline)
        if type(point) is shapely.geometry.GeometryCollection and len(point) is 0:
            #logging.debug("Cut seg no intersection")
            segcoords.append(seg.coords[-1])
            i += 1
        elif type(point) is shapely.geometry.Point:
            #logging.debug("Cut intersected at point")
            #logging.debug("  inter point: %r"%list(point.coords[0]))
            #logging.debug("  seg final point: %r"%list(seg.coords[-1]))
            #logging.debug("  close? = %r"%(close(point, seg.coords[-1], tol)))
            if close(point, seg.coords[-1], tol):
                # intersects at the far point
                segs.append(shapely.geometry.LineString(segcoords+[point,]))
                #logging.debug("  appended full segment: %r"%(list(segs[-1].coords)))

                if (i < len(coords)-2):
                    #logging.debug("    (not the end)")
                    segcoords = [point,coords[i+2]]
                else:
                    #logging.debug("    (the end)")
                    segcoords = [point,]
                i += 2 # also skip the next seg, which would also
                       # intersect at that seg's start point
            elif close(point, seg.coords[0], tol):
                # intersects at the near point
                if i is not 0:
                    assert(len(segcoords) > 1)
                    segs.append(shapely.geometry.LineString(segcoords[:-1]+[point,]))
                    segcoords = [point,]
                else:
                    assert(len(segcoords) is 1)
                    segcoords[0] = point
                segcoords.append(seg.coords[-1])
                i += 1
            else:
                # intersects in the middle
                segs.append(shapely.geometry.LineString(segcoords+[point,]))
                #logging.debug("  appended partial segment: %r"%(list(segs[-1].coords)))
                segcoords = [point,seg.coords[-1]]
                i += 1
        else:
            raise RuntimeError("Dual/multiple intersection in a single seg... ugh!")

    if len(segcoords) > 1:
        segs.append(shapely.geometry.LineString(segcoords))
    return segs

def distance(p1, p2):
    """Distance between two points in tuple form"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def in_neighborhood(obj1, obj2, tol=0.1):
    """Determines if two objects can possibly intersect by performing a
    quick check of their bounding boxes.
    """
    minx1,miny1,maxx1,maxy1 = obj1.bounds
    minx2,miny2,maxx2,maxy2 = obj2.bounds
    if maxx2 < minx1 - tol or \
       maxy2 < miny1 - tol or \
       minx2 > maxx1 + tol or \
       miny2 > maxy1 + tol:
        return False
    return True


def intersect_point_to_segment(point, line_start, line_end):
    """Finds the nearest point on a line segment to a point"""
    line_magnitude = line_end.distance(line_start)
    assert(line_magnitude > _tol)    
    u = ((point.x - line_start.x) * (line_end.x - line_start.x) +
         (point.y - line_start.y) * (line_end.y - line_start.y)) \
         / (line_magnitude ** 2)

    # closest point does not fall within the line segment, 
    # take the shorter distance to an endpoint
    if u < 0.:
        return line_start
    elif u > 1.:
        return line_end
    else:
        ix = line_start.x + u * (line_end.x - line_start.x)
        iy = line_start.y + u * (line_end.y - line_start.y)
        return shapely.geometry.Point([ix, iy])

def nearest_point(line, point):
    """Returns the nearest coordinate on the line to point.  

    Note point is expected as coordinates."""
    return line.interpolate(line.project(shapely.geometry.Point(point))).coords[0]
    
def find_perp(line, point):
    # need another point, perpendicular to the line, to intersect
    k = line.project(shapely.geometry.Point(point))
    if k < 0.001:
        k2 = 0.001
    else:
        k2 = k - 0.001
    p2 = line.interpolate(k2).coords[0]
    dp = (p2[0] - point[0], p2[1] - point[1])
    p3 = (point[0] - dp[1], point[1] - dp[0])
    return p3
    
def triangle_area(vertices):
    """Area of a triangle in 2D"""
    xy1 = vertices[0]
    xy2 = vertices[1]
    xy3 = vertices[2]

    A = 0.5 * (xy2[0] * xy3[1] -
            xy3[0] * xy2[1] -
            xy1[0] * xy3[1] +
            xy3[0] * xy1[1] +
            xy1[0] * xy2[1] -
            xy2[0] * xy1[1])
    return A


def center(objects):
    """Centers a collection of objects by removing their collective centroid"""
    union = shapely.ops.cascaded_union(objects)
    centroid = union.centroid
    new_objs = [shapely.affinity.translate(obj, -centroid.coords[0][0], -centroid.coords[0][1]) for obj in objects]
    return new_objs, centroid
    
def get_git_revision_hash():
    """Returns the git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')


def merge(ml1, ml2):
    """Merges two multilines that are assumed to overlap except at their endpoints."""
    assert(len(ml1) > 1)
    assert(len(ml2) > 1)
    assert(close(ml1[0].coords[0], ml2[0].coords[0]))
    assert(close(ml1[-1].coords[-1], ml2[-1].coords[-1]))

    new_ml = []
    i1 = 0
    c1 = ml1[i1]
    i2 = 0
    c2 = ml2[i2]
    done = False
    tol = 1.e-5
    while not done:
        if close(c1,c2,tol):
            new_ml.append(c1)
            i1 += 1
            if i1 < len(ml1):
                c1 = ml1[i1]
            else:
                c1 = None
                done = True
            
            i2 += 1
            if i2 < len(ml2):
                c2 = ml2[i2]
            else:
                c2 = None
                done = True
            
        elif c1.length < c2.length:
            for k in range(len(c1.coords)-1):
                assert(close(c1.coords[k], c2.coords[k], tol))
            new_ml.append(c1)

            assert(not close(c1.coords[-1], c2.coords[len(c1.coords)-1], tol))
            c2 = shapely.geometry.LineString([c1.coords[-1],]+c2.coords[len(c1.coords)-1:])

            i1 += 1
            if i1 < len(ml1):
                c1 = ml1[i1]
            else:
                c1 = None
                done = True

        elif c2.length < c1.length:
            for k in range(len(c2.coords)-1):
                assert(close(c2.coords[k], c1.coords[k], tol))
            new_ml.append(c2)

            assert(not close(c2.coords[-1], c1.coords[len(c2.coords)-1], tol))
            c1 = shapely.geometry.LineString([c2.coords[-1],]+c1.coords[len(c2.coords)-1:])

            i2 += 1
            if i2 < len(ml2):
                c2 = ml2[i2]
            else:
                c2 = None
                done = True

        else:
            raise RuntimeError("ruh roh rorge")

    if c1 is not None:
        new_ml.append(c1)
    if c2 is not None:
        new_ml.append(c2)
    return new_ml

        
        
def non_point_intersection(shp1, shp2):
    inter = shp1.intersection(shp2)
    int_type = type(inter)
    if int_type == shapely.geometry.Point:
        return False
    elif int_type == shapely.geometry.GeometryCollection and len(inter) is 0:
        return False
    return True
