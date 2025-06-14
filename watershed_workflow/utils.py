"""Shape and geometry utilities for working with fiona and shapely objects.

Note this module contains a lot of other simple functions that are commonly
used by other functions, but are not included in documentation because they are
likely not useful to users.

"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from typing import Iterable, Tuple, List, Any, Optional, Literal
    import numpy.typing as npt
    import geopandas as gpd
    import xarray.core.types
    import matplotlib.axes

    from watershed_workflow.crs import CRS
    
import datetime, cftime
import logging
import numpy as np
import math
import scipy.interpolate
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.prepared
import shapely.affinity
import xarray

import watershed_workflow.warp

_tol = 1.e-7

#
# Geometric utilities
#
def computeDistance(p1 : Tuple[float,float],
                    p2 : Tuple[float,float]) -> float:
    """Distance between two points in tuple form"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def computeTriangleArea(p1 : Tuple[float,float],
                        p2 : Tuple[float,float],
                        p3 : Tuple[float,float]) -> float:
    """Area of a triangle in 2D"""
    return 0.5 * (p2[0] * p3[1] - p3[0] * p2[1] - p1[0] * p3[1] + p3[0] * p1[1]
               + p1[0] * p2[1] - p2[0] * p1[1])

def computeTriangleCentroid(p1 : Tuple[float,float],
                            p2 : Tuple[float,float],
                            p3 : Tuple[float,float]) -> float:
    """Centroid of a triangle in 2D"""
    return np.array([p1,p2,p3]).mean(axis=0)


def isCollinear(p1 : Tuple[float,float],
                p2 : Tuple[float,float],
                p3 : Tuple[float,float],
                tol : float = 1e-6) -> bool:
    """this function checks if three points are collinear for given tolerance value"""
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1*y2 - x2*y1) < tol


def computeArea(vertices : Iterable[Tuple[float]]) -> float:
    """Area of polygons in 2D"""
    area = shapely.geometry.Polygon(vertices).area
    return area


def computeAngle(v1 : Tuple[float, float] | shapely.geometry.LineString,
                 v2 : Tuple[float, float] | shapely.geometry.LineString) -> float:
    """Given two 2D vectors represented as len 2 arrays or tuples,
    find the angle (in degrees) of 2 relative to 1 in a clockwise
    notion.

    """
    if isinstance(v1, shapely.geometry.LineString):
        if isinstance(v2, shapely.geometry.LineString):
            assert isClose(v1.coords[0], v2.coords[-1])
        
        c1 = np.array(v1.coords[0:2])[:,0:2]
        return computeAngle(c1[1] - c1[0], v2)
    if isinstance(v2, shapely.geometry.LineString):
        c2 = np.array(v2.coords[-2:])[:,0:2]
        return computeAngle(v1, c2[0] - c2[1])

    x1, y1 = v1
    x2, y2 = v2

    # Compute the angle of each vector with respect to the positive x-axis
    angle1 = math.atan2(y1, x1)  # Angle of vec1
    angle2 = math.atan2(y2, x2)  # Angle of vec2

    # Compute the difference in angles, clockwise
    delta_angle = angle1 - angle2

    # Convert to clockwise angle in degrees
    clockwise_angle = math.degrees(delta_angle)

    # Normalize the angle to be between 0 and 360 degrees
    if clockwise_angle < 0:
        clockwise_angle += 360

    return clockwise_angle    

    

def projectVectorAtAngle(v1 : Tuple[float,float] | shapely.geometry.LineString,
                        angle : float,
                        distance : float) -> Tuple[float,float]:
    """Given a vector v1 (or one that can be computed from a
    downstream-oriented linestring), find the vector v2 such that
    ||v2|| == distance and computeAngle(v1,v2) == angle.

    angle is in degrees

    """
    if isinstance(v1, shapely.geometry.LineString):
        c1 = np.array(v1.coords[0:2])
        return projectVectorAtAngle(c1[1] - c1[0], angle, distance)

    x1, y1 = v1

    # Compute the angle of vec1 with respect to the positive x-axis
    angle1 = math.atan2(y1, x1)

    # Convert the clockwise angle to radians
    angle_offset = math.radians(angle)

    # Compute the resulting angle of vec2
    angle2 = angle1 - angle_offset

    # Compute the components of vec2 using the distance and angle2
    x2 = distance * math.cos(angle2)
    y2 = distance * math.sin(angle2)
    return (x2, y2)    

    
def computeMidpoint(p1 : Tuple[float,float],
                    p2 : Tuple[float,float]) -> Tuple[float,float]:

    """Returns the midpoint of two points"""
    return ((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)


def findClosestPointIndex(point : Tuple[float, float],
                          points : npt.ArrayLike) -> int:
    """Returns the index of closest point from an array of points"""
    points2 = np.asarray(points)
    point2 = np.array(point)
    dist = np.sum((points2 - point2)**2, axis=1)
    return int(np.argmin(dist))


def cluster(points : np.ndarray,
            tol : float) -> Tuple[np.ndarray, np.ndarray]:
    """Given a list of points, determine a list of clusters.

    Each cluster is within tol of each other.

    Returns (cluster_index, cluster_centroid)
    """
    import scipy.cluster.hierarchy as hcluster
    if type(points) is list:
        points = np.array(points)

    if len(points) > 1:
        indices = hcluster.fclusterdata(points, tol, criterion='distance')
        centroids = np.array([points[indices == (i + 1)].mean(axis=0) for i in range(indices.max())])
    else:
        indices = np.array([1,]*len(points))
        centroids = points
    return indices - 1, centroids


#
# Shape utilities
#

def flatten(list_of_shps : Any) -> List[BaseGeometry]:
    """Flattens a list of shapes, that may contain Multi-objects, into a list without multi-objects"""
    new_list = []
    for shp in list_of_shps:
        if isinstance(shp, shapely.geometry.MultiLineString) or \
           isinstance(shp, shapely.geometry.MultiPoint) or \
           isinstance(shp, shapely.geometry.MultiPolygon):
            new_list.extend(list(shp.geoms))
        else:
            new_list.append(shp)
    return new_list


def recenter(objects : Iterable[BaseGeometry],
             centering : bool = True) -> Tuple[Iterable[BaseGeometry], shapely.geometry.Point]:
    """Centers a collection of objects by removing their collective centroid"""
    if type(centering) is shapely.geometry.Point:
        centroid = centering
    elif centering is True or centering == 'geometric':
        union = shapely.ops.unary_union(objects)
        centroid = shapely.geometry.Point([(union.bounds[0] + union.bounds[2]) / 2.,
                                           (union.bounds[1] + union.bounds[3]) / 2.])
    elif centering == 'mass':
        union = shapely.ops.unary_union(objects)
        centroid = union.centroid
    else:
        raise ValueError('Centering: option centering = "{}" unknown'.format(centering))

    new_objs = [
        shapely.affinity.translate(obj, -centroid.coords[0][0], -centroid.coords[0][1])
        for obj in objects
    ]

    for new, old in zip(new_objs, objects):
        if hasattr(old, 'properties'):
            new.properties = old.properties

    return new_objs, centroid


def intersects(shp1 : BaseGeometry,
               shp2 : BaseGeometry) -> bool:
    """Checks whether an intersection exists.
    
    Note that intersection being empty and intersects are not always reliably
    the same... we avoid using shapely.intersects() for this reason.
    """
    inter = shp1.intersection(shp2)
    return not isEmpty(inter)


def isNonPointIntersection(shp1 : BaseGeometry,
                           shp2 : BaseGeometry) -> bool:
    """Checks whether an intersection is larger than a point.
    
    Note that intersection being empty and intersects are not always reliably
    the same... we avoid using intersects() for this reason.
    """
    inter = shp1.intersection(shp2)
    return not (isEmpty(inter) or \
                isinstance(inter, shapely.geometry.Point))


def isVolumetricIntersection(shp1 : BaseGeometry,
                             shp2 : BaseGeometry) -> bool:
    """Checks whether an intersection includes volume and not just points and lines."""
    inter = shp1.intersection(shp2)
    return inter.area > 0


def filterToShape(df : gpd.GeoDataFrame,
                  shape : BaseGeometry,
                  shape_crs : CRS,
                  method : str = 'contains',
                  tol : Optional[float] = None) -> gpd.GeoDataFrame:
    """Filters out reaches (or reaches in rivers) not inside the HUCs provided.

    method is one of 'contains' or 'intersects' to indicate whether
    to include things entirely in shape or partially in shape,
    respectively.
    """
    if shape_crs != df.crs:
        shape = watershed_workflow.warp.shply(shape, shape_crs, df.crs)

    if tol is None: tol = _tol
    if method == 'contains':
        shape = shapely.prepared.prep(shape.buffer(2 * tol))
        op = shape.contains
    elif method == 'intersects':
        shape = shapely.prepared.prep(shape.buffer(2 * tol))
        op = shape.intersects
    elif method == 'non_point_intersection':
        op = lambda a: isNonPointIntersection(shape, a)
    else:
        raise ValueError("method must be one of 'intersects', 'contains', or 'non_point_intersection'")

    return df[df.geometry.apply(op)]


def isEmpty(shply : BaseGeometry | None) -> bool:
    return shply is None or shply.is_empty


def isConvex(points : Iterable[Tuple[float,float]]) -> bool:
    poly = shapely.geometry.Polygon(points)
    return math.isclose(poly.area, poly.convex_hull.area, rel_tol=1e-4)


def breakLineStringCollinearity(linestring_coords : np.ndarray,
                             tol : float = 1e-5) -> np.ndarray:
    """This functions removes collinearity from a node linestring by making small pertubations orthogonal to the linestring"""
    # traversing along the linestring, checking 3 consecutive points at a time
    for i, (p0, p1, p2) in enumerate(zip(linestring_coords, linestring_coords[1:], linestring_coords[2:])):
        # treating collinearity through a small pertubation
        if isCollinear(p0, p1, p2, tol=tol):
            dp = p2 - p0
            ortho = 10 * tol * np.array([-dp[1], dp[0]]) / np.linalg.norm(dp)

            linestring_coords[i + 1] = p1 + ortho
    return linestring_coords


def reverseLineString(ls):
    return shapely.geometry.LineString(reversed(ls.coords))


def isClose(s1 : BaseGeometry,
            s2 : BaseGeometry,
            tol : float = _tol) -> bool:
    """Are two shapely shapes topologically equivalent and geometrically close?

    Note this deals with things like rotations of polygons (clock-rotating the
    coordinates of the same shape are still close) and other gotchas that keep
    you from just comparing coordinates.
    """
    # deal with Multi* or list objects
    def is_multi(thing):
        if isinstance(thing, shapely.geometry.MultiPoint):
            return True
        if isinstance(thing, shapely.geometry.MultiLineString):
            return True
        if isinstance(thing, shapely.geometry.MultiPolygon):
            return True
        if isinstance(thing, list):
            return True
        return False

    def local_len(thing):
        try:
            return len(thing)
        except AttributeError:
            return len(thing.geoms)

    def iter(thing):
        assert (is_multi(thing))
        if isinstance(thing, list):
            for t in thing:
                yield t
        else:
            for t in thing.geoms:
                yield t

    if is_multi(s1):
        if local_len(s1) == 1:
            return isClose(next(iter(s1)), s2, tol)

    if is_multi(s2):
        if local_len(s2) == 1:
            return isClose(s1, next(iter(s2)), tol)

    if is_multi(s1) and is_multi(s2):
        if local_len(s1) != local_len(s2):
            return False
        return all(isClose(i1, i2, tol) for (i1, i2) in zip(s1, s2))

    # points get compared as tuples
    if isinstance(s1, shapely.geometry.Point):
        return isClose(s1.coords[0], s2, tol)
    elif isinstance(s1, np.ndarray) and len(s1.shape) == 1:
        return isClose((s1[0], s1[1]), s2)

    if isinstance(s2, shapely.geometry.Point):
        return isClose(s1, s2.coords[0], tol)
    elif isinstance(s2, np.ndarray) and len(s2.shape) == 1:
        return isClose(s1, (s2[0], s2[1]))

    # types should be the same now
    if type(s1) != type(s2):
        return False

    # compare tuples
    if isinstance(s1, tuple):
        if len(s1) != len(s2):
            return False
        return sum((p1 - p2)**2 for p1, p2 in zip(s1, s2)) < tol**2

    # compare lines
    elif isinstance(s1, shapely.geometry.LineString):
        if len(s1.coords) != len(s2.coords):
            return False
        if np.allclose(np.array(s1.coords), np.array(s2.coords), tol, tol):
            return True
        if np.allclose(np.array(s1.coords), np.array(list(reversed(s2.coords))), tol, tol):
            return True
        return False

    # compare polygons
    elif type(s1) is shapely.geometry.Polygon:
        # note, this does not correctly deal with nonequal holes...
        if len(s1.boundary.coords) != len(s2.boundary.coords):
            return False
        ls1 = s1.boundary.coords[:-1]
        ls2 = np.array(s2.boundary.coords[:-1])
        ls2f = np.flipud(ls2)
        return any(np.allclose(ls1, np.roll(ls2, i, 0), tol, tol) for i in range(len(ls2))) or \
            any(np.allclose(ls1, np.roll(ls2f, i, 0), tol, tol) for i in range(len(ls2)))

    # compare multi-shapes by checking if each one has a match in the
    # other and lengths are the same
    elif isinstance(s1, (shapely.geometry.MultiPoint, shapely.geometry.MultiLineString,
                         shapely.geometry.MultiPolygon)):
        if len(s1) != len(s2):
            return False
        good2 = [False, ] * len(s2)
        for l1 in s1:
            found = False
            for i, l2 in enumerate(s2):
                if not good2[i]:
                    if isClose(l1, l2, tol):
                        good2[i] = True
                        found = True
                        break
            if not found:
                return False
        return True
    else:
        raise NotImplementedError("Not implemented for type '%r'" % type(s1))


def contains(s1 : BaseGeometry,
             s2 : BaseGeometry,
             tol : float = _tol) -> bool:
    """A contains algorithm that deals with close/roundoff issues"""
    return s1.buffer(tol, 2).contains(s2)


class CutError(Exception):
    def __init__(self, message, line, seg, cutline):
        super(Exception, self).__init__(message)
        self.line = line
        self.seg = seg
        self.cutline = cutline


def cut(line1 : shapely.geometry.LineString,
        line2 : shapely.geometry.LineString) -> Tuple[List[shapely.geometry.LineString],
                                                      List[shapely.geometry.LineString]]:
    """Cuts two linestrings at their (one) intersection point."""
    return list(shapely.ops.split(line1, line2).geoms), \
        list(shapely.ops.split(line2, line1).geoms)

        
# def cut(line : shapely.geometry.LineString,
#         cutline : shapely.geometry.LineString,
#         tol : float = 1.e-5) -> List[shapely.geometry.LineString]:
#     def plot():
#         from matplotlib import pyplot as plt
#         plt.plot(cutline.xy[0], cutline.xy[1], 'k-x', linewidth=3)
#         plt.plot(line.xy[0], line.xy[1], 'g-+', linewidth=3)

#     assert type(line) is shapely.geometry.LineString
#     assert type(cutline) is shapely.geometry.LineString
#     assert line.intersects(cutline)

#     segs = []
#     coords = list(line.coords)

#     segcoords = [coords[0], ]
#     i = 0
#     while i < len(coords) - 1:
#         seg = shapely.geometry.LineString(coords[i:i + 2])
#         #logging.debug("Intersecting seg %d"%i)
#         point = seg.intersection(cutline)
#         if type(point) is shapely.geometry.LineString and len(point.coords) == 0:
#             #logging.debug("Cut seg no intersection")
#             segcoords.append(seg.coords[-1])
#             i += 1
#         elif type(point) is shapely.geometry.Point:
#             #logging.debug("Cut intersected at point")
#             #logging.debug("  inter point: %r"%list(point.coords[0]))
#             #logging.debug("  seg final point: %r"%list(seg.coords[-1]))
#             #logging.debug("  close? = %r"%(isClose(point, seg.coords[-1], tol)))
#             if isClose(point, seg.coords[-1], tol):
#                 # intersects at the far point
#                 segs.append(shapely.geometry.LineString(segcoords + [point, ]))
#                 #logging.debug("  appended full linestring: %r"%(list(segs[-1].coords)))

#                 if (i < len(coords) - 2):
#                     #logging.debug("    (not the end)")
#                     segcoords = [point, coords[i + 2]]
#                 else:
#                     #logging.debug("    (the end)")
#                     segcoords = [point, ]
#                 i += 2  # also skip the next seg, which would also
#                 # intersect at that seg's start point
#             elif isClose(point, seg.coords[0], tol):
#                 # intersects at the near point
#                 if i != 0:
#                     assert (len(segcoords) > 1)
#                     segs.append(shapely.geometry.LineString(segcoords[:-1] + [point, ]))
#                     segcoords = [point, ]
#                 else:
#                     assert (len(segcoords) == 1)
#                     segcoords[0] = point
#                 segcoords.append(seg.coords[-1])
#                 i += 1
#             else:
#                 # intersects in the middle
#                 segs.append(shapely.geometry.LineString(segcoords + [point, ]))
#                 #logging.debug("  appended partial linestring: %r"%(list(segs[-1].coords)))
#                 segcoords = [point, seg.coords[-1]]
#                 i += 1
#         else:
#             print("Dual/multiple section: type = {}".format(type(point)))
#             print(" point = {}".format(point))
#             raise CutError(
#                 "Dual/multiple intersection in a single seg... ugh!  "
#                 + "Intersection is of type '{}'".format(type(point)), line, seg, cutline)

#     if len(segcoords) > 1:
#         segs.append(shapely.geometry.LineString(segcoords))
#     return segs


def inNeighborhood(shp1 : BaseGeometry,
                   shp2 : BaseGeometry,
                   tol : float = 0.1):
    """Determines if two shapes can possibly intersect by performing a
    quick check of their bounding boxes.
    """
    minx1, miny1, maxx1, maxy1 = shp1.bounds
    minx2, miny2, maxx2, maxy2 = shp2.bounds
    if maxx2 < minx1 - tol or \
       maxy2 < miny1 - tol or \
       minx2 > maxx1 + tol or \
       miny2 > maxy1 + tol:
        return False
    return True


def intersectPointToSegment(point : shapely.geometry.Point,
                            seg_start : shapely.geometry.Point,
                            seg_end : shapely.geometry.Point) -> shapely.geometry.Point:
    """Finds the nearest point on a line linestring to a point"""
    seg_magnitude = seg_end.distance(seg_start)
    assert (seg_magnitude > _tol)
    u = ((point.x - seg_start.x) * (seg_end.x - seg_start.x) +
         (point.y - seg_start.y) * (seg_end.y - seg_start.y)) \
         / (seg_magnitude ** 2)

    # closest point does not fall within the line linestring,
    # take the shorter distance to an endpoint
    if u < 0.:
        return seg_start
    elif u > 1.:
        return seg_end
    else:
        ix = seg_start.x + u * (seg_end.x - seg_start.x)
        iy = seg_start.y + u * (seg_end.y - seg_start.y)
        return shapely.geometry.Point([ix, iy])


def findNearestPoint(point : shapely.geometry.Point,
                     line : shapely.geometry.LineString,
                     tol : Optional[float] = None) -> shapely.geometry.Point | None:
    """Returns the nearest coordinate on the line to point.  

    Note point is expected as coordinates."""
    if tol is None:
        if isinstance(point, tuple):
            point = shapely.geometry.Point(point)
        return shapely.ops.nearest_points(point, line)[1].coords[0]
    else:
        if inNeighborhood(shapely.geometry.Point(point), line, tol):
            logging.debug("  - in neighborhood")
            nearest_p = findNearestPoint(point, line)
            assert nearest_p is not None
            dist = computeDistance(nearest_p, point)
            logging.debug("  - nearest p = {0}, dist = {1}, tol = {2}".format(nearest_p, dist, tol))
            if dist < tol:
                return nearest_p
        return None


def removeThirdDimension(geom : shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Removes the third dimension of a shapely object."""
    def _drop_z(*args):
        return tuple(filter(None, [args[0],args[1]]))
    return shapely.ops.transform(_drop_z, geom)


def computeSegmentLengths(ls : shapely.geometry.LineString) -> np.ndarray:
    """Computes the incremental segment length between each coord of ls."""
    coords = np.array(ls.coords)
    return np.linalg.norm((coords[1:] - coords[:-1]), axis=1)


def computeArclengths(ls : shapely.geometry.LineString) -> np.ndarray:
    """Computes the arclength of each coord in ls."""
    ds = computeSegmentLengths(ls)
    return np.cumsum(np.concatenate([[0.0,], ds]))


def logMinMaxMedianSegment(iterable : Iterable[shapely.geometry.LineString],
                           name : str,
                           assert_on_zero : bool = False,
                           ax : Optional[matplotlib.axes.Axes] = None,
                           color : Optional[str | Tuple] = None) -> None:
    """Computes min, median, and max segment length across all linestrings."""
    seg_mins : List[float] = []
    seg_maxs : List[float] = []
    seg_meds : List[float] = []
    seg_lens : List[np.ndarray] = []
    geom_lens : List[float] = []
    
    for ls in iterable:
        geom_lens.append(ls.length)
        seg_len = computeSegmentLengths(ls)
        seg_lens.append(seg_len)
        if assert_on_zero:
            assert(min(seg_len) > 1.e-10)

    seg_lens_a = np.concatenate(seg_lens)
    seg_diags = np.min(seg_lens_a), np.median(seg_lens_a), np.max(seg_lens_a)

    geom_lens_a = np.array(geom_lens)
    geom_diags = (min(geom_lens_a), np.median(geom_lens_a), max(geom_lens_a))

    if ax is not None:
        ax.hist(seg_lens_a, max(len(seg_lens_a)//20, 10), color=color, label=name)
        ax.set_xlabel('segment length')
        ax.set_ylabel('count')
        ax.legend()
                  
    logging.info(f"  {name}: min seg length: \t{seg_diags[0]:16.10f} \tmin geom length: \t{geom_diags[0]:16.10f}")
    logging.info(f"  {name}: med seg length: \t{seg_diags[1]:16.10f} \tmed geom length: \t{geom_diags[1]:16.10f}")
    logging.info(f"  {name}: max seg length: \t{seg_diags[2]:16.10f} \tmax geom length: \t{geom_diags[2]:16.10f}")
    logging.info('')
    return


#
# Dataset utilities
#


#
# fiona utilities -- probably need to go away?
#
def generateRings(obj):
    """Generator for a fiona shape's coordinates object and yield rings.

    As long as the input is conforming, the type of the geometry doesn't matter.

    Parameter
    ---------
    obj : fiona shape

    Returns
    -------
    rings : iterator
      Iterates over rings, each of which is a list of coordinate tuples.    
    """
    def _generateRings(coords):
        for e in coords:
            if isinstance(e[0], (float, int)):
                yield coords
                break
            else:
                for r in _generateRings(e):
                    yield r

    if 'geometry' in obj:
        obj = obj['geometry']
    for r in _generateRings(obj['coordinates']):
        yield r


def generateCoords(obj):
    """Generator for a fiona geometry's coordinates.

    As long as the input is conforming, the type of the geometry doesn't
    matter.

    Parameter
    ---------
    obj : fiona shape

    Returns
    -------
    coord : iterator
      Iterates over coordinate tuples.
    """
    if 'geometry' in obj:
        obj = obj['geometry']

    if obj['type'] == 'Point':
        yield obj['coordinates']
    else:
        for ring in generateRings(obj):
            for c in ring:
                yield c
