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
    from watershed_workflow.crs import CRS
    import xarray.core.types

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
        c1 = np.array(v1.coords[0:2])
        return computeAngle(c1[1] - c1[0], v2)
    if isinstance(v2, shapely.geometry.LineString):
        c2 = np.array(v2.coords[-2:])
        return computeAngle(v1, c2[0] - c2[1])

    x1, y1 = v1
    x2, y2 = v2

    # Compute the angle of each vector with respect to the positive x-axis
    angle1 = math.atan2(y1, x1)  # Angle of vec1
    angle2 = math.atan2(y2, x2)  # Angle of vec2
    print(angle1, angle2)

    # Compute the difference in angles, clockwise
    delta_angle = angle1 - angle2

    # Convert to clockwise angle in degrees
    clockwise_angle = math.degrees(delta_angle)

    # Normalize the angle to be between 0 and 360 degrees
    if clockwise_angle < 0:
        clockwise_angle += 360

    print(v1, v2, clockwise_angle)
        
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
    if isinstance(s2, shapely.geometry.Point):
        return isClose(s1, s2.coords[0], tol)

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
                if dist < 1.e-7:
                    # filter case where the point is already there
                    if any(isClose(point, c) for c in line.coords):
                        return None
                return nearest_p
        return None


def removeThirdDimension(geom : shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Removes the third dimension of a shapely object."""
    def _drop_z(*args):
        return tuple(filter(None, [args[0],args[1]]))
    return shapely.ops.transform(_drop_z, geom)


def computeSegmentLengths(ls : shapely.geometry.LineString) -> np.array:
    coords = np.array(ls.coords)
    return np.linalg.norm((coords[1:] - coords[:-1]), axis=1)


def logMinMaxMedianSegment(iterable : Iterable[shapely.geometry.LineString],
                           name : str,
                           assert_on_zero : bool = False,
                           ax : Optional[matplotlib.Axes] = None,
                           color : Optional[str | Tuple] = None) -> None:
    """Computes min, median, and max segment length across all linestrings."""
    seg_mins = []
    seg_maxs = []
    seg_meds = []
    seg_lens = []
    geom_lens = []
    
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
def valuesFromArray(points : np.ndarray,
                    points_crs : CRS,
                    array : xarray.DataArray,
                    method : xarray.core.types.InterpOptions = 'nearest'):
    """For each point in points, interpolates a value from the array.

    Valid methods include:
    - "nearest" for nearest neighbor
    - "linear" for bilinear interpolation

    """
    array_crs = watershed_workflow.crs.from_xarray(array)
    xy = watershed_workflow.warp.xy(points[:,0], points[:,1], points_crs, array_crs)
    x = xarray.DataArray(points, dims='points')
    y = xarray.DataArray(points, dims='points')
    return array.interp(x=x, y=y, method=method).values


def imputeHoles2D(arr : np.ndarray,
                  nodata : Any = np.nan,
                  method : str = 'cubic') -> np.ndarray:
    """Very simple imputation algorithm to interpolate values for missing data in rasters.

    Note, this may throw if there is a hole on the boundary?

    Note, valid 'method' values are those of scipy.interpolate.griddata.
    """
    if nodata is np.nan:
        mask = np.isnan(arr)
    else:
        mask = (arr == nodata)

    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])
    xx, yy = np.meshgrid(x, y)

    #get only the valid values
    x1 = xx[~mask]
    y1 = yy[~mask]
    newarr = arr[~mask]

    res = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method)
    return res


def smoothRaster(img_in : np.ndarray,
                method : str = 'gaussian',
                **kwargs) -> np.ndarray:
    """Smooths an image according to an algorithm, passing kwargs on to that algorithm."""
    if method == 'gaussian':
        if 'method' not in kwargs:
            kwargs['method'] = 'nearest'
        if 'sigma' not in kwargs:
            sigma = 5
        else:
            sigma = kwargs.pop('sigma')
        return scipy.ndimage.gaussian_filter(img_in, sigma, **kwargs)
    else:
        raise ValueError(f'Unknown smoothing method: "{method}"')


def computeAverageYear(data,
                       output_nyears : int = 1,
                       smooth : bool = False,
                       **kwargs):
    """Averages and smooths to form a "typical" year.

    Parameters
    ----------
    data : np.ndarray(shape=(NTIMES, ...))
      Daily array to average, note that NTIMES > 365.
    output_nyears : int, optional
      Number of years to repeat the output.  Default is 1.
    filter : bool, optional
      If true, filters the data using a Sav-Gol filter from Scipy

    All other parameters are passed to the filter.  See
    scipy.signal.savgol_filter, but sane default values are used if
    these are not provided.  Note that if NTIMES % 365 != 0, the data
    is truncated.

    Returns
    -------
    np.ndarray(shape=(365*output_nyears, ...))
      The averaged data.

    """
    nyears = data.shape[0] // 365
    if nyears == 0:
        raise ValueError('Not enough data to compute average year. Need at least 365 days.')

    # reshape the data to (nyears, 365, ...)
    data = np.array(data[0:nyears * 365][:])
    original_shape = data.shape
    new_shape = (nyears, 365)
    if len(original_shape) > 1:
        new_shape = new_shape + original_shape[1:]
    data = data.reshape(new_shape)
    data = data.mean(axis=0)

    # smooth if requested
    if smooth:
        data = smoothArray(data, smooth, axis=0, **kwargs)

    # repeat the data if requested
    if output_nyears != 1:
        tiled_data_shape = (output_nyears, )
        for i in range(len(original_shape) - 1):
            tiled_data_shape = tiled_data_shape + (1, ) # type: ignore
        data = np.tile(data, tiled_data_shape)
    return data


def interpolateInTimeRegular(times, data, start, end,
                                dt=datetime.timedelta(days=1),
                                axis=0,
                                **kwargs):
    """Interpolate time-dependent data to a regularly spaced time array.

    Parameters
    ----------
    times : np.1darray(dtype=cftime.datetime)
      An array of times, of length NTIMES.
    data : np.ndarray
      Data to interpolate, data.shape[axis] == NTIMES.
    start, end : cftime.datetime
      Times to begin and end (inclusive) the interpolated array.
    dt : datetime.timedelta, optional
      Delta to interpolate to.  Defaults to 1 day.
    axis : int, optional
      Axis of data that corresponds to time.  Default is 0.
    
    All other parameters are passed to scipy.interpolate.interp1d.  Of
    use particularly is 'kind' which can be 'linear' (default) or
    'quadratic', 'cubic' or others.

    Returns
    -------
    new_times : np.1darray(dtype=datetime.date)
      Times of the new array.
    new_data : np.ndarray
      The data interpolated.

    """
    if data.shape[axis] != len(times):
        raise ValuerError("Data and times array are not of the expected shape.")

    # new_times for interpolation
    new_count = int(np.ceil((end-start) / dt))
    new_times = np.array([start + i*dt for i in range(new_count + 1)])

    # interpolate onto new_times
    new_data = interpolate_in_time(times, data, new_times, axis, **kwargs)
    return new_times, new_data


def interpolateInTime(times, data, new_times, axis=0, units="days since 2000", **kwargs):
    """Interpolate time-dependent data to an arbitrary other time array.

    Parameters
    ----------
    times : np.1darray(dtype=cftime.datetime)
      An array of times, of length NTIMES.
    data : np.ndarray
      Data to interpolate, data.shape[axis] == NTIMES.
    new_times : np.1darray(dtype=cftime.datetime)
      An array of times to interpolate to.
    axis : int, optional
      Axis of data that corresponds to time.  Default is 0.
    units : str, optional
      Interpolation must happen in a numeric coordinate -- this unit
      is used to convert from dates to numbers using
      cftime.date2num. Valid cfunits for time are strings like "days since
      2000-1-1", which is the default.
    
    All other parameters are passed to scipy.interpolate.interp1d.  Of
    use particularly is 'kind' which can be 'linear' (default) or
    'quadratic', 'cubic' or others.

    Returns
    -------
    new_data : np.ndarray
      The data interpolated.
    """
    if data.shape[axis] != len(times):
        raise ValueError("Data and times array are not of the expected shape.")

    if times[0].calendar != new_times[0].calendar:
        raise ValueError("times and new_times must have the same calendar.")

    # create an interpolator in a modified coordinate system
    x = cftime.date2num(times, units)
    interp = scipy.interpolate.interp1d(x, data, axis=axis, assume_sorted=True, **kwargs)

    # interpolate at new_times in the modified coordinate system
    new_x = cftime.date2num(new_times, units)
    new_data = interp(new_x)
    return new_data


def smoothArray(data, method, axis=0, **kwargs):
    """Smooths fixed-interval time-series data using a Sav-Gol filter from scipy.

    Note that this wrapper just sets some sane default values for
    daily data -- one could just as easily call
    scipy.signal.savgol_filter themselves.
    
    Parameters
    ----------
    data : np.ndarray
      The data to smooth.
    window_length : int, optional
      Length of the moving window over which to fit the polynomial.
      Default is 61.
    polyorder : int, optional
      Order of the fitting polynomial. Default is 2.
    axis : int, optional
      Time axis over which to smooth. Default is 0.
    mode : str, optional
      See scipy.signal.savgol_filter documentation, but 'wrap' is the
      best bet for data in multiples of years. Default is 'wrap.'

    Any additional kwargs are passed to scipy.signal.savgol_filter

    Returns
    -------
    np.ndarray
      Smoothed data in the same shape as data

    """
    if method is True:
        method = 'savgol_filter'

    if method == 'savgol_filter':
        if 'window_length' not in kwargs:
            kwargs['window_length'] = 61
        if 'polyorder' not in kwargs:
            kwargs['polyorder'] = 2
        if 'mode' not in kwargs:
            kwargs['mode'] = 'wrap'
        return scipy.signal.savgol_filter(data, axis=axis, **kwargs)
    elif method == 'convolve':
        if 'window' not in kwargs:
            kwargs['window'] = 'hann'
        if 'Nx' not in kwargs:
            kwargs['Nx'] = 50

        win = scipy.signal.windows.get_window(**kwargs)
        win = win / win.sum()
        assert (len(data.shape) == 3 and axis == 0)
        data_new = np.empty_like(data)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                data_new[:, i, j] = scipy.signal.convolve(data[:, i, j],
                                                          win)[len(win) // 2:-len(win) // 2 + 1]
        return data_new

    else:
        raise ValueError(f'Invalid smooth method {method}')

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
