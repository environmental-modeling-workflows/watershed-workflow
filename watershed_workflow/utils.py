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
def computeDistance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points.
    
    Parameters
    ----------
    p1 : tuple of float
        First point as (x, y) coordinates.
    p2 : tuple of float
        Second point as (x, y) coordinates.
        
    Returns
    -------
    float
        Euclidean distance between the two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def computeTriangleArea(p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float]) -> float:
    """Compute the area of a triangle given three vertices.
    
    Parameters
    ----------
    p1 : tuple of float
        First vertex as (x, y) coordinates.
    p2 : tuple of float
        Second vertex as (x, y) coordinates.
    p3 : tuple of float
        Third vertex as (x, y) coordinates.
        
    Returns
    -------
    float
        Area of the triangle.
    """
    return 0.5 * (p2[0] * p3[1] - p3[0] * p2[1] - p1[0] * p3[1] + p3[0] * p1[1] + p1[0] * p2[1]
                  - p2[0] * p1[1])


def computeTriangleCentroid(p1: Tuple[float, float], p2: Tuple[float, float],
                            p3: Tuple[float, float]) -> Tuple[float, float]:
    """Compute the centroid of a triangle given three vertices.
    
    Parameters
    ----------
    p1 : tuple of float
        First vertex as (x, y) coordinates.
    p2 : tuple of float
        Second vertex as (x, y) coordinates.
    p3 : tuple of float
        Third vertex as (x, y) coordinates.
        
    Returns
    -------
    tuple of float
        Centroid of the triangle as (x, y) coordinates.
    """
    return np.array([p1, p2, p3]).mean(axis=0)


def isCollinear(p1: Tuple[float, float],
                p2: Tuple[float, float],
                p3: Tuple[float, float],
                tol: float = 1e-6) -> bool:
    """Check if three points are collinear within a given tolerance.
    
    Parameters
    ----------
    p1 : tuple of float
        First point as (x, y) coordinates.
    p2 : tuple of float
        Second point as (x, y) coordinates.
    p3 : tuple of float
        Third point as (x, y) coordinates.
    tol : float, optional
        Tolerance for collinearity test. Default is 1e-6.
        
    Returns
    -------
    bool
        True if the three points are collinear within tolerance.
    """
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1*y2 - x2*y1) < tol


def computeArea(vertices: Iterable[Tuple[float]]) -> float:
    """Compute the area of a polygon given its vertices.
    
    Parameters
    ----------
    vertices : iterable of tuple
        Vertices of the polygon as coordinate tuples.
        
    Returns
    -------
    float
        Area of the polygon.
    """
    area = shapely.geometry.Polygon(vertices).area
    return area


def computeAngle(v1: Tuple[float, float] | shapely.geometry.LineString,
                 v2: Tuple[float, float] | shapely.geometry.LineString) -> float:
    """Compute the angle (in degrees) of v2 relative to v1 in clockwise direction.
    
    Parameters
    ----------
    v1 : tuple of float or shapely.geometry.LineString
        First vector as (x, y) coordinates or downstream-oriented LineString.
    v2 : tuple of float or shapely.geometry.LineString
        Second vector as (x, y) coordinates or downstream-oriented LineString.
        
    Returns
    -------
    float
        Angle in degrees of v2 relative to v1 in clockwise direction (0-360).
    """
    if isinstance(v1, shapely.geometry.LineString):
        if isinstance(v2, shapely.geometry.LineString):
            assert isClose(v1.coords[0], v2.coords[-1])

        c1 = np.array(v1.coords[0:2])[:, 0:2]
        return computeAngle(c1[1] - c1[0], v2)
    if isinstance(v2, shapely.geometry.LineString):
        c2 = np.array(v2.coords[-2:])[:, 0:2]
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


def projectVectorAtAngle(v1: Tuple[float, float] | shapely.geometry.LineString, angle: float,
                         distance: float) -> Tuple[float, float]:
    """Project a vector from v1 at a specified angle and distance.
    
    Parameters
    ----------
    v1 : tuple of float or shapely.geometry.LineString
        Reference vector as (x, y) coordinates or downstream-oriented LineString.
    angle : float
        Angle in degrees for the projection.
    distance : float
        Distance (magnitude) of the projected vector.
        
    Returns
    -------
    tuple of float
        Projected vector as (x, y) coordinates with specified angle and distance.
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


def computeMidpoint(p1: Tuple[float, float] | np.ndarray,
                    p2: Tuple[float, float] | np.ndarray) -> Tuple[float, float]:
    """Compute the midpoint between two points.
    
    Parameters
    ----------
    p1 : tuple of float
        First point as (x, y) coordinates.
    p2 : tuple of float
        Second point as (x, y) coordinates.
        
    Returns
    -------
    tuple of float
        Midpoint as (x, y) coordinates.
    """
    return ((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)


def findClosestPointIndex(point: Tuple[float, float], points: npt.ArrayLike) -> int:
    """Find the index of the closest point in an array of points.
    
    Parameters
    ----------
    point : tuple of float
        Reference point as (x, y) coordinates.
    points : array_like
        Array of points to search, shape (n, 2).
        
    Returns
    -------
    int
        Index of the closest point in the points array.
    """
    points2 = np.asarray(points)
    point2 = np.array(point)
    dist = np.sum((points2 - point2)**2, axis=1)
    return int(np.argmin(dist))


def cluster(points: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster points based on distance tolerance.
    
    Parameters
    ----------
    points : np.ndarray
        Array of points to cluster, shape (n, 2).
    tol : float
        Distance tolerance for clustering.
        
    Returns
    -------
    tuple of np.ndarray
        Tuple containing (cluster_indices, cluster_centroids) where cluster_indices
        is an array of cluster assignments and cluster_centroids are the centroid
        coordinates of each cluster.
    """
    import scipy.cluster.hierarchy as hcluster
    if type(points) is list:
        points = np.array(points)

    if len(points) > 1:
        indices = hcluster.fclusterdata(points, tol, criterion='distance')
        centroids = np.array(
            [points[indices == (i + 1)].mean(axis=0) for i in range(indices.max())])
    else:
        indices = np.array([1, ] * len(points))
        centroids = points
    return indices - 1, centroids


#
# Shape utilities
#


def flatten(list_of_shps: Any) -> List[BaseGeometry]:
    """Flatten a list of shapes by expanding Multi-objects into individual geometries.
    
    Parameters
    ----------
    list_of_shps : iterable
        List of shapely geometry objects, may contain Multi-objects.
        
    Returns
    -------
    list of BaseGeometry
        Flattened list containing only single geometry objects.
    """
    new_list = []
    for shp in list_of_shps:
        if isinstance(shp, shapely.geometry.MultiLineString) or \
           isinstance(shp, shapely.geometry.MultiPoint) or \
           isinstance(shp, shapely.geometry.MultiPolygon):
            new_list.extend(list(shp.geoms))
        else:
            new_list.append(shp)
    return new_list


def recenter(
    objects: Iterable[BaseGeometry],
    centering: bool | str | shapely.geometry.Point = True
) -> Tuple[List[BaseGeometry], shapely.geometry.Point]:
    """Center a collection of objects by translating to remove their centroid.
    
    Parameters
    ----------
    objects : iterable of BaseGeometry
        Collection of shapely geometry objects to center.
    centering : bool, str, or shapely.geometry.Point, optional
        Centering method: True or 'geometric' for geometric center,
        'mass' for center of mass, or Point for custom center.
        
    Returns
    -------
    tuple
        Tuple containing (centered_objects, centroid_point) where centered_objects
        is the list of translated geometries and centroid_point is the original centroid.
        
    Raises
    ------
    ValueError
        If centering method is not recognized.
    """
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


def intersects(shp1: BaseGeometry, shp2: BaseGeometry) -> bool:
    """Check whether two geometries intersect.
    
    Parameters
    ----------
    shp1 : BaseGeometry
        First shapely geometry object.
    shp2 : BaseGeometry
        Second shapely geometry object.
        
    Returns
    -------
    bool
        True if geometries intersect, False otherwise.
        
    Notes
    -----
    This function computes the actual intersection rather than using
    shapely.intersects() for more reliable results.
    """
    inter = shp1.intersection(shp2)
    return not isEmpty(inter)


def isNonPointIntersection(shp1: BaseGeometry, shp2: BaseGeometry) -> bool:
    """Check whether two geometries intersect with more than just a point.
    
    Parameters
    ----------
    shp1 : BaseGeometry
        First shapely geometry object.
    shp2 : BaseGeometry
        Second shapely geometry object.
        
    Returns
    -------
    bool
        True if geometries intersect with line or area, False if no intersection
        or point intersection only.
        
    Notes
    -----
    This function computes the actual intersection to check its dimensionality.
    """
    inter = shp1.intersection(shp2)
    return not (isEmpty(inter) or \
                isinstance(inter, shapely.geometry.Point))


def isVolumetricIntersection(shp1: BaseGeometry, shp2: BaseGeometry) -> bool:
    """Check whether two geometries have a volumetric (area) intersection.
    
    Parameters
    ----------
    shp1 : BaseGeometry
        First shapely geometry object.
    shp2 : BaseGeometry
        Second shapely geometry object.
        
    Returns
    -------
    bool
        True if intersection has positive area, False otherwise.
    """
    inter = shp1.intersection(shp2)
    return inter.area > 0


def filterToShape(df: gpd.GeoDataFrame,
                  shape: BaseGeometry,
                  shape_crs: CRS,
                  method: str = 'contains',
                  tol: Optional[float] = None) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame features based on spatial relationship with shape.
    
    Parameters
    ----------
    df : gpd.GeoDataFrame
        GeoDataFrame to filter.
    shape : BaseGeometry
        Shapely geometry to use as filter.
    shape_crs : CRS
        Coordinate reference system of the shape.
    method : str, optional
        Spatial relationship method: 'contains', 'intersects', or
        'non_point_intersection'. Default is 'contains'.
    tol : float, optional
        Tolerance for spatial operations. Uses default if None.
        
    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame containing features that satisfy the spatial relationship.
        
    Raises
    ------
    ValueError
        If method is not one of the supported options.
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
        raise ValueError(
            "method must be one of 'intersects', 'contains', or 'non_point_intersection'")

    return df[df.geometry.apply(op)]


def isEmpty(shply: BaseGeometry | None) -> bool:
    """Check if a shapely geometry is None or empty.
    
    Parameters
    ----------
    shply : BaseGeometry or None
        Shapely geometry object to check.
        
    Returns
    -------
    bool
        True if geometry is None or empty.
    """
    return shply is None or shply.is_empty


def isConvex(points: Iterable[Tuple[float, float]]) -> bool:
    """Check if a set of points forms a convex polygon.
    
    Parameters
    ----------
    points : iterable of tuple
        Points as coordinate tuples to check for convexity.
        
    Returns
    -------
    bool
        True if points form a convex polygon.
    """
    poly = shapely.geometry.Polygon(points)
    return math.isclose(poly.area, poly.convex_hull.area, rel_tol=1e-4)


def breakLineStringCollinearity(linestring_coords: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Remove collinearity from linestring by adding small orthogonal perturbations.
    
    Parameters
    ----------
    linestring_coords : np.ndarray
        Array of linestring coordinates, shape (n, 2).
    tol : float, optional
        Tolerance for collinearity detection and perturbation size. Default is 1e-5.
        
    Returns
    -------
    np.ndarray
        Modified coordinates with collinearity removed.
    """
    # traversing along the linestring, checking 3 consecutive points at a time
    for i, (p0, p1,
            p2) in enumerate(zip(linestring_coords, linestring_coords[1:], linestring_coords[2:])):
        # treating collinearity through a small pertubation
        if isCollinear(p0, p1, p2, tol=tol):
            dp = p2 - p0
            ortho = 10 * tol * np.array([-dp[1], dp[0]]) / np.linalg.norm(dp)

            linestring_coords[i + 1] = p1 + ortho
    return linestring_coords


def reverseLineString(ls: shapely.geometry.LineString) -> shapely.geometry.LineString:
    """Reverse the direction of a LineString.
    
    Parameters
    ----------
    ls : shapely.geometry.LineString
        LineString to reverse.
        
    Returns
    -------
    shapely.geometry.LineString
        LineString with reversed coordinate order.
    """
    return shapely.geometry.LineString(reversed(ls.coords))


def isClose(s1: BaseGeometry, s2: BaseGeometry, tol: float = _tol) -> bool:
    """Check if two shapely geometries are topologically equivalent and geometrically close.
    
    Parameters
    ----------
    s1 : BaseGeometry
        First shapely geometry object.
    s2 : BaseGeometry
        Second shapely geometry object.
    tol : float, optional
        Tolerance for geometric comparison. Default uses module tolerance.
        
    Returns
    -------
    bool
        True if geometries are topologically equivalent and geometrically close.
        
    Notes
    -----
    This function handles complex cases like polygon rotations and coordinate
    ordering that simple coordinate comparison would miss.
        
    Raises
    ------
    NotImplementedError
        If geometry type is not supported.
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


def contains(s1: BaseGeometry, s2: BaseGeometry, tol: float = _tol) -> bool:
    """Check if one geometry contains another with tolerance for roundoff issues.
    
    Parameters
    ----------
    s1 : BaseGeometry
        Container geometry.
    s2 : BaseGeometry
        Geometry to test for containment.
    tol : float, optional
        Tolerance buffer for containment test. Default uses module tolerance.
        
    Returns
    -------
    bool
        True if s1 contains s2 within tolerance.
    """
    return s1.buffer(tol, 2).contains(s2)


class CutError(Exception):
    """Exception raised when cutting geometries fails.
    
    Parameters
    ----------
    message : str
        Error message.
    line : shapely.geometry.LineString
        The line being cut.
    seg : shapely.geometry.LineString
        The segment causing issues.
    cutline : shapely.geometry.LineString
        The cutting line.
    """
    def __init__(self, message: str, line: shapely.geometry.LineString,
                 seg: shapely.geometry.LineString, cutline: shapely.geometry.LineString) -> None:
        super(Exception, self).__init__(message)
        self.line = line
        self.seg = seg
        self.cutline = cutline


def cut(line1: shapely.geometry.LineString,
        line2: shapely.geometry.LineString,
        eps : float = 1.e-8
) -> Tuple[List[shapely.geometry.LineString], List[shapely.geometry.LineString]]:
    """Cut two LineStrings at their intersection point.
    
    Parameters
    ----------
    line1 : shapely.geometry.LineString
        First LineString to cut.
    line2 : shapely.geometry.LineString
        Second LineString to cut.
        
    Returns
    -------
    tuple of list
        Tuple containing (line1_segments, line2_segments) where each
        is a list of LineString segments created by the cut operation.
        Note the ordering is enforced to be such that coordinate
        ordering is preserved.

    """
    l1_geoms = list(shapely.ops.split(line1, line2).geoms)
    l2_geoms = list(shapely.ops.split(line2, line1).geoms)

    # permute if needed
    def _permute(l0, l0_geoms):
        if len(l0_geoms) == 1:
            l0g = l0_geoms[0]
            d0 = computeDistance(l0.coords[0], l0g.coords[0])
            d1 = computeDistance(l0.coords[0], l0g.coords[-1])

            if d1 < d0 and d1 < eps:
                return [reverseLineString(l0g),]
            elif d0 < eps:
                return l0_geoms
            else:
                raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 1')

        elif len(l0_geoms) == 2:
            res = []
            l0g = l0_geoms[0]
            l1g = l0_geoms[1]

            # find closest point of the 4 endpoints to the 0th endpoint of l0
            d0 = computeDistance(l0.coords[0], l0g.coords[0])
            d1 = computeDistance(l0.coords[0], l0g.coords[-1])
            d2 = computeDistance(l0.coords[0], l1g.coords[0])
            d3 = computeDistance(l0.coords[0], l1g.coords[-1])
            dmin = min([d0, d1, d2, d3])

            # orient the first segment off of the closest point
            if d0 == dmin and d0 < eps:
                l00_is_0 = True
                res.append(l0g)
            elif d1 == dmin and d1 < eps:
                l00_is_0 = True
                res.append(reverseLineString(l0g))
            elif d2 == dmin and d2 < eps:
                l00_is_0 = False
                res.append(l1g)
            elif d3 == dmin and d3 < eps:
                l00_is_0 = False
                res.append(reverseLineString(l1g))
            else:
                raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 2')

            # find closest point of the 4 endpoints to the last endpoint of l0
            d0 = computeDistance(l0.coords[-1], l0g.coords[0])
            d1 = computeDistance(l0.coords[-1], l0g.coords[-1])
            d2 = computeDistance(l0.coords[-1], l1g.coords[0])
            d3 = computeDistance(l0.coords[-1], l1g.coords[-1])
            dmin = min([d0, d1, d2, d3])

            # orient the second segment off of the closest point
            if d0 == dmin and d0 < eps:
                if l00_is_0:
                    raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 3')
                res.append(reverseLineString(l0g))
            elif d1 == dmin and d1 < eps:
                if l00_is_0:
                    raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 4')
                res.append(l0g)
            elif d2 == dmin and d2 < eps:
                if not l00_is_0:
                    raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 5')
                res.append(reverseLineString(l1g))
            elif d3 == dmin and d3 < eps:
                if not l00_is_0:
                    raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 6')
                res.append(l1g)
            else:
                raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 7')

            if computeDistance(res[0].coords[-1], res[1].coords[0]) > eps:
                raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 8')
            return res

        else:
            raise CutError('Cutting resulted in lines that do not share an endpoint with the original line? Error 9')

    return _permute(line1, l1_geoms), _permute(line2, l2_geoms)
        
    

    


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


def inNeighborhood(shp1: BaseGeometry, shp2: BaseGeometry, tol: float = 0.1) -> bool:
    """Check if two geometries can possibly intersect using bounding box test.
    
    Parameters
    ----------
    shp1 : BaseGeometry
        First shapely geometry object.
    shp2 : BaseGeometry
        Second shapely geometry object.
    tol : float, optional
        Tolerance for bounding box expansion. Default is 0.1.
        
    Returns
    -------
    bool
        True if bounding boxes indicate possible intersection.
    """
    minx1, miny1, maxx1, maxy1 = shp1.bounds
    minx2, miny2, maxx2, maxy2 = shp2.bounds
    if maxx2 < minx1 - tol or \
       maxy2 < miny1 - tol or \
       minx2 > maxx1 + tol or \
       miny2 > maxy1 + tol:
        return False
    return True


def intersectPointToSegment(point: shapely.geometry.Point, seg_start: shapely.geometry.Point,
                            seg_end: shapely.geometry.Point) -> shapely.geometry.Point:
    """Find the nearest point on a line segment to a given point.
    
    Parameters
    ----------
    point : shapely.geometry.Point
        Point to find nearest location for.
    seg_start : shapely.geometry.Point
        Start point of the line segment.
    seg_end : shapely.geometry.Point
        End point of the line segment.
        
    Returns
    -------
    shapely.geometry.Point
        Nearest point on the line segment to the input point.
    """
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


def findNearestPoint(point: shapely.geometry.Point | Tuple[float, float],
                     line: shapely.geometry.LineString,
                     tol: Optional[float] = None) -> Tuple[float, float] | None:
    """Find the nearest point on a LineString to a given point.
    
    Parameters
    ----------
    point : shapely.geometry.Point or tuple of float
        Point to find nearest location for.
    line : shapely.geometry.LineString
        LineString to search along.
    tol : float, optional
        Distance tolerance. If provided, returns None if nearest point
        is farther than tolerance.
        
    Returns
    -------
    tuple of float or None
        Nearest point coordinates as (x, y) tuple, or None if outside tolerance.
    """
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


def removeThirdDimension(
        geom: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Remove the third dimension (Z-coordinate) from a shapely geometry.
    
    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input geometry that may have Z-coordinates.
        
    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Geometry with only X and Y coordinates.
    """
    def _drop_z(*args):
        return tuple(filter(None, [args[0], args[1]]))

    return shapely.ops.transform(_drop_z, geom)


def computeSegmentLengths(ls: shapely.geometry.LineString) -> np.ndarray:
    """Compute the length of each segment in a LineString.
    
    Parameters
    ----------
    ls : shapely.geometry.LineString
        LineString to analyze.
        
    Returns
    -------
    np.ndarray
        Array of segment lengths, shape (n-1,) for n coordinates.
    """
    coords = np.array(ls.coords)
    return np.linalg.norm((coords[1:] - coords[:-1]), axis=1)


def computeArclengths(ls: shapely.geometry.LineString) -> np.ndarray:
    """Compute cumulative arc length at each coordinate in a LineString.
    
    Parameters
    ----------
    ls : shapely.geometry.LineString
        LineString to analyze.
        
    Returns
    -------
    np.ndarray
        Array of cumulative arc lengths, shape (n,) for n coordinates.
    """
    ds = computeSegmentLengths(ls)
    return np.cumsum(np.concatenate([[0.0, ], ds]))


def logMinMaxMedianSegment(iterable: Iterable[shapely.geometry.LineString],
                           name: str,
                           assert_on_zero: bool = False,
                           ax: Optional[matplotlib.axes.Axes] = None,
                           color: Optional[str | Tuple] = None) -> None:
    """Log statistics and optionally plot histogram of segment lengths.
    
    Parameters
    ----------
    iterable : iterable of shapely.geometry.LineString
        Collection of LineStrings to analyze.
    name : str
        Name for logging and plot labels.
    assert_on_zero : bool, optional
        If True, raise assertion error if any segment has zero length.
    ax : matplotlib.axes.Axes, optional
        Axes to plot histogram on. No plot if None.
    color : str or tuple, optional
        Color for histogram plot.
        
    Raises
    ------
    AssertionError
        If assert_on_zero is True and zero-length segments are found.
    """
    seg_mins: List[float] = []
    seg_maxs: List[float] = []
    seg_meds: List[float] = []
    seg_lens: List[np.ndarray] = []
    geom_lens: List[float] = []

    for ls in iterable:
        geom_lens.append(ls.length)
        seg_len = computeSegmentLengths(ls)
        seg_lens.append(seg_len)
        if assert_on_zero:
            assert (min(seg_len) > 1.e-10)

    seg_lens_a = np.concatenate(seg_lens)
    seg_diags = np.min(seg_lens_a), np.median(seg_lens_a), np.max(seg_lens_a)

    geom_lens_a = np.array(geom_lens)
    geom_diags = (min(geom_lens_a), np.median(geom_lens_a), max(geom_lens_a))

    if ax is not None:
        ax.hist(seg_lens_a, max(len(seg_lens_a) // 20, 10), color=color, label=name)
        ax.set_xlabel('segment length')
        ax.set_ylabel('count')
        ax.legend()

    logging.info(
        f"  {name}: min seg length: \t{seg_diags[0]:16.10f} \tmin geom length: \t{geom_diags[0]:16.10f}"
    )
    logging.info(
        f"  {name}: med seg length: \t{seg_diags[1]:16.10f} \tmed geom length: \t{geom_diags[1]:16.10f}"
    )
    logging.info(
        f"  {name}: max seg length: \t{seg_diags[2]:16.10f} \tmax geom length: \t{geom_diags[2]:16.10f}"
    )
    logging.info('')
    return


#
# Dataset utilities
#


#
# fiona utilities -- probably need to go away?
#
def generateRings(obj: Any) -> Iterable[List[Tuple[float, float]]]:
    """Generate coordinate rings from a fiona shape object.

    Parameters
    ----------
    obj : dict
        Fiona shape dictionary with 'coordinates' or 'geometry' key.

    Yields
    ------
    list of tuple
        Each ring as a list of coordinate tuples.
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


def generateCoords(obj: Any) -> Iterable[Tuple[float, float]]:
    """Generate all coordinates from a fiona geometry object.

    Parameters
    ----------
    obj : dict
        Fiona shape dictionary with 'coordinates' or 'geometry' key.

    Yields
    ------
    tuple of float
        Individual coordinate tuples.
    """
    if 'geometry' in obj:
        obj = obj['geometry']

    if obj['type'] == 'Point':
        yield obj['coordinates']
    else:
        for ring in generateRings(obj):
            for c in ring:
                yield c
