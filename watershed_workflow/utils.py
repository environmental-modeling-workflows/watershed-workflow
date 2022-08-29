"""Shape and geometry utilities for working with fiona and shapely objects.

Note this module contains a lot of other simple functions that are commonly
used by other functions, but are not included in documentation because they are
likely not useful to users.

"""

import logging
import subprocess
import numpy as np
import math
import scipy.interpolate
import shapely.geometry
import shapely.ops
import shapely.prepared
import shapely.affinity
import rasterio

import watershed_workflow.crs

_tol = 1.e-7


def generate_rings(obj):
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
    def _generate_rings(coords):
        for e in coords:
            if isinstance(e[0], (float, int)):
                yield coords
                break
            else:
                for r in _generate_rings(e):
                    yield r

    if 'geometry' in obj:
        obj = obj['geometry']
    for r in _generate_rings(obj['coordinates']):
        yield r


def generate_coords(obj):
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
        for ring in generate_rings(obj):
            for c in ring:
                yield c


def bounds(f):
    """General bounding box for fiona and shapely types."""
    # fiona type
    x, y = zip(*list(generate_coords(f)))
    # except TypeError:
    #     # shapely type
    #     return f.bounds
    # else:
    return min(x), min(y), max(x), max(y)


def shply(shape, properties=None, flip=False):
    """Converts a fiona style shape to a shapely shape with as much collapsing as possible.

    Note this collapses objects -- for instance, fiona MultiPolygons of length
    1 are turned into shapely Polygons.

    Parameters
    ----------
    shape : fiona shape
      Fiona shape to convert to shapely.
    properties : dict, optional
      A dictionary of parameters to associate with the object.  Defaults to
      shape['properties'] if it exists, None otherwise.
    flip : bool, optional
      Flip x,y coordinates while making the translation.  This helps if files
      provide lat-long ordered coordinates (note that is y,x) as opposed to
      long-lat (x,y).  Default is False.

    Returns
    -------
    thing : shapely shape
    """
    if 'geometry' in shape:
        if properties is None:
            if 'properties' in shape:
                properties = shape['properties']
            else:
                properties = dict()
        shape = shape['geometry']

    try:
        thing = shapely.geometry.shape(shape)
        if type(thing) is shapely.geometry.MultiPoint and len(thing.geoms) == 1:
            thing = thing.geoms[0]
        elif type(thing) is shapely.geometry.MultiLineString and len(thing.geoms) == 1:
            thing = thing.geoms[0]
        elif type(thing) is shapely.geometry.MultiPolygon and len(thing.geoms) == 1:
            thing = thing.geoms[0]

        # first check for latlon instead of lonlat
        if flip:
            thing = shapely.ops.transform(lambda x, y: (y, x), thing)

        thing.properties = properties
        thing_2D = remove_third_dimension(thing)
        thing_2D.properties = thing.properties
        return thing_2D

    except ValueError:
        raise ValueError(
            'Converting to shapely got error: "%s"  Maybe you forgot to do shp["geometry"]?')


def round_shapes(list_of_things, digits):
    """Rounds coordinates in things or shapes to a given digits."""
    for shp in list_of_things:
        for ring in generate_rings(shp):
            ring[:] = list(np.array(ring).round(digits))


def round_shplys(list_of_things, digits):
    """Rounds coordinates in things or shapes to a given digits."""
    return [
        shapely.wkt.loads(shapely.wkt.dumps(thing, rounding_precision=digits)).simplify(0)
        for thing in list_of_things
    ]


def close(s1, s2, tol=_tol):
    """Are two shapely shapes topologically equivalent and geometrically close?

    Note this deals with things like rotations of polygons (clock-rotating the
    coordinates of the same shape are still close) and other gotchas that keep
    you from just comparing coordinates.

    Parameters
    ----------
    s1, s2 : shapely shapes
      Objects to compare.
    tol : double
      Distance to compare geometric closeness.

    Returns
    -------
    close : bool
      Is close?
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
            return close(next(iter(s1)), s2, tol)

    if is_multi(s2):
        if local_len(s2) == 1:
            return close(s1, next(iter(s2)), tol)

    if is_multi(s1) and is_multi(s2):
        if local_len(s1) != local_len(s2):
            return False
        return all(close(i1, i2, tol) for (i1, i2) in zip(s1, s2))

    # points get compared as tuples
    if isinstance(s1, shapely.geometry.Point):
        return close(s1.coords[0], s2, tol)
    if isinstance(s2, shapely.geometry.Point):
        return close(s1, s2.coords[0], tol)

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
                    if close(l1, l2, tol):
                        good2[i] = True
                        found = True
                        break
            if not found:
                return False
        return True
    else:
        raise NotImplementedError("Not implemented for type '%r'" % type(s1))


def contains(s1, s2, tol=_tol):
    """A contains algorithm that deals with close/roundoff issues"""
    return s1.buffer(tol, 2).contains(s2)


def cut(line, cutline, tol=1.e-5):
    """Cuts a line at all intersections with cutline.  If an existing
    point in line is within tol of the cutline, do not add an additional
    coordinate, just move that coordinate.  Otherwise, add a new
    coordinate."""
    def plot():
        from matplotlib import pyplot as plt
        plt.plot(cutline.xy[0], cutline.xy[1], 'k-x', linewidth=3)
        plt.plot(line.xy[0], line.xy[1], 'g-+', linewidth=3)

    assert (type(line) is shapely.geometry.LineString)
    assert (type(cutline) is shapely.geometry.LineString)
    assert (line.intersects(cutline))

    segs = []
    coords = list(line.coords)

    segcoords = [coords[0], ]
    i = 0
    while i < len(coords) - 1:
        seg = shapely.geometry.LineString(coords[i:i + 2])
        #logging.debug("Intersecting seg %d"%i)
        point = seg.intersection(cutline)
        if type(point) is shapely.geometry.LineString and len(point.coords) == 0:
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
                segs.append(shapely.geometry.LineString(segcoords + [point, ]))
                #logging.debug("  appended full segment: %r"%(list(segs[-1].coords)))

                if (i < len(coords) - 2):
                    #logging.debug("    (not the end)")
                    segcoords = [point, coords[i + 2]]
                else:
                    #logging.debug("    (the end)")
                    segcoords = [point, ]
                i += 2  # also skip the next seg, which would also
                # intersect at that seg's start point
            elif close(point, seg.coords[0], tol):
                # intersects at the near point
                if i != 0:
                    assert (len(segcoords) > 1)
                    segs.append(shapely.geometry.LineString(segcoords[:-1] + [point, ]))
                    segcoords = [point, ]
                else:
                    assert (len(segcoords) == 1)
                    segcoords[0] = point
                segcoords.append(seg.coords[-1])
                i += 1
            else:
                # intersects in the middle
                segs.append(shapely.geometry.LineString(segcoords + [point, ]))
                #logging.debug("  appended partial segment: %r"%(list(segs[-1].coords)))
                segcoords = [point, seg.coords[-1]]
                i += 1
        else:
            print("Dual/multiple section: type = {}".format(type(point)))
            print(" point = {}".format(point))
            raise RuntimeError("Dual/multiple intersection in a single seg... ugh!  "
                               + "Intersection is of type '{}'".format(type(point)))

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
    minx1, miny1, maxx1, maxy1 = obj1.bounds
    minx2, miny2, maxx2, maxy2 = obj2.bounds
    if maxx2 < minx1 - tol or \
       maxy2 < miny1 - tol or \
       minx2 > maxx1 + tol or \
       miny2 > maxy1 + tol:
        return False
    return True


def intersect_point_to_segment(point, line_start, line_end):
    """Finds the nearest point on a line segment to a point"""
    line_magnitude = line_end.distance(line_start)
    assert (line_magnitude > _tol)
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
    if isinstance(point, tuple):
        point = shapely.geometry.Point(point)
    return shapely.ops.nearest_points(point, line)[1].coords[0]


def triangle_area(vertices):
    """Area of a triangle in 2D"""
    xy1 = vertices[0]
    xy2 = vertices[1]
    xy3 = vertices[2]

    A = 0.5 * (xy2[0] * xy3[1] - xy3[0] * xy2[1] - xy1[0] * xy3[1] + xy3[0] * xy1[1]
               + xy1[0] * xy2[1] - xy2[0] * xy1[1])
    return A


def area(vertices):
    """Area of polygons in 2D"""
    area = shapely.geometry.Polygon(vertices).area
    return area


def angle(v1, v2):
    """Given two 2D vectors represented as len 2 arrays or tuples, find the angle
    of 2 relative to 1 in a clockwise notion."""
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    mag = 180. / np.pi * np.arccos((x1*x2 + y1*y2) /
                                   (np.sqrt(x1*x1 + y1*y1) * np.sqrt(x2*x2 + y2*y2)))
    sign = x1*y2 - x2*y1
    if sign < 0:
        return -mag
    else:
        return mag


def midpoint(p1, p2):
    """Returns the midpoint of two points"""
    if isinstance(p1, shapely.geometry.Point):
        return midpoint(p1.coords[0], p2)
    if isinstance(p2, shapely.geometry.Point):
        return midpoint(p1, p2.coords[0])
    return ((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)


def closest_point_ind(point, points):
    """Returns the index of closest point from an array of points"""
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)


def center(objects, centering=True):
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


def orientation(p1, p2, p3):
    """to find the orientation of an ordered triplet (p1,p2,p3) function returns the following values:
      0 : Collinear points
      1 : Clockwise points
      2 : Counterclockwise """

    val = (float(p2.y - p1.y) * (p3.x - p2.x)) - \
           (float(p2.x - p1.x) * (p3.y - p2.y))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0


def empty_shapely(shp):
    """Is shp None or empty"""
    return shp is None or shp.is_empty


def intersects(shp1, shp2):
    """Checks whether an intersection exists.
    
    Note that intersection being empty and intersects are not always reliably
    the same... we avoid using shapely.intersects() for this reason.
    """
    inter = shp1.intersection(shp2)
    return not empty_shapely(inter)


def non_point_intersection(shp1, shp2):
    """Checks whether an intersection is larger than a point.
    
    Note that intersection being empty and intersects are not always reliably
    the same... we avoid using intersects() for this reason.
    """
    inter = shp1.intersection(shp2)
    return not (empty_shapely(inter) or isinstance(inter, shapely.geometry.Point))


def filter_to_shape(shape, to_filter, tol=None, algorithm='intersects'):
    """Filters out reaches (or reaches in rivers) not inside the HUCs provided.

    algorithm is one of 'contains' or 'intersects' to indicate whether
    to include things entirely in shape or partially in shape,
    respectively.
    """
    if tol is None: tol = _tol
    if algorithm == 'contains':
        op = shape.contains
        shape = shapely.prepared.prep(shape.buffer(2 * tol))
    elif algorithm == 'intersects':
        op = shape.intersects
        shape = shapely.prepared.prep(shape.buffer(2 * tol))
    elif algorithm == 'non_point_intersection':
        op = lambda a: non_point_intersection(shape, a)
        shape = shape.buffer(2 * tol)
    else:
        raise ValueError("algorithm must be one of 'intersects' or 'contains'")
    return [s for s in to_filter if op(s)]


def flatten(list_of_shps):
    """Flattens a list of shapes, that may contain Multi-objects, into  list without multi-objects"""
    new_list = []
    for shp in list_of_shps:
        if isinstance(shp, shapely.geometry.MultiLineString) or \
           isinstance(shp, shapely.geometry.MultiPoint) or \
           isinstance(shp, shapely.geometry.MultiPolygon):
            new_list.extend(list(shp.geoms))
        else:
            new_list.append(shp)
    return new_list


def remove_third_dimension(geom):
    """Removes the third dimension of a shapely object."""
    if geom.is_empty:
        return geom

    if isinstance(geom, shapely.geometry.Polygon):
        exterior = geom.exterior
        new_exterior = remove_third_dimension(exterior)
        interiors = geom.interiors
        new_interiors = []
        for int in interiors:
            new_interiors.append(remove_third_dimension(int))
        return shapely.geometry.Polygon(new_exterior, new_interiors)

    elif isinstance(geom, shapely.geometry.LinearRing):
        return shapely.geometry.LinearRing([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, shapely.geometry.LineString):
        return shapely.geometry.LineString([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, shapely.geometry.Point):
        return shapely.geometry.Point([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, shapely.geometry.MultiPoint):
        points = list(geom.geoms)
        new_points = []
        for point in points:
            new_points.append(remove_third_dimension(point))

        return shapely.geometry.MultiPoint(new_points)

    elif isinstance(geom, shapely.geometry.MultiLineString):
        lines = list(geom.geoms)
        new_lines = []
        for line in lines:
            new_lines.append(remove_third_dimension(line))

        return shapely.geometry.MultiLineString(new_lines)

    elif isinstance(geom, shapely.geometry.MultiPolygon):
        pols = list(geom.geoms)

        new_pols = []
        for pol in pols:
            new_pols.append(remove_third_dimension(pol))

        return shapely.geometry.MultiPolygon(new_pols)

    elif isinstance(geom, shapely.geometry.GeometryCollection):
        geoms = list(geom.geoms)

        new_geoms = []
        for geom in geoms:
            new_geoms.append(remove_third_dimension(geom))

        return shapely.geometry.GeometryCollection(new_geoms)

    else:
        raise RuntimeError("Currently this type of geometry is not supported: {}".format(
            type(geom)))


def create_empty_raster(target_bounds, crs, target_dx, dtype, nodata):
    """Generates a profile and a nodata-filled 2D array."""
    target_x0 = np.round(target_bounds[0] - target_dx/2)
    target_y1 = np.round(target_bounds[3] + target_dx/2)
    width = int(np.ceil((target_bounds[2] + target_dx/2 - target_x0) / target_dx))
    height = int(np.ceil((target_y1 - target_bounds[1] - target_dx/2) / target_dx))

    out_bounds = [target_x0, target_y1 - target_dx*height, target_x0 + target_dx*width, target_y1]

    logging.info('  target_bounds = {}'.format(target_bounds))
    logging.info('  out_bounds = {}'.format(out_bounds))
    logging.info('  pixel_size = {}'.format(target_dx))
    logging.info('  width = {}, height = {}'.format(width, height))

    transform = rasterio.transform.from_origin(target_x0, target_y1, target_dx, target_dx)

    out_profile = {
        'height': height,
        'width': width,
        'count': 1,
        'dtype': dtype,
        'crs': watershed_workflow.crs.to_rasterio(crs),
        'transform': transform,
        'nodata': nodata
    }
    out = nodata * np.ones((height, width), dtype)
    return out_profile, out


def is_convex(points):
    poly = shapely.geometry.Polygon(points)
    return math.isclose(poly.area, poly.convex_hull.area, rel_tol=1e-4)


def cluster(points, tol):
    """Given a list of points, determine a list of clusters.

    Each cluster is within tol of each other.

    Returns (cluster_index, cluster_centroid)
    """
    import scipy.cluster.hierarchy as hcluster
    if type(points) is list:
        points = np.array(points)
    indices = hcluster.fclusterdata(points, tol, criterion='distance')
    centroids = [points[indices == (i + 1)].mean(axis=0) for i in range(indices.max())]
    return indices - 1, centroids


def impute_holes2D(arr, nodata=np.nan, method='cubic'):
    """Very simple imputation algorithm to interpolate values for missing data in rasters.

    Note, this may throw if there is a hole on the boundary?

    Parameters
    ----------
    arr : np.ndarray
      2D array, with missing data.
    nodata : optional = np.nan
      Value to treat as a hole to fill.
    method : str, optional = 'cubic'
      Algorithm to use (see scipy.interpolate.griddata).  Likely
      'cubic', 'linear', or 'nearest'.

    Returns
    -------
    np.ndarray
      New array with no values of nodata.

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

    res = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
    return res
