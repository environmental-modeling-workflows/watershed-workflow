"""Shape and geometry utilities for working with fiona and shapely objects.

Note this module contains a lot of other simple functions that are commonly
used by other functions, but are not included in documentation because they are
likely not useful to users.

"""

import logging
import subprocess
import numpy as np
import shapely.geometry
import shapely.ops
import shapely.affinity
import rasterio
import watershed_workflow 

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
        if type(thing) is shapely.geometry.MultiPoint and len(thing) == 1:
            thing = thing[0]
        elif type(thing) is shapely.geometry.MultiLineString and len(thing) == 1:
            thing = thing[0]
        elif type(thing) is shapely.geometry.MultiPolygon and len(thing) == 1:
            thing = thing[0]

        # first check for latlon instead of lonlat
        if flip:
            thing = shapely.ops.transform(lambda x,y:(y,x), thing)

        thing.properties = properties
        thing_2D = remove_third_dimension(thing)
        thing_2D.properties = thing.properties
        return thing_2D

    except ValueError:
        raise ValueError('Converting to shapely got error: "%s"  Maybe you forgot to do shp["geometry"]?')


def round_shapes(list_of_things, digits):
    """Rounds coordinates in things or shapes to a given digits."""
    for shp in list_of_things:
        for ring in generate_rings(shp):
            ring[:] = list(np.array(ring).round(digits))

def round_shplys(list_of_things, digits):
    """Rounds coordinates in things or shapes to a given digits."""
    return [shapely.wkt.loads(shapely.wkt.dumps(thing, rounding_precision=digits)).simplify(0) for thing in list_of_things]

_tol = 1.e-7
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
    # points get compared as tuples
    if isinstance(s1, shapely.geometry.Point):
        return close(s1.coords[0], s2, tol)
    if isinstance(s2, shapely.geometry.Point):
        return close(s1, s2.coords[0], tol)

    # length 1 multi-shapes get compared as individual shapes
    if isinstance(s1, (shapely.geometry.MultiPoint,
                       shapely.geometry.MultiLineString,
                       shapely.geometry.MultiPolygon)) and \
       len(s1) == 1:
        return close(s1[0], s2, tol)
    if isinstance(s2, (shapely.geometry.MultiPoint,
                       shapely.geometry.MultiLineString,
                       shapely.geometry.MultiPolygon)) and \
       len(s2) == 1:
        return close(s1, s2[0], tol)

    # types should be the same now
    if type(s1) != type(s2):
        return False

    # compare tuples
    if isinstance(s1, tuple):    
        if len(s1) != len(s2):
            return False
        return sum((p1 - p2)**2 for p1,p2 in zip(s1,s2)) < tol**2

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
    elif isinstance(s1, (shapely.geometry.MultiPoint,
                         shapely.geometry.MultiLineString,
                         shapely.geometry.MultiPolygon)):
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
                if i != 0:
                    assert(len(segcoords) > 1)
                    segs.append(shapely.geometry.LineString(segcoords[:-1]+[point,]))
                    segcoords = [point,]
                else:
                    assert(len(segcoords) == 1)
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
            print("Dual/multiple section: type = {}".format(type(point)))
            print(" point = {}".format(point))
            raise RuntimeError("Dual/multiple intersection in a single seg... ugh!  "+
                               "Intersection is of type '{}'".format(type(point)))
        
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


def angle(v1, v2):
    """Given two 2D vectors represented as len 2 arrays or tuples, find the angle
    of 2 relative to 1 in a clockwise notion."""
    x1 = v1[0]; y1 = v1[1]
    x2 = v2[0]; y2 = v2[1]
    mag = 180./np.pi * np.arccos((x1 * x2 + y1 * y2) / ( np.sqrt(x1*x1 + y1*y1) * np.sqrt(x2*x2 + y2*y2)))
    sign =  x1*y2 - x2*y1
    if sign < 0:
        return -mag
    else:
        return mag


def midpoint(p1, p2):
    """Returns the midpoint of two points"""
    if type(p1) is tuple:
        return ( (p1[0] + p2[0])/2., (p1[1] + p2[1])/2. )
    else:
        return shapely.geometry.Point(midpoint(p1.coords[0], p2.coords[0]))
    
def center(objects, centering=True):
    """Centers a collection of objects by removing their collective centroid"""
    if type(centering) is shapely.geometry.Point:
        centroid = centering
    elif centering is True or centering == 'geometric':
        union = shapely.ops.cascaded_union(objects)
        centroid = shapely.geometry.Point([(union.bounds[0] + union.bounds[2])/2., (union.bounds[1] + union.bounds[3])/2.])
    elif centering == 'mass':
        union = shapely.ops.cascaded_union(objects)
        centroid = union.centroid
    else:
        raise ValueError('Centering: option centering = "{}" unknown'.format(centering))

    new_objs = [shapely.affinity.translate(obj, -centroid.coords[0][0], -centroid.coords[0][1]) for obj in objects]
    
    for new, old in zip(new_objs, objects):
        if hasattr(old, 'properties'):
            new.properties = old.properties
        
    return new_objs, centroid
    

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


def empty_shapely(shp):
    if shp is None:
        return True
    if type(shp) is shapely.geometry.GeometryCollection and len(shp) == 0:
        return True
    if type(shp) is shapely.geometry.LineString and len(shp.coords) == 0:
        return True
    if type(shp) is shapely.geometry.Polygon and len(shp.exterior.coords) == 0:
        return True
    return False
        
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
    if type(inter) == shapely.geometry.Point:
        return False
    elif empty_shapely(inter):
        return False
    return True

def flatten(list_of_shps):
    """Flattens a list of shapes, that may contain Multi-objects, into  list without multi-objects"""
    new_list = []
    for shp in list_of_shps:
        if type(shp) == shapely.geometry.MultiLineString or type(shp) == shapely.geometry.MultiPolygon:
            new_list.extend([s for s in shp])
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
        raise RuntimeError("Currently this type of geometry is not supported: {}".format(type(geom)))


def create_empty_raster(target_bounds,crs,target_dx,dtype, nodata):
    target_x0 = np.round(target_bounds[0] - target_dx/2)
    target_y1 = np.round(target_bounds[3] + target_dx/2)
    width = int(np.ceil((target_bounds[2] + target_dx/2 - target_x0)/target_dx))
    height = int(np.ceil((target_y1 - target_bounds[1] - target_dx/2)/target_dx))

    out_bounds = [target_x0, target_y1 - target_dx*height, target_x0 + target_dx*width, target_y1]

    logging.info('  target_bounds = {}'.format(target_bounds))
    logging.info('  out_bounds = {}'.format(out_bounds))
    logging.info('  pixel_size = {}'.format(target_dx))
    logging.info('  width = {}, height = {}'.format(width, height))

    transform = rasterio.transform.from_origin(target_x0, target_y1, target_dx, target_dx)

    out_profile = {'height':height,
                      'width':width,
                      'count':1,
                      'dtype':dtype,
                      'crs':watershed_workflow.crs.to_rasterio(crs),
                      'transform':transform,
                      'nodata':nodata}
    out = nodata * np.ones((height, width), dtype)
    return out_profile, out

def CrossProduct(A):
     
    # Stores coefficient of X
    # direction of vector A[1]A[0]
    X1 = (A[1][0] - A[0][0])
 
    # Stores coefficient of Y
    # direction of vector A[1]A[0]
    Y1 = (A[1][1] - A[0][1])
 
    # Stores coefficient of X
    # direction of vector A[2]A[0]
    X2 = (A[2][0] - A[0][0])
 
    # Stores coefficient of Y
    # direction of vector A[2]A[0]
    Y2 = (A[2][1] - A[0][1])
 
    # Return cross product
    return (X1 * Y2 - Y1 * X2)
 

def isConvex(points):
    """# Function to check if the polygon is convex polygon or not"""
     
    # Stores count of
    # edges in polygon
    N = len(points)
 
    # Stores direction of cross product
    # of previous traversed edges
    prev = 0
 
    # Stores direction of cross product
    # of current traversed edges
    curr = 0
 
    # Traverse the array
    for i in range(N):
         
        # Stores three adjacent edges
        # of the polygon
        temp = [points[i], points[(i + 1) % N],
                           points[(i + 2) % N]]
 
        # Update curr
        curr = CrossProduct(temp)
 
        # If curr is not equal to 0
        if (curr != 0):
             
            # If direction of cross product of
            # all adjacent edges are not same
            if (curr * prev < 0):
                return False
            else:
                 
                # Update curr
                prev = curr
 
    return True


def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)

def cluster(points, tol):
    """Given a list of points, determine a list of clusters.

    Each cluster is within tol of each other.

    Returns (cluster_index, cluster_centroid)
    """
    import scipy.cluster.hierarchy as hcluster
    if type(points) is list:
        points = np.array(points)
    indices = hcluster.fclusterdata(points, tol, criterion='distance')
    centroids = [points[indices==(i+1)].mean(axis=0) for i in range(indices.max())]
    return indices-1, centroids
    
