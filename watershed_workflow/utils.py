"""Shape and geometry utilities for working with fiona and shapely objects.

Note this module contains a lot of other simple functions that are commonly
used by other functions, but are not included in documentation because they are
likely not useful to users.

"""

import calendar
import datetime
import cftime
import logging
import subprocess
import numpy as np
import math
import scipy.interpolate
import shapely.geometry
import shapely.ops
import shapely.wkt
import shapely.prepared
import shapely.affinity
import rasterio
import rasterio.transform
import copy

import watershed_workflow.crs

_tol = 1.e-7


def createRasterProfile(bounds, crs, resolution, dtype=None, nodata=None, count=1):
    """Creates a profile for a raster.

    Parameters
    ----------
    bounds : [x_min, y_min, x_max, y_max]
      Bounding box for the raster.
    crs : CRS object
      Target coordinate system.
    resolution : tuple or float
      Pixel width, in units of the crs.  If a tuple, (dx,dy).  If a
      float, then dx = dy.
    dtype : optional
      If provided, sets the data type.
    nodata : dtype, optional
      If provided, sets the nodata value.
    count : int, optional

    Note that dx/dy are always used.  The bounds are adjusted to make
    them an even multiple of dx/dy.


    Returns
    -------
    dict 
      Dictionary profile, including a transform and all other needed
      metadata to create a raster.

    """
    try:
        dx, dy = resolution
    except TypeError:
        dx = resolution
        dy = resolution

    if dtype is None and nodata is not None:
        dytpe = type(nodata)

    x0 = np.round(bounds[0] - dx/2)
    y1 = np.round(bounds[3] + dx/2)
    width = int(np.ceil((bounds[2] + dx/2 - x0) / dx))
    height = int(np.ceil((y1 - bounds[1] - dx/2) / dx))

    out_bounds = [x0, y1 - dy*height, x0 + dx*width, y1]
    transform = rasterio.transform.from_origin(x0, y1, dx, dx)

    out_profile = {
        'height': height,
        'width': width,
        'count': count,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'nodata': nodata
    }
    return out_profile


def createEmptyRaster(bounds, crs, resolution, nodata, count=1):
    """Generates a profile and a nodata-filled array."""
    profile = create_raster_profile(bounds, crs, resolution, nodata=nodata, count=count)
    out = profile['nodata'] * np.ones(
        (profile['count'], profile['height'], profile['width']), profile['dtype'])
    return profile, out


def roundShplys(df, digits):
    """Rounds coordinates in things or shapes to a given digits."""
    new_geom = [
        shapely.wkt.loads(shapely.wkt.dumps(thing, rounding_precision=digits)).simplify(0)
        for thing in df.geometry
    ]
    df['geometry'] = new_geom
    

def _removeThirdDimension(shply):
    """Removes the third dimension of a shapely object."""
    if shply.is_empty:
        return shply

    if isinstance(shply, shapely.geometry.Polygon):
        exterior = shply.exterior
        new_exterior = remove_third_dimension(exterior)
        interiors = shply.interiors
        new_interiors = []
        for int in interiors:
            new_interiors.append(remove_third_dimension(int))
        return shapely.geometry.Polygon(new_exterior, new_interiors)

    elif isinstance(shply, shapely.geometry.LinearRing):
        return shapely.geometry.LinearRing([xy[0:2] for xy in list(shply.coords)])

    elif isinstance(shply, shapely.geometry.LineString):
        return shapely.geometry.LineString([xy[0:2] for xy in list(shply.coords)])

    elif isinstance(shply, shapely.geometry.Point):
        return shapely.geometry.Point([xy[0:2] for xy in list(shply.coords)])

    elif isinstance(shply, shapely.geometry.MultiPoint):
        points = list(shply.geoms)
        new_points = []
        for point in points:
            new_points.append(remove_third_dimension(point))

        return shapely.geometry.MultiPoint(new_points)

    elif isinstance(shply, shapely.geometry.MultiLineString):
        lines = list(shply.geoms)
        new_lines = []
        for line in lines:
            new_lines.append(remove_third_dimension(line))

        return shapely.geometry.MultiLineString(new_lines)

    elif isinstance(shply, shapely.geometry.MultiPolygon):
        pols = list(shply.geoms)

        new_pols = []
        for pol in pols:
            new_pols.append(remove_third_dimension(pol))

        return shapely.geometry.MultiPolygon(new_pols)

    elif isinstance(shply, shapely.geometry.GeometryCollection):
        shplys = list(shply.geoms)

        new_shplys = []
        for shply in shplys:
            new_shplys.append(remove_third_dimension(shply))

        return shapely.geometry.GeometryCollection(new_shplys)

    else:
        raise RuntimeError("Currently this type of geometry is not supported: {}".format(
            type(shply)))


def removeThirdDimension(df):
    df['geometry'] = [_removeThirdDimension(g) for g in df['geometry']]
    return
    

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


def imputeHoles2D(arr, nodata=np.nan, method='cubic'):
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


def computeAverageYear(data, output_nyears=1, smooth=False, **kwargs):
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
        data = smooth_array(data, smooth, axis=0, **kwargs)

    # repeat the data if requested
    if output_nyears != 1:
        tiled_data_shape = (output_nyears, )
        for i in range(len(original_shape) - 1):
            tiled_data_shape = tiled_data_shape + (1, )
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


#
# Geometry
#


def isClose(s1, s2, tol=_tol):
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


def contains(s1, s2, tol=_tol):
    """A contains algorithm that deals with close/roundoff issues"""
    return s1.buffer(tol, 2).contains(s2)


class CutError(Exception):
    def __init__(self, message, line, seg, cutline):
        super(Exception, self).__init__(message)
        self.line = line
        self.seg = seg
        self.cutline = cutline


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
            #logging.debug("  close? = %r"%(isClose(point, seg.coords[-1], tol)))
            if isClose(point, seg.coords[-1], tol):
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
            elif isClose(point, seg.coords[0], tol):
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
            raise CutError(
                "Dual/multiple intersection in a single seg... ugh!  "
                + "Intersection is of type '{}'".format(type(point)), line, seg, cutline)

    if len(segcoords) > 1:
        segs.append(shapely.geometry.LineString(segcoords))
    return segs


def computeDistance(p1, p2):
    """Distance between two points in tuple form"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def inNeighborhood(obj1, obj2, tol=0.1):
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


def intersectPointToSegment(point, line_start, line_end):
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


def findNearestPoint(point, line, tol=None):
    """Returns the nearest coordinate on the line to point.  

    Note point is expected as coordinates."""
    if tol is None:
        if isinstance(point, tuple):
            point = shapely.geometry.Point(point)
        return shapely.ops.nearest_points(point, line)[1].coords[0]
    else:
        if inNeighborhood(shapely.geometry.Point(point), line, tol):
            logging.debug("  - in neighborhood")
            nearest_p = findNearestPoint(point, line, None)
            dist = computeDistance(nearest_p, point)
            logging.debug("  - nearest p = {0}, dist = {1}, tol = {2}".format(nearest_p, dist, tol))
            if dist < tol:
                if dist < 1.e-7:
                    # filter case where the point is already there
                    if any(isClose(point, c) for c in line.coords):
                        return None
                return nearest_p
        return None


def computeTriangleArea(vertices):
    """Area of a triangle in 2D"""
    xy1 = vertices[0]
    xy2 = vertices[1]
    xy3 = vertices[2]

    A = 0.5 * (xy2[0] * xy3[1] - xy3[0] * xy2[1] - xy1[0] * xy3[1] + xy3[0] * xy1[1]
               + xy1[0] * xy2[1] - xy2[0] * xy1[1])
    return A


def isCollinear(p0, p1, p2, tol=1e-6):
    """this function checks if three points are collinear for given tolerance value"""
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1*y2 - x2*y1) < tol


def computeArea(vertices):
    """Area of polygons in 2D"""
    area = shapely.geometry.Polygon(vertices).area
    return area


def computeAngle(v1, v2):
    """Given two 2D vectors represented as len 2 arrays or tuples, find the angle
    of 2 relative to 1 in a clockwise notion."""
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    numer = x1*x2 + y1*y2
    denom = np.sqrt(x1*x1 + y1*y1) * np.sqrt(x2*x2 + y2*y2)
    assert (denom > 0)
    arg = numer / denom
    assert (arg < 1.1 and arg > -1.1)  # roundoff problems
    arg = min(max(numer / denom, -1), 1)
    mag = 180. / np.pi * np.arccos(arg)
    sign = x1*y2 - x2*y1
    if sign < 0:
        return -mag
    else:
        return mag


def computeMidpoint(p1, p2):
    """Returns the midpoint of two points"""
    if isinstance(p1, shapely.geometry.Point):
        return midpoint(p1.coords[0], p2)
    if isinstance(p2, shapely.geometry.Point):
        return midpoint(p1, p2.coords[0])
    return ((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)


def findClosestPointInd(point, points):
    """Returns the index of closest point from an array of points"""
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)


def recenter(objects, centering=True):
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


def computeOrientation(p1, p2, p3):
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


def intersects(shp1, shp2):
    """Checks whether an intersection exists.
    
    Note that intersection being empty and intersects are not always reliably
    the same... we avoid using shapely.intersects() for this reason.
    """
    inter = shp1.intersection(shp2)
    return not isEmptyShapely(inter)


def isNonPointIntersection(shp1, shp2):
    """Checks whether an intersection is larger than a point.
    
    Note that intersection being empty and intersects are not always reliably
    the same... we avoid using intersects() for this reason.
    """
    inter = shp1.intersection(shp2)
    return not (isEmpty(inter) or \
                isinstance(inter, shapely.geometry.Point))


def isVolumetricIntersection(shp1, shp2):
    """Checks whether an intersection includes volume and not just points and lines."""
    inter = shp1.intersection(shp2)
    return inter.area > 0


def filterToShape(shape, to_filter, tol=None, algorithm='intersects'):
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


def isEmpty(shply):
    return shply is None or shply.is_empty

def isConvex(points):
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


def breakSegmentCollinearity(segment_coords, tol=1e-5):
    """This functions removes collinearity from a node segment by making small pertubations orthogonal to the segment"""
    col_checks = []
    for i in range(0,
                   len(segment_coords)
                   - 2):  # traversing along the segment, checking 3 consecutive points at a time
        p0 = segment_coords[i]
        p1 = segment_coords[i + 1]
        p2 = segment_coords[i + 2]
        if watershed_workflow.utils.isCollinear(
                p0, p1, p2, tol=tol):  # treating collinearity through a small pertubation
            del_ortho = 10 * tol  # shift in the middle point
            if (p2[0] - p0[0]) == 0:
                m = 1e6
            else:
                m = (p2[1] - p0[1]) / (p2[0] - p0[0])
            del_y = del_ortho / (1 + m**2)**0.5
            del_x = -1 * del_ortho * m / (1 + m**2)**0.5
            p1 = (p1[0] + del_x, p1[1] + del_y)
            segment_coords[i + 1] = p1
        col_checks.append(isCollinear(p0, p1, p2))
    assert (sum(col_checks) == 0)
    return segment_coords
