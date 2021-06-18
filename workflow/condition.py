import numpy as np
import attr
import sortedcontainers

@attr.s
class Point:
    """POD struct that stores coords, a np array of length 3 (x,y,z) and neighbors, 
    a list of IDs of neighboring points.
    """
    coords = attr.ib()
    neighbors = attr.ib(factory=set)

def points_from_mesh(m2):
    """Generates a Point dictionary from a surface mesh, for use with fill_pits"""
    points = dict( (i, Point(c)) for (i,c) in enumerate(m2.coords) )
    for conn in m2.conn:
        for c in conn:
            points[c].neighbors.update(conn)
    for i,p in points.items():
        p.neighbors.remove(i)
    return points


def fill_pits1(points, outletID=None):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is the origional, 2-pass algorithm.
    """
    if outletID is None:
        outletID = np.argmin(np.array([points[i].coords[2] for i in range(len(points))]))

    # create a sorted list of elevations, from largest to smallest
    elev = sorted(list(points.items()), key=lambda id_p:-id_p[1].coords[2])

    visited = set([outletID,])
    pits = set()
    waterway = set([outletID,])

    # loop over elevation list from small to large
    while len(elev) != 0:
        current, current_p = elev.pop()
        if current in visited:
            # still in the waterway
            waterway.add(current)
            visited.update(current_p.neighbors)
        else:
            # not in the waterway, add to pits
            pits.add(current)
    
    # post-conditions
    assert(len(pits.union(waterway)) == len(points))

    # loop over waterway and raise up pits as they touch the waterway
    waterway = sorted([ (ID,points[ID]) for ID in waterway], key=lambda id_p:-id_p[1].coords[2])
    while len(waterway) != 0:
        current, current_p = waterway.pop()
        for n in current_p.neighbors:
            if n in pits:
                points[n].coords[2] = max(current_p.coords[2], points[n].coords[2])
                pits.remove(n)
                waterway.append( (n,points[n]) )

    # post-conditions
    assert(len(pits) == 0)
    return


def fill_pits2(points, outletID):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is a refactored, single pass algorithm that leverages a sorted list.
    """

    # create a sorted list of elevations, from largest to smallest
    elev = sortedcontainers.SortedList(list(points.items()), key=lambda id_p:id_p[1].coords[2])
    waterway = set([outletID,])

    # loop over elevation list from small to large
    while len(elev) != 0:
        current, current_p = elev.pop(0)
        if current in waterway:
            # still in the waterway
            waterway.update(current_p.neighbors)
        else:
            # not in the waterway, fill
            ww_neighbors = [n for n in current_p.neighbors if n in waterway]
            if len(ww_neighbors) != 0:
                current_p.coords[2] = min(points[n].coords[2] for n in ww_neighbors)
            else:
                current_p.coords[2] = min(points[n].coords[2] for n in current_p.neighbors)
            
            # push back into elev list with new, higher elevation
            elev.add( (current,current_p) )
    return


def fill_pits3(points, outletID):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is a third algorithm, based on a boundary marching method.
    """
    # Waterway is the list of things that are already conditioned and
    # can be reached.
    waterway = set()

    # Boundary is an elevation-sorted list of (ID, point) tuples which
    # are NOT yet in the waterway, but have a neighbor in the
    # waterway.  Additionally, points in the boundary have been
    # conditioned such that all boundary points must be at least as
    # high as the highest elevation in the waterway.
    boundary = sortedcontainers.SortedList([(outletID,points[outletID]), ], key=lambda id_p:id_p[1].coords[2])
    waterway_max = -1e10

    while len(boundary) > 0:
        # pop the lowest boundary point and stick it in the waterway
        next_p = boundary.pop(0)
        waterway.add(next_p[0])

        # increment the waterway elevation
        assert(next_p[1].coords[2] >= waterway_max)
        waterway_max = next_p[1].coords[2]

        # Insert all neighbors (that aren't in the waterway already)
        # into the boundary, first checking that their elevation is at
        # least as high as the new waterway elevation.  Note that
        # there can be no "alternate" pathway to the waterway, as this
        # pathway would already be in the boundary (somewhere along
        # that pathway) and therefore would have been popped before
        # this path due to our sorted elevation boundary list.
        for n in next_p[1].neighbors:
            if n not in waterway and (n,points[n]) not in boundary:
                points[n].coords[2] = max(points[n].coords[2], waterway_max)
                boundary.add( (n,points[n]) )

    assert(len(waterway) == len(points))
    return
                    
        

def fill_pits(mesh, outlet=None, algorithm=3):
    """Condition a 2D mesh, in place.
    
    Starts at outlet, if not provided, this defaults to the lowpoint on the boundary.

    Available algorithms:
     1: original, 2-pass algorithm
     2: refactored single-pass algorithm based on sorted lists
     3: boundary marching method.  Should be fastest, and likely equivalent?
    """
    points_dict = points_from_mesh(mesh)
    if outlet is None:
        boundary_nodes = mesh.boundary_nodes()
        outlet = boundary_nodes[np.argmin(mesh.coords[boundary_nodes,2])]

    if algorithm == 1:
        fill_pits1(points_dict, outlet)
    elif algorithm == 2:
        fill_pits2(points_dict, outlet)
    elif algorithm == 3:
        fill_pits3(points_dict, outlet)
    else:
        raise RuntimeError('Unknown algorithm "%r"'%(algorithm))

    mesh.points = np.array([p.coords for p in points_dict.values()])
    

def smooth(img_in, algorithm='gaussian', **kwargs):
    """Smooths an image according to an algorithm, passing kwargs on to that algorithm."""
    if algorithm == 'gaussian':
        import scipy.ndimage
        if 'method' not in kwargs:
            kwargs['method'] = 'nearest'
        if 'sigma' not in kwargs:
            sigma = 5
        else:
            sigma = kwargs.pop('sigma')
        return scipy.ndimage.gaussian_filter(img_in, sigma, **kwargs)
    else:
        raise ValueError(f'Unknown smoothing algorithm: "{algorithm}"')


def fill_gaps(img_in, nodata=np.nan):
    import scipy.ndimage
    import scipy.interpolate

    if nodata is np.nan:
        mask = ~np.isnan(img_in)
    else:
        mask = ~(img_in == nodata)

    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(img_in.shape[1]), np.arange(img_in.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values, as 1D arrays (in the same order as their coordinates in xym)
    img_in0 = np.ravel(img_in[:,:][mask])

    # interpolator
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, img_in0)

    # interpolate the whole image
    return interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)


