import numpy as np
import attr
import sortedcontainers

@attr.s
class Point:
    """POD struct that stores coords, a np array of length 3 (x,y,z) and neighbors, 
    a list of IDs of neighboring points.
    """
    coords = attr.ib()
    neighbors = attr.ib()


def condition1(points, outletID):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is the origional, 2-pass algorithm.
    """

    # create a sorted list of elevations, from largest to smallest
    elev = sorted(list(points.items()), key=lambda id_p:-id_p[1].coords[2])

    visited = set([outletID,])
    pits = set()
    waterway = set([outletID,])

    # loop over elevation list from small to large
    while len(elev) is not 0:
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
    while len(waterway) is not 0:
        current, current_p = waterway.pop()
        for n in current_p.neighbors:
            if n in pits:
                points[n].coords[2] = max(current_p.coords[2], points[n].coords[2])
                pits.remove(n)
                waterway.append( (n,points[n]) )

    # post-conditions
    assert(len(pits) == 0)
    return


def condition2(points, outletID):
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
    while len(elev) is not 0:
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


def condition3(points, outletID):
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
        next_p = boundary.pop()
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
                    
        

def condition(points, outletID, algorithm=3):
    if algorithm == 1:
        return condition1(points, outletID)
    elif algorithm == 2:
        return condition2(points, outletID)
    elif algorithm == 3:
        return condition3(points, outletID)
    else:
        raise RuntimeError('Unknown algorithm "%r"'%(algorithm))

