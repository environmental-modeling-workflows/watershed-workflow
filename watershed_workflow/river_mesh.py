"""creates river mesh using quad, pentagon and hexagon elements"""

import numpy as np
import pandas as pd
import logging
from typing import Callable
from matplotlib import pyplot as plt

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree
import watershed_workflow.angles
import watershed_workflow.sources.standard_names as names


# def _isOverlappingCorridor(corr, river):
#     if len(corr.interiors) > 0:
#         # there is an overlap upstream of the junction of two tributaries,
#         # creating a hole
#         return 2
#     n = 0
#     if not _isExpectedNumPoints(corr, river, n):
#         # overlaps at the junction result in losing points in the corridor polygon.
#         return 1
#     return 0


# def _isOverlappingCorridors(corrs, rivers):
#     """Corridors can overlap"""
#     if any(_isOverlappingCorridor(c, river) for (c, river) in zip(corrs, rivers)):
#         return True

#     corrs_area = shapely.ops.unary_union(corrs).area
#     summed_area = sum(c.area for c in corrs)
#     if abs(summed_area - corrs_area) > 1.e-3:
#         return True
#     return False


def _computeExpectedNumCoords(river):
    """Compute the number of expected coordinates."""
    # two outlet points    
    n = 2

    # internal points
    n += sum(2 * (len(reach.linestring.coords) - 2) for reach in river)

    # endpoints
    n += sum(len(reach.children)+1 for reach in river)
    return n

def _computeExpectedNumElems(river):
    return sum(len(reach.linestring.coords)-1 for reach in river)


def createWidthFunction(arg):
    if isinstance(arg, dict):
        def func(reach):
            return arg[reach[names.ORDER]]
    elif isinstance(arg, Callable):
        func = arg
    else:
        def func(reach):
            return arg

    return func


def createRiversMesh(hucs, rivers, computeWidth):
    """Creates meshes for each river and merges them."""
    elems = []
    coords = []
    corridors = []
    hole_points = []
    i = 0
    for river in rivers:
        # create the mesh
        lcoords, lelems = createRiverMesh(river, computeWidth)

        # adjust the HUC linestrings to include the small cross-stream
        # segment
        adjustHUCsToRiverMesh(hucs, river, lcoords)

        # hole point is the centroid of the outlet element
        hole_points.append(lcoords[lelems[-1]].mean(axis=0))

        # corridor is the trace of the outside
        corridors.append(shapely.geometry.Polygon(lcoords))
        
        # shift to get a global ordering
        if i != 0:
            lelems = [[j+i for j in e] for e in lelems]

        elems.extend(lelems)
        coords.append(lcoords)
        i += len(lcoords)

    return np.concatenate(coords), elems, corridors, hole_points
        
        
def createRiverMesh(river, computeWidth):
    """Returns list of elems and river corridor polygons for a given list of river trees

    Parameters:
    -----------
    rivers: list(watershed_workflow.river_tree.River object)
        List of river tree along which river meshes are to be created
    widths: float or dict or callable or boolean 
       Width of the quads, either a float or a dictionary providing a
       {StreamOrder : width} mapping.
       Or a function (callable) that computer width using node properties
       Or boolean, where True means, width for each reach is explicitely provided properties as "width"
    enforce_convexity: boolean 
        If true, enforce convexity of the pentagons/hexagons at the
        junctions.
    ax : matplotlib Axes object, optional
        For debugging -- plots troublesome reaches as quad elements are
        generated to find tricky areas.
    label : bool, optional = True
        If true and ax is provided, animates the debugging plot with
        reach ID labels as the user hovers over the plot.  Requires a
        widget backend for matplotlib.
    
    Returns
    -------
    corrs: list(shapely.geometry.Polygon)
        List of river corridor polygons, one per river, storing the
        coordinates used in elems.
    elems: list(list)
        List of river elements, each element a list of indices into
        corr.coords.

    """
    coords = np.nan * np.ones((_computeExpectedNumCoords(river), 2), 'd')
    river.df['elems'] = pd.Series([ [list() for i in range(len(ls.coords)-1)] for ls in river.df.geometry], index=river.df.index)

    # project the starting point
    k = 0

    for touch, reach in river.prePostInBetweenOrder():
        halfwidth = computeWidth(reach) / 2.
        reach_elems = reach['elems']
        #logging.debug(f'PRE: reach = {reach.index}, touch = {touch}, elems = {reach_elems}')

        if touch == 0:
            # add paddler's right downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = projectOne(reach.linestring.coords[-2],
                                        reach.linestring.coords[-1],
                                        halfwidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} outlet right")
                reach_elems[-1].append(k)
                k += 1

            # add paddler's right internal points by two-touches projection
            for i in reversed(range(1, len(reach.linestring.coords)-1)):
                coords[k] = projectTwo(reach.linestring.coords[i-1],
                                       reach.linestring.coords[i],
                                       reach.linestring.coords[i+1],
                                       halfwidth, halfwidth, k==4)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} interal right")
                reach_elems[i].append(k)
                reach_elems[i-1].append(k)
                k += 1

            # add the upstream point
            if len(reach.children) == 0:
                # add an upstream midpoint
                coords[k] = reach.linestring.coords[0]
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} leaf tip")
                reach_elems[0].append(k)
                k += 1

            elif len(reach.children) == 1:
                # project based on two touching segments
                child_halfwidth = computeWidth(reach.children[0]) / 2.
                coords[k] = projectTwo(reach.children[0].linestring.coords[-2],
                                       reach.linestring.coords[0],
                                       reach.linestring.coords[1],
                                       child_halfwidth, halfwidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} inline child upstream right")
                reach_elems[0].append(k)
                child_elems = reach.children[0]['elems']
                child_elems[-1].append(k)
                k += 1

            else:
                # project downstream paddler's right of junction
                coords[k] = projectJunction(reach, touch, computeWidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} junction child upstream right")
                reach_elems[0].append(k)
                reach.children[0]['elems'][-1].append(k)
                k += 1

        if touch == len(reach.children):
            if len(reach.children) == 0:
                pass # no second point

            elif len(reach.children) == 1:
                # project based on two touching segments
                child_halfwidth = computeWidth(reach.children[-1]) / 2.
                coords[k] = projectTwo(reach.linestring.coords[1],
                                       reach.linestring.coords[0],
                                       reach.children[-1].linestring.coords[-2],
                                       halfwidth, child_halfwidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} inline child upstream left")
                reach_elems[0].append(k)
                reach.children[-1]['elems'][-1].append(k)
                k += 1

            else:
                # project downstream paddler's left of junction
                coords[k] = projectJunction(reach, touch, computeWidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} junction child upstream left")
                reach_elems[0].append(k)
                reach.children[-1]['elems'][-1].append(k)
                k += 1

            # add paddler's left internal points by flipping coords order
            for i in range(1, len(reach.linestring.coords)-1):
                coords[k] = projectTwo(reach.linestring.coords[i+1],
                                       reach.linestring.coords[i],
                                       reach.linestring.coords[i-1],
                                       halfwidth, halfwidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} internal left")
                reach_elems[i-1].append(k)
                reach_elems[i].append(k)
                k += 1

            # add paddler's left downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = projectOne(reach.linestring.coords[-2],
                                        reach.linestring.coords[-1],
                                        -halfwidth)
                #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} outlet left")
                reach_elems[-1].append(k)
                k += 1

        if touch != 0 and touch != len(reach.children):
            # add the junction point
            coords[k] = projectJunction(reach, touch, computeWidth)
            #logging.debug(f" -- adding coord {k} = {coords[k]} as {reach.index} junction midpoint")
            reach_elems[0].append(k)
            reach.children[touch-1]['elems'][-1].append(k)
            reach.children[touch]['elems'][-1].append(k)
            k += 1

        #logging.debug(f'POST: reach = {reach.index}, touch = {touch}, elems = {reach_elems}')
    assert k == len(coords)
            
    # check convexity
    for reach in river:
        for elem in reach['elems']:
            ecoords = coords[elem]
            if not watershed_workflow.utils.isConvex(ecoords):
                fig, ax = plt.subplots(1,1)

                reaches = [reach, reach.parent] + list(reach.children)
                for r in reaches:
                    ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'r-x')

                poly = shapely.geometry.Polygon(ecoords)
                ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'b-x')
                plt.show()
                assert False

    # gather elems
    elems = [e for reach in river.postOrder() for e in reach['elems']]
    assert len(elems) == _computeExpectedNumElems(river)

    return coords, elems


def adjustHUCsToRiverMesh(hucs, river, coords):
    """Adjust HUC segments that touch reach endpoints to match the corridor coordinates."""

    # downstream the river outlet
    remerge, touches = watershed_workflow.angles._getOutletLinestrings(hucs, river)

    if len(touches) > 0:
        assert len(touches) == 3
        # touches[1] is paddler's left
        left_old_coords = touches[1][1].coords
        left_new_coord = coords[river['elems'][-1][-1]]
        left_new_ls = shapely.geometry.LineString(left_old_coords[:-1] + [left_new_coord,])

        right_old_coords = touches[2][1].coords
        right_new_coord = coords[river['elems'][-1][0]]
        right_new_ls = shapely.geometry.LineString(right_old_coords[:-1] + [right_new_coord,])

        if remerge:
            new_ls = shapely.geometry.LineString(list(reversed(left_new_ls.coords)) + list(right_new_ls.coords[1:]))
            hucs.linestrings[touches[1][0]] = new_ls
        else:
            hucs.linestrings[touches[1][0]] = watershed_workflow.utils.reverseLineString(left_new_ls) if touches[1][2] else left_new_ls
            hucs.linestrings[touches[2][0]] = watershed_workflow.utils.reverseLineString(right_new_ls) if touches[2][2] else right_new_ls

    # adjust all upstream endpoints
    for reach in river:
        touches = watershed_workflow.angles._getUpstreamLinestrings(hucs, reach)
        if len(touches) > len(reach.children)+1:
            # yes, there are junctions involved...
            point_i = 1
            touch_i = 1
            while touch_i < len(touches):
                if touches[touch_i][0] >= 0:
                    # make sure touches before and after are reaches
                    #
                    # This will fail if there are two successive HUC
                    # strings.  I'm not sure that should ever happen,
                    # but if it does, we would have to choose which
                    # point to put on the reach junction element
                    # coordinate, and wierd stuff would probably
                    # happen in triangulation anyway.
                    assert touches[touch_i-1][0] is None or touches[touch_i-1][0] < 0
                    assert touches[(touch_i+1)%len(touches)][0] is None or touches[(touch_i+1)%len(touches)][0] < 0
                    
                    # it is a HUC, insert the point
                    new_coord = coords[reach['elems'][0][point_i]]
                    old_coords = touches[touch_i][1].coords
                    new_ls = shapely.geometry.LineString(old_coords[:-1] + [new_coord,])
                    hucs.linestrings[touches[touch_i][0]] = new_ls
                    point_i += 1
                touch_i += 1
    

def computeLine(p1, p2):
    """Compute line coefficients (Ax + By + C = 0) for a line defined by two points"""
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = -(A * p1[0] + B * p1[1])
    return A, B, C


def translateLinePerpendicular(line, distance):
    """Translates a line by a specified distance in the direction
    perpendicular to the line.

    Args:
        line: Tuple of line coefficients (A, B, C).
        distance: Scalar distance to translate the line.

    Returns:
        Tuple of new line coefficients (A, B, C).

    """
    A, B, C = line
    # Normalize A and B to get the unit normal vector
    normal_length = np.hypot(A, B)
    if normal_length == 0:
        raise ValueError("Invalid line coefficients.")

    # Translate C by the perpendicular distance
    C_new = C - distance * normal_length
    return A, B, C_new


def findIntersection(line1, line2, debug=False):
    """Finds the intersection point of two lines given by coefficients."""
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    determinant = A1 * B2 - A2 * B1
    if debug:
        logging.debug(f"  Parallel?  det = {determinant}")
    if abs(determinant) < 1.e-6:
        return None  # Lines are parallel or coincident

    x = (B1 * C2 - B2 * C1) / determinant
    y = (A2 * C1 - A1 * C2) / determinant
    return np.array([x, y])


def projectOne(p_up, p, width):
    """Find a point p_out that is width away from p and such that
    p_up --> p is right-perpendicular to p --> p_out."""
    c_up = np.array(p_up)
    c = np.array(p)

    dp = (c - c_up)
    dp /= np.linalg.norm(dp)
    perp = np.array([dp[1], -dp[0]])
    return p + width * perp


def projectTwo(p_up, p, p_dn, width1, width2, debug=False):
    """Find a point that is "perpendicular" to the linestring p_up -->
    p --> p_dn by intersecting two lines, one of which is parallel to
    p_up --> p and width1 away, and one of which is parallel to p -->
    p_dn and width2 away.
    """
    l1 = computeLine(p_up, p)
    l1 = translateLinePerpendicular(l1, width1)
    
    l2 = computeLine(p, p_dn)
    l2 = translateLinePerpendicular(l2, width2)

    intersection = findIntersection(l1, l2, debug)
    if intersection is None:
        return projectOne(p_up, p, (width1 + width2)/2.)
    return intersection


def projectJunction(reach, child_idx, computeWidth):
    """Find points around the junction between reach and its children."""
    if child_idx == 0:
        return projectTwo(reach.children[0].linestring.coords[-2],
                          reach.linestring.coords[0],
                          reach.linestring.coords[1],
                          computeWidth(reach.children[0]) / 2.,
                          computeWidth(reach) / 2.)
    elif child_idx == len(reach.children):
        return projectTwo(reach.linestring.coords[1],
                          reach.linestring.coords[0],
                          reach.children[-1].linestring.coords[-2],
                          computeWidth(reach) / 2.,
                          computeWidth(reach.children[-1]) / 2.)
    else:
        return projectTwo(reach.children[child_idx].linestring.coords[-2],
                          reach.linestring.coords[0],
                          reach.children[child_idx-1].linestring.coords[-2],
                          computeWidth(reach.children[child_idx]) / 2.,
                          computeWidth(reach.children[child_idx-1]) / 2.)
                          
