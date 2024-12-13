"""creates river mesh using quad, pentagon and hexagon elements"""

import numpy as np
import pandas as pd
import logging

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree


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
    elems: list(list)
        List of river elements, each element a list of indices into
        corr.coords.
    corrs: list(shapely.geometry.Polygon)
        List of river corridor polygons, one per river, storing the
        coordinates used in elems.

    """
    coords = np.nan * np.ones((_computeExpectedNumCoords(river), 2), 'd')
    river.df['elems'] = pd.Series([ [list() for i in range(len(ls.coords)-1)] for ls in river.df.geometry])

    # project the starting point
    k = 0

    for touch, reach in river.prePostInBetweenOrder():
        halfwidth = computeWidth(reach) / 2.
        reach_elems = reach['elems']
        print('PRE:', reach.index, touch, reach_elems)

        if touch == 0:
            # add paddler's right downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = projectOne(reach.linestring.coords[-2],
                                        reach.linestring.coords[-1],
                                        halfwidth)
                reach_elems[-1].append(k)
                k += 1

            # add paddler's right internal points by two-touches projection
            for i in reversed(range(1, len(reach.linestring.coords)-1)):
                logging.info(f'Point {k} based on coords:')
                logging.info(f'   {reach.linestring.coords[i-1]}')
                logging.info(f'   {reach.linestring.coords[i]}')
                logging.info(f'   {reach.linestring.coords[i+1]}')
                coords[k] = projectTwo(reach.linestring.coords[i-1],
                                       reach.linestring.coords[i],
                                       reach.linestring.coords[i+1],
                                       halfwidth, halfwidth)

                logging.info(f'   and distance {halfwidth} --> {coords[k]}')

                reach_elems[i].append(k)
                reach_elems[i-1].append(k)
                k += 1

            # add the upstream point
            if len(reach.children) == 0:
                # add an upstream midpoint
                coords[k] = reach.linestring.coords[0]
                reach_elems[0].append(k)
                k += 1

            elif len(reach.children) == 1:
                # project based on two touching segments
                child_halfwidth = computeWidth(reach.children[0]) / 2.
                coords[k] = projectTwo(reach.children[0].linestring.coords[-2],
                                       reach.linestring.coords[0],
                                       reach.linestring.coords[1],
                                       child_halfwidth, halfwidth)
                reach_elems[0].append(k)
                child_elems = reach.children[0]['elems']
                child_elems[-1].append(k)
                k += 1

            else:
                # project downstream paddler's right of junction
                coords[k] = projectJunction(reach, touch, computeWidth)
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
                reach_elems[0].append(k)
                reach.children[-1]['elems'][-1].append(k)
                k += 1

            else:
                # project downstream paddler's left of junction
                coords[k] = projectJunction(reach, touch, computeWidth)
                reach_elems[0].append(k)
                reach.children[-1]['elems'][-1].append(k)
                k += 1

            # add paddler's left internal points by flipping coords order
            for i in range(1, len(reach.linestring.coords)-1):
                coords[k] = projectTwo(reach.linestring.coords[i+1],
                                       reach.linestring.coords[i],
                                       reach.linestring.coords[i-1],
                                       halfwidth, halfwidth)
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
                reach_elems[-1].append(k)
                k += 1

        if touch != 0 and touch != len(reach.children):
            # add the junction point
            coords[k] = projectJunction(reach, touch, computeWidth)
            reach_elems[0].append(k)
            reach.children[touch-1]['elems'][-1].append(k)
            reach.children[touch]['elems'][-1].append(k)
            k += 1

        print('POST:', reach.index, touch, reach_elems)
    assert k == len(coords)
            
    # gather elems
    elems = [e for reach in river.postOrder() for e in reach['elems']]
    return elems, coords


def computeLine(p1, p2):
    """Compute line coefficients (Ax + By + C = 0) for a line defined by two points"""
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = -(A * p1[0] + B * p1[1])
    return A, B, C


def translateLinePerpendicular(line, distance):
    """
    Translates a line by a specified distance in the direction perpendicular to the line.

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


def findIntersection(line1, line2):
    """Finds the intersection point of two lines given by their coefficients."""
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # Lines are parallel or coincident

    x = (B1 * C2 - B2 * C1) / determinant
    y = (A2 * C1 - A1 * C2) / determinant
    return np.array([x, y])


def projectOne(p_up, p, width):
    c_up = np.array(p_up)
    c = np.array(p)

    dp = (c - c_up)
    dp /= np.linalg.norm(dp)
    perp = np.array([dp[1], -dp[0]])
    return p + width * perp


def projectTwo(p_up, p, p_dn, width1, width2):
    l1 = computeLine(p_up, p)
    l1 = translateLinePerpendicular(l1, width1)
    
    l2 = computeLine(p, p_dn)
    l2 = translateLinePerpendicular(l2, width2)
    intersection = findIntersection(l1, l2)
    if intersection is None:
        return projectOne(p_up, p, (width1 + width2)/2.)
    return intersection


def projectInner(p, p1, p2, width1, width2):
    l1 = computeLine(p1, p)
    l1 = translateLinePerpendicular(l1, -width1)
    l2 = computeLine(p2, p)
    l2 = translateLinePerpendicular(l2, width2)
    return findIntersection(l1,l2)


def projectJunction(reach, child_idx, computeWidth):
    # this is probably insufficient!
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
        return projectInner(reach.linestring.coords[0],
                            reach.children[child_idx-1].linestring.coords[-2],
                            reach.children[child_idx].linestring.coords[-2],
                            computeWidth(reach.children[child_idx-1]) / 2.,
                            computeWidth(reach.children[child_idx]) / 2.)
        
    
