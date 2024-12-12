"""creates river mesh using quad, pentagon and hexagon elements"""

import numpy as np
import pandas as pd
import logging

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree


def _isOverlappingCorridor(corr, river):
    if len(corr.interiors) > 0:
        # there is an overlap upstream of the junction of two tributaries,
        # creating a hole
        return 2
    n = 0
    if not _isExpectedNumPoints(corr, river, n):
        # overlaps at the junction result in losing points in the corridor polygon.
        return 1
    return 0


def _isOverlappingCorridors(corrs, rivers):
    """Corridors can overlap"""
    if any(_isOverlappingCorridor(c, river) for (c, river) in zip(corrs, rivers)):
        return True

    corrs_area = shapely.ops.unary_union(corrs).area
    summed_area = sum(c.area for c in corrs)
    if abs(summed_area - corrs_area) > 1.e-3:
        return True
    return False


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
    

def _isExpectedNumPoints(corr, river):
    """Check if the points on the corridor are same as calculated theoretically"""
    num_points = _computeExpectedNumPoints(river)
    return len(corr.exterior.coords) == num_points


def createRiverMeshes(rivers, computeWidth):
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
    river.df['elems'] = [None for i in range(len(river.df))]
    for reach in river:
        reach['elems'] = [list() for i in range(len(reach.linestring.coords)-1)]
    

    # project the starting point
    k = 0

    for touch, reach in river.prePostInBetweenOrder():
        width = computeWidth(reach)
        reach_elems = reach['elems']

        if touch == 0:
            # add paddler's right downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = _projectOne(reach.linestring.coords[-2],
                                        reach.linestring.coords[-1],
                                        width)
                reach_elems[-1].append(k)
                k += 1

            # add paddler's right internal points by two-touches projection
            for i in reversed(range(1, len(reach.linestring.coords)-1)):
                coords[k] = _projectTwo(river.linestring.coords[i+1],
                                        river.linestring.coords[i],
                                        river.linestring.coords[i-1],
                                        width)
                reach_elems[i].append(k)
                reach_elems[i-1].append(k)
                k += 1

            if len(reach.children) == 0:
                # add the upstream midpoint
                coords[k] = river.linestring.coords[0]
                reach_elems[0].append(k)
                k += 1

            elif len(reach.children) == 1:
                # project based on two touching segments
                child_width = computeWidth(reach.children[0])
                lwidth = (width + child_width) / 2
                coords[k] = _projectTwo(reach.children[0].linestring.coords[-2],
                                        reach.linestring.coords[0],
                                        reach.linestring.coords[1],
                                        lwidth)
                reach['elems'][0].append(k)
                reach.children[0]['elems'][-1].append(k)
                k += 1

            else:
                # project downstream paddler's right of junction
                coords[k] = _projectJunction(reach, touch)
                reach['elems'][0].append(k)
                reach.children[0]['elems'][-1].append(k)
                k += 1

        if touch == len(reach.children):
            if len(reach.children) == 0:
                pass # no second point

            elif len(reach.children) == 1:
                # project based on two touching segments
                child_width = computeWidth(reach.children[-1])
                lwidth = (width + child_width) / 2
                coords[k] = _projectTwo(reach.linestring.coords[1],
                                        reach.linestring.coords[0],
                                        reach.children[-1].linestring.coords[-2],
                                        lwidth)
                reach['elems'][0].append(k)
                reach.children[-1]['elems'][-1].append(k)
                k += 1

            else:
                # project downstream paddler's left of junction
                coords[k] = _projectJunction(reach, touch)
                reach['elems'][0].append(k)
                reach.children[-1]['elems'][-1].append(k)
                k += 1

            # add paddler's left internal points
            for i in range(1, len(reach.linestring.coords)-1):
                coords[k] = _projectTwo(river.linestring.coords[i-1],
                                        river.linestring.coords[i],
                                        river.linestring.coords[i+1],
                                        width)
                reach_elems[i-1].append(k)
                reach_elems[i].append(k)
                k += 1

            # add paddler's left downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = _projectOne(reach.parent.linestring.coords[-2],
                                        reach.linestring.coords[-1],
                                        width, True)
                reach_elems[-1].append(k)

        else:
            # add the junction point
            coords[k] = _projectJunction(reach, touch)
            reach_elems[0].append(k)
            reach.children[touch-1]['elems'].append(k)
            reach.children[touch]['elems'].append(k)
            k += 1

    # gather elems
    elems = [e for reach in river.postOrder() for e in reach['elems']]
    return elems, coords


def _projectOne(p_up, p, width, is_left=False):
    c_up = np.array(p_up)
    c = np.array(p)

    dp = (c - c_up)
    dp /= np.linalg.norm(dp)
    perp = np.array([dp[1], -dp[0]])
    if is_left:
        perp = -perp

    return p + width * perp


def _projectTwo(p_up, p, p_dn, width):
    # shift p->p_dn right by width
    pass
