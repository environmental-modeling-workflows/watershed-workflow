"""creates river mesh using quad, pentagon and hexagon elements"""

import numpy as np
import pandas as pd
import logging
from typing import Callable, List, Tuple, Dict, Optional

from matplotlib import pyplot as plt
import matplotlib.axes

import geopandas as gpd

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree
import watershed_workflow.angles
from watershed_workflow.river_tree import River
from watershed_workflow.split_hucs import SplitHUCs
import watershed_workflow.sources.standard_names as names


def _isNonoverlapping(points : np.ndarray,
                      elems : List[List[int]],
                      tol : float = 1) -> bool:
    """Are a set of shapes nonoverlapping?"""
    shps = [shapely.geometry.Polygon(points[e]) for e in elems]
    total_area = shapely.unary_union(shps).area
    summed_area = sum(shp.area for shp in shps)
    return abs(total_area - summed_area) < tol


def _computeExpectedNumCoords(river : River) -> int:
    """Compute the number of expected coordinates."""
    # two outlet points    
    n = 2

    # internal points
    n += sum(2 * (len(reach.linestring.coords) - 2) for reach in river)

    # endpoints
    n += sum(len(reach.children)+1 for reach in river)
    return n

def _computeExpectedNumElems(river : River) -> int:
    return sum(len(reach.linestring.coords)-1 for reach in river)
    

def createWidthFunction(arg : Dict[int, float] | Callable[[int,], float] | float) -> Callable[[int,], float]:
    if isinstance(arg, dict):
        def func(reach):
            return arg[reach[names.ORDER]]
    elif callable(arg):
        func = arg
    else:
        def func(reach):
            return arg
    return func


def _plotRiver(river : River,
               coords : np.ndarray,
               ax : matplotlib.axes.Axes) -> None:
    """Plot the river and elements for a debugging plot"""
    river.plot(color='b', marker='+', ax=ax)

    elems = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon(coords[elem]) for reach in river for elem in reach[names.ELEMS]])
    watershed_workflow.plot.linestringsWithCoords(elems.boundary, color='g', marker='x', ax=ax)


def createRiversMesh(hucs : SplitHUCs,
                     rivers : List[River],
                     computeWidth : Callable[[int,], float],
                     ax : Optional[matplotlib.axes.Axes] = None) -> \
                     Tuple[np.ndarray,
                           List[List[int]],
                           List[shapely.geometry.Polygon],
                           List[shapely.geometry.Point],
                           gpd.GeoDataFrame | None,
                           ]:
    """Creates meshes for each river and merges them."""
    elems : List[List[int]] = []
    coords : List[np.ndarray] = []
    corridors : List[shapely.geometry.Polygon] = []
    hole_points : List[shapely.geometry.Point] = []
    i = 0
    elems_gid_start = 0
    
    for river in rivers:
        # create the mesh
        lcoords, lelems = createRiverMesh(river, computeWidth, elems_gid_start)
        elems_gid_start += len(lelems)

        # adjust the HUC linestrings to include the small cross-stream
        # segment
        adjustHUCsToRiverMesh(hucs, river, lcoords)

        if ax is not None:
            logging.info('Plotting the river mesh')
            _plotRiver(river, lcoords, ax)

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

    if ax is not None:
        hucs.plotAsLinestrings(color='k', marker='x', ax=ax)

    all_coords = np.concatenate(coords)
        
    if not _isNonoverlapping(all_coords, elems):
        logging.warning(f'Found at least one intersection overlapping elements in the river mesh... searching for the first intersection now')

        # find overlaps
        # -- create reach polygons
        reach_polys = [shapely.unary_union([shapely.geometry.Polygon([all_coords[e] for e in elem]) for elem in reach[names.ELEMS]])
                       for river in rivers for reach in river]
        reach_ids = [reach[names.ID] for river in rivers for reach in river]

        # -- find pairwise intersections
        intersection_i = []
        intersection_j = []
        intersection_p = []
        for i in range(0, len(reach_polys) - 1):
            for j in range(i+1, len(reach_polys)):
                if reach_polys[i].intersection(reach_polys[j]).area > 0:
                    intersection_i.append(reach_ids[i])
                    intersection_j.append(reach_ids[j])
                    intersection_p.append(reach_polys[i].intersection(reach_polys[j]))
        
        intersections_df = gpd.GeoDataFrame(data={'i': intersection_i,
                                                  'j': intersection_j,},
                                            geometry=intersection_p,
                                            crs=hucs.crs)
        if ax is not None:
            intersections_df.plot(color='k', ax=ax)

    else:
        intersections_df = None
        
    return all_coords, elems, corridors, hole_points, intersections_df
        
        
def createRiverMesh(river : River,
                    computeWidth : Callable[[int,], float],
                    elems_gid_start : int = 0):
    """Returns list of elems and river corridor polygons for a given list of river trees

    Parameters:
    -----------
    rivers: list(River object)
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
    river.df[names.ELEMS] = pd.Series([ [list() for i in range(len(ls.coords)-1)] for ls in river.df.geometry], index=river.df.index)

    # project the starting point
    # k tracks the index of the point/coordinate
    k = 0

    debug : Tuple[int | None,int | None] = None, None  # reach index, coordinate index
    if debug[0] != None and debug[1] != None:
        node = river.getNode(debug[0])
        if node is not None:
            logging.info(f"Debugging reach {debug[0]}, coordinate {debug[1]}, at {node.linestring.coords[debug[1]]}")
    
    for touch, reach in river.prePostInBetweenOrder():
        halfwidth = computeWidth(reach) / 2.
        reach_elems = reach[names.ELEMS]

        if debug[0] == reach.index:
            logging.info(f'PRE: reach = {reach.index}, touch = {touch}, elems = {reach_elems}')

        if touch == 0:
            # add paddler's right downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = projectOne(reach.linestring.coords[-2],
                                        reach.linestring.coords[-1],
                                        halfwidth)
                if reach.index == debug[0] and (-1 == debug[1] or len(reach.linestring.coords)-1 == debug[1]):
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} outlet right")
                reach_elems[-1].append(k)
                k += 1

            # add paddler's right internal points by two-touches projection
            for i in reversed(range(1, len(reach.linestring.coords)-1)):
                coords[k] = projectTwo(reach.linestring.coords[i-1],
                                       reach.linestring.coords[i],
                                       reach.linestring.coords[i+1],
                                       halfwidth, halfwidth,
                                       reach.index == debug[0] and i == debug[1])
                if reach.index == debug[0] and i == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} internal right")
                reach_elems[i].append(k)
                reach_elems[i-1].append(k)
                k += 1

            # add the upstream point
            if len(reach.children) == 0:
                # add an upstream triangle tip at stream midpoint
                coords[k] = reach.linestring.coords[0]
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} leaf tip")
                reach_elems[0].append(k)
                k += 1

            elif len(reach.children) == 1:
                # add an upstream, paddler's right point based on inline junction of two reaches
                child_halfwidth = computeWidth(reach.children[0]) / 2.
                coords[k] = projectTwo(reach.children[0].linestring.coords[-2],
                                       reach.linestring.coords[0],
                                       reach.linestring.coords[1],
                                       child_halfwidth, halfwidth,
                                       reach.index == debug[0] and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} inline child upstream right")
                reach_elems[0].append(k)
                child_elems = reach.children[0][names.ELEMS]
                child_elems[-1].append(k)
                k += 1

            else:
                # add an upstream, paddler's right point based on junction of multiple reaches
                coords[k] = projectJunction(reach, touch, computeWidth, reach.index == debug[0] and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} junction child upstream right")
                reach_elems[0].append(k)
                reach.children[0][names.ELEMS][-1].append(k)
                k += 1

        if touch == len(reach.children):
            if len(reach.children) == 0:
                pass # no second point

            elif len(reach.children) == 1:
                # add an upstream, paddler's left pont based on inline junction of two reaches
                child_halfwidth = computeWidth(reach.children[-1]) / 2.
                coords[k] = projectTwo(reach.linestring.coords[1],
                                       reach.linestring.coords[0],
                                       reach.children[-1].linestring.coords[-2],
                                       halfwidth, child_halfwidth,
                                       reach.index == debug[0] and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} inline child upstream left")
                reach_elems[0].append(k)
                reach.children[-1][names.ELEMS][-1].append(k)
                k += 1

            else:
                # add an upstream, paddler's left point based on junction of multiple reaches
                coords[k] = projectJunction(reach, touch, computeWidth, reach.index == debug[0] and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} junction child upstream left")
                reach_elems[0].append(k)
                reach.children[-1][names.ELEMS][-1].append(k)
                k += 1

            # add a paddler's left internal point
            for i in range(1, len(reach.linestring.coords)-1):
                coords[k] = projectTwo(reach.linestring.coords[i+1],
                                       reach.linestring.coords[i],
                                       reach.linestring.coords[i-1],
                                       halfwidth, halfwidth,
                                       (reach.index == debug[0] and i == debug[1]))
                if reach.index == debug[0] and i == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} internal left")
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
                if reach.index == debug[0] and (-1 == debug[1] or len(reach.linestring.coords)-1 == debug[1]):
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} outlet left")
                reach_elems[-1].append(k)
                k += 1

        if touch != 0 and touch != len(reach.children):
            # add a mid-tributary junction point
            coords[k] = projectJunction(reach, touch, computeWidth, reach.index == debug[0] and 0 == debug[1])
            if reach.index == debug[0] and 0 == debug[1]:
                logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} junction midpoint")
            reach_elems[0].append(k)
            reach.children[touch-1][names.ELEMS][-1].append(k)
            reach.children[touch][names.ELEMS][-1].append(k)
            k += 1

        if debug[0] == reach.index:
            logging.info(f'POST: reach = {reach.index}, touch = {touch}, elems = {reach_elems}')

    assert k == len(coords)
            
    # check convexity
    for reach in river:
        for k, elem in enumerate(reach[names.ELEMS]):
            e_coords = coords[elem]
            if not watershed_workflow.utils.isConvex(e_coords):
                assert k == 0
                new_e_coords = fixConvexity(reach, e_coords, computeWidth)
                for c_index, coord in zip(elem, new_e_coords):
                    coords[c_index] = coord
                
    # gather elems
    elems = [e for reach in river.postOrder() for e in reach[names.ELEMS]]
    assert len(elems) == _computeExpectedNumElems(river)

    # assign GID to each elem start
    # note this must be done in the same order as above elems
    if names.ELEMS_GID_START not in river.df.columns:
        river.df[names.ELEMS_GID_START] = -np.ones(len(river.df), 'i')

    for reach in river.postOrder():
        reach[names.ELEMS_GID_START] = elems_gid_start
        elems_gid_start += len(reach[names.ELEMS])

    return coords, elems


def adjustHUCsToRiverMesh(hucs, river, coords):
    """Adjust HUC segments that touch reach endpoints to match the corridor coordinates."""

    # downstream the river outlet
    remerge, touches = watershed_workflow.angles._getOutletLinestrings(hucs, river)

    if len(touches) > 0:
        assert len(touches) == 3
        logging.info(f"Adjusting HUC to match reaches at outlet")
        # touches[1] is paddler's left
        left_old_coords = touches[1][1].coords
        left_new_coord = coords[river[names.ELEMS][-1][-1]]
        left_new_ls = shapely.geometry.LineString(left_old_coords[:-1] + [left_new_coord,])

        right_old_coords = touches[2][1].coords
        right_new_coord = coords[river[names.ELEMS][-1][0]]
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
            logging.info(f"Adjusting HUC to match reaches at reach {reach.index} and coordinate {reach.linestring.coords[0]}")
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
                    assert touches[touch_i-1][0] is None or touches[touch_i-1][0] < 0, \
                        f"Neighboring touch at reach {reach.index} coords {reach.linestring.coords[0]} is wierd"
                    
                    assert touches[(touch_i+1)%len(touches)][0] is None or touches[(touch_i+1)%len(touches)][0] < 0, \
                        f"Neighboring touch at reach {reach.index} coords {reach.linestring.coords[0]} is wierd"

                    
                    # it is a HUC, insert the point
                    new_coord = coords[reach[names.ELEMS][0][point_i]]
                    old_coords = touches[touch_i][1].coords
                    new_ls = shapely.geometry.LineString(old_coords[:-1] + [new_coord,])
                    hucs.linestrings[touches[touch_i][0]] = new_ls
                else:
                    point_i += 1

                touch_i += 1
    

def computeLine(p1, p2):
    """Compute line coefficients (Ax + By + C = 0) for a line defined by two points"""
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p2[0]*p1[1] - p1[0]*p2[1]
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

    # shift the intercept to near the origin
    determinant = A1 * B2 - A2 * B1
    eps = 1.e-8 * max(abs(A1), abs(A2), abs(B1), abs(B2), abs(C1), abs(C2))
    if debug:
        logging.info(f"  Parallel?  det = {determinant} relative {eps}")
    if abs(determinant) < eps:
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
    if debug:
        logging.info(f"line defined by: {p_up} --> {p} = {l1}")
    l1 = translateLinePerpendicular(l1, width1)
    if debug:
        logging.info(f"  translated by {width1} = {l1}")
    
    l2 = computeLine(p, p_dn)
    if debug:
        logging.info(f"line defined by: {p} --> {p_dn} = {l2}")
    l2 = translateLinePerpendicular(l2, width2)
    if debug:
        logging.info(f"  translated by {width2} = {l2}")

    intersection = findIntersection(l1, l2, debug)
    if intersection is None:
        new_p = projectOne(p_up, p, (width1 + width2)/2.)
        if debug:
            logging.info(f"parallel! results in intersection = {new_p}")
        return new_p

    elif watershed_workflow.utils.computeDistance(intersection, p) > 4*max(abs(width1), abs(width2)):
        # a degenerative point often caused by differing widths and nearly parallel lines
        new_p1 = projectOne(p_up, p, width1)
        new_p2 = projectOne(p_dn, p, -width2)
        new_p = watershed_workflow.utils.computeMidpoint(new_p1, new_p2)
        if debug:
            logging.info(f"results in failure of intersection, computing midpoint = {new_p}")
        return new_p

    else:
        if debug:
            logging.info(f"results in intersection = {intersection}")
        return intersection


def projectJunction(reach, child_idx, computeWidth, debug = False):
    """Find points around the junction between reach and its children."""
    if child_idx == 0:
        return projectTwo(reach.children[0].linestring.coords[-2],
                          reach.linestring.coords[0],
                          reach.linestring.coords[1],
                          computeWidth(reach.children[0]) / 2.,
                          computeWidth(reach) / 2.,
                          debug)
    elif child_idx == len(reach.children):
        return projectTwo(reach.linestring.coords[1],
                          reach.linestring.coords[0],
                          reach.children[-1].linestring.coords[-2],
                          computeWidth(reach) / 2.,
                          computeWidth(reach.children[-1]) / 2.,
                          debug)
    else:
        return projectTwo(reach.children[child_idx].linestring.coords[-2],
                          reach.linestring.coords[0],
                          reach.children[child_idx-1].linestring.coords[-2],
                          computeWidth(reach.children[child_idx]) / 2.,
                          computeWidth(reach.children[child_idx-1]) / 2.,
                          debug)
                          

def fixConvexity(reach, e_coords, computeWidth):
    """Snap e_coords onto the convex hull of e_coords, but respect the upstream stream width."""
    e_poly = shapely.geometry.Polygon(e_coords)
    e_poly_hull = e_poly.convex_hull

    # snap to convex hull
    fix_points = []
    for i, coord in enumerate(e_coords):
        closest_p = watershed_workflow.utils.findNearestPoint(shapely.geometry.Point(coord), e_poly_hull.boundary)

        if not watershed_workflow.utils.isClose(coord, closest_p, 1.e-4):
            fix_points.append(i)
            
            if i == 1 or i == len(e_coords) - 2:
                # intersect the shifted line with the convex hull to find the new point
                if i == 1:
                    child = reach.children[0]
                    sign = 1
                elif i == len(e_coords) - 2:
                    child = reach.children[-1]
                    sign = -1

                p0 = child.linestring.coords[-1]
                p1 = child.linestring.coords[-2]
                halfwidth = computeWidth(child)/2.
                p0p = projectOne(p1, p0, sign * halfwidth)
                p1p = projectOne(p0, p1, -sign * halfwidth)
                ls = shapely.geometry.LineString([p0p, p1p])
                if not ls.intersects(e_poly_hull.boundary):
                    fig, ax = plt.subplots(1,1)

                    reaches = [reach, reach.parent] + list(reach.children)
                    for r in reaches:
                        ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

                    poly = shapely.geometry.Polygon(e_coords)
                    ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')

                    ax.plot(ls.xy[0], ls.xy[1], 'r-x')
                    ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
                    ax.set_aspect('equal', adjustable='box')
                    plt.show()
                    assert False, "No intersection point with convex hull?"

                new_c_p = ls.intersection(e_poly_hull.boundary)
                assert isinstance(new_c_p, shapely.geometry.Point)
                new_c = new_c_p.coords[0]

            elif i > 1 and i < len(e_coords) - 2:
                # snap it to the nearest point?
                new_c = closest_p

            else:
                # a 0th or last point should never be non-convex?
                fig, ax = plt.subplots(1,1)

                reaches = [reach, reach.parent] + list(reach.children)
                for r in reaches:
                    ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

                poly = shapely.geometry.Polygon(e_coords)
                ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')
                ax.scatter([coord[0],], [coord[1],], color='g', marker='o')
                ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
                ax.set_aspect('equal', adjustable='box')
                plt.show()

                assert False, "Do not know how to deal with non-convexity that doesn't start at the outermost children."

            e_coords[i] = new_c

    if not watershed_workflow.utils.isConvex(e_coords):
        # a 0th or last point should never be non-convex?
        fig, ax = plt.subplots(1,1)

        reaches = [reach, reach.parent] + list(reach.children)
        for r in reaches:
            ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

        poly = shapely.geometry.Polygon(e_coords)
        ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')

        for i in fix_points:
            ax.scatter([e_coords[i,0],], [e_coords[i,1],], color='g', marker='o')
            
        ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        assert False, "Cannot fix nonconvexity?"


    return e_coords
    
