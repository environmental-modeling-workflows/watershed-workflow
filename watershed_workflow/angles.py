"""This module includes functions to deal with bad angles which can
introduce small triangles or bad quads when meshing.
"""

from typing import List, Optional, Tuple, Any, Iterable, Callable

import logging
import numpy as np
import math
from scipy import interpolate
import shapely
import abc

import watershed_workflow.utils
from watershed_workflow.river_tree import River
from watershed_workflow.split_hucs import SplitHUCs


def smoothSharpAngles(rivers : List[River],
                      hucs : SplitHUCs,
                      min_angle : float,
                      junction_min_angle : Optional[float] = None) -> None:
    """Tries to clean up any sharp angles in the rivers and HUCs."""
    if junction_min_angle is None:
        junction_min_angle = min_angle

    for river in rivers:
        smoothOutletSharpAngles(river, hucs, junction_min_angle)
        
        for reach in river:
            reach.linestring = smoothInternalSharpAngles(reach.linestring, min_angle)

            if reach['inlet'] is not None:
                smoothInletSharpAngles(reach, hucs, junction_min_angle)
            else:
                smoothJunctionSharpAngles(reach, junction_min_angle)

                
def _badInternalAngle(coords : np.ndarray | shapely.geometry.LineString,
                      i : int,
                      min_angle : float) -> bool:
    if isinstance(coords, shapely.geometry.LineString):
        return _badInternalAngle(np.array(coords.coords), i, min_angle)

    angle = watershed_workflow.utils.computeAngle(coords[i-1] - coords[i],
                                                  coords[i+1] - coords[i])
    return (angle < min_angle) or (angle > (360 - min_angle))

def isBadInternalAngle(linestring : shapely.geometry.LineString,
                       min_angle : float):
    """Diagnostic function"""
    coords = np.array(linestring.coords)
    return any(_badInternalAngle(coords, i, min_angle) for i in range(1, len(coords)-1))


def _smoothInternalSharpAngles(linestring : shapely.geometry.LineString,
                          min_angle : float) -> shapely.geometry.LineString:
    """Smooths any internal angle less than min_angle."""
    coords = np.array(linestring.coords)
    if len(coords) < 3:
        return linestring

    new_coords = np.copy(coords)

    if len(coords) == 3:
        # cannot move endpoints, so only the middle point can move
        if _badInternalAngle(coords, 1, min_angle):
            # what is the right condition?  Should we remove the point or move it?
            if watershed_workflow.utils.computeDistance(coords[0], coords[2]) \
               < max(watershed_workflow.utils.computeDistance(coords[0], coords[1]),
                     watershed_workflow.utils.computeDistance(coords[1], coords[2])):
                # just remove the extra point
                return shapely.geometry.LineString([coords[0], coords[2]])
            else:
                # move the point to the triangle centroid
                new_coords[1] = watershed_workflow.utils.computeTriangleCentroid(coords[0], coords[1], coords[2])
                return shapely.geometry.LineString(new_coords)
        else:
            return linestring

    else:
        i = 1
        while i < len(coords)-1:
            if _badInternalAngle(coords, i, min_angle):
                if i == 1:
                    # borders endpoint, cannot recenter the triangle, so instead place new point at midpoint
                    new_coords[i+1,:] = np.nan
                    new_coords[i] = watershed_workflow.utils.computeMidpoint(coords[i], coords[i+1])
                elif i == len(coords) - 2:
                    # borders endpoint, cannot recenter the triangle, so instead place new point at midpoint
                    new_coords[i-1,:] = np.nan
                    new_coords[i] = watershed_workflow.utils.computeMidpoint(coords[i], coords[i-1])
                else:
                    # not bordering endpoint, simply replace all three with the triangle centroid
                    new_coords[[i-1,i+1],:] = np.nan
                    new_coords[i] = watershed_workflow.utils.computeTriangleCentroid(coords[i-1], coords[i], coords[i+1])
                i += 2
            else:
                i += 1

        new_coords2 = new_coords[~np.isnan(new_coords[:,0]),:]
        return shapely.geometry.LineString(new_coords2)


def smoothInternalSharpAngles(linestring : shapely.geometry.LineString,
                              min_angle : float) -> shapely.geometry.LineString:
    """Smooths any internal angle less than min_angle."""
    # note this just iterates until no points are removed
    if min_angle == 0:
        return linestring
    
    done = False
    while not done:
        new_linestring = _smoothInternalSharpAngles(linestring, min_angle)
        done = len(new_linestring.coords) == len(linestring.coords)
    return new_linestring


def _reverseLinestring(ls):
    return shapely.geometry.LineString(reversed(ls.coords))

def _smoothJunctionSharpAnglesHelper(linestrings : List[shapely.geometry.LineString],
                                     min_angle : float) -> None:
    """Given a list of linestrings, all of whose terminal coordinate
    is the same (origin) and whose ordering is clockwise around that
    origin, nudge the second-to-last point of all linestrings (except
    the first) to ensure that all angles are >= min_angle.
    """
    angles = np.array([watershed_workflow.utils.computeAngle(_reverseLinestring(linestrings[i]), linestrings[(i+1) % len(linestrings)]) \
              for i in range(len(linestrings))])
    assert(abs(sum(angles) - 360) < 0.01)

    # compute a valid set of angles, all > min_angle
    itrs = 0
    while min(angles) < min_angle:
        where = np.where(angles < min_angle)[0]
        angles[where] += 2
        angles[(where - 1) % len(angles)] -= 1
        angles[(where + 1) % len(angles)] -= 1
        itrs += 1
        if itrs > 100:
            raise RuntimeError("Cannot smooth sharp angles at junction.")

    assert(abs(sum(angles) - 360) < 0.01)
    assert(min(angles) >= min_angle)

    # now that we have a valid set of angles, we can redistribute the
    # linestrings to those angles by moving the first interior points.
    for i in range(len(linestrings)-1):
        fixed_ls = linestrings[i]
        move_ls = linestrings[i+1]
        v1 = np.array(fixed_ls.coords[-1]) - np.array(fixed_ls.coords[-2])
        dist = watershed_workflow.utils.computeDistance(move_ls.coords[-1], move_ls.coords[-2])
        v2 = watershed_workflow.utils.projectVectorAtAngle(v1, angles[i], dist)

        new_coords = np.array(linestrings[i+1].coords)
        new_coords[-2] = new_coords[-1] - np.array(v2)
        linestrings[i+1] = shapely.geometry.LineString(new_coords)
    return


def getAnglesAtJunction(reach : River):
    """Helper function"""
    assert(len(reach.children) > 0)
    linestrings = [_reverseLinestring(reach.linestring),] \
        + [c.linestring for c in reach.children]
    angles = np.array([watershed_workflow.utils.computeAngle(_reverseLinestring(linestrings[i]),
                                                             linestrings[(i+1) % len(linestrings)]) \
                       for i in range(len(linestrings))])
    assert(abs(sum(angles) - 360) < 0.01)
    return angles


def isBadJunctionSharpAngle(reach : River,
                              min_angle : float) -> bool:
    """Diagnostic function for post-facto checking."""
    if min_angle == 0: return False
    if len(reach.children) == 0: False
    angles = getAnglesAtJunction(reach)
    return min(angles) < min_angle


def smoothJunctionSharpAngles(reach : River,
                              min_angle : float) -> None:
    if min_angle == 0: return
    if len(reach.children) == 0: return

    linestrings = [_reverseLinestring(reach.linestring),] \
        + [c.linestring for c in reach.children]
    _smoothJunctionSharpAnglesHelper(linestrings, min_angle)
    for i,c in enumerate(reach.children):
        c.linestring = linestrings[i+1]
    return 
    

def smoothOutletSharpAngles(reach : River,
                            hucs : SplitHUCs,
                            min_angle : float) -> None:
    if reach.parent is not None:
        raise RuntimeError('Do not call smoothOutletSharpAngles on an internal outlet -- only on domain outlets.')

    # this could be relaxed to have multiple rivers leaving the
    # domain, but that would require better implementation of
    # hydrography.findOutlets()
    assert reach.linestring.coords[0] == hucs.exterior_outlet
    
    # find the segment whose endpoint touches outlet -- there should be exactly two
    huc_touches = []
    for handle in hucs.boundaries:
        #if hucs.linestrings[handle]
        #handle for handle in hucs.boundaries if 
        pass
    #for ls in hucs.exterior


def treatSmallAngleBetweenChildNodes(node : River, angle_limit : float = 10) -> bool:
    """Zippers up junctions that have small angles.  Returns True if a section was removed.

    \  |       \ /
     ||   -->   |
      \          \

    """
    if len(node.children) > 1 and len(node.children) < 3:
        # note, we don't really have a good implementation for 3
        # children.  Could work on one if it shows up...
        ls1 = node.linestring
        angles = []
        for child in node.children:
            ls2 = child.linestring
            ls_up = shapely.geometry.LineString([ls2.coords[-2], ls2.coords[-1]])
            assert (watershed_workflow.utils.isClose(ls2.coords[-1], ls1.coords[0], 1.e-10))
            ls_down = shapely.geometry.LineString([ls1.coords[0], ls1.coords[1]])
            angle = watershed_workflow.river_mesh.angleRiversLss(ref_ls=ls_down, ls=ls_up)
            angles.append(angle)

        if abs(angles[1] - angles[0]) < angle_limit:
            logging.info(
                f"removing sharp angle between children: {abs(angles[1]-angles[0])} for node {node.properties['ID']}"
            )
            # zip up the last section of the two children, and give it to the parent
            new_junction = watershed_workflow.utils.computeMidpoint(node.children[0].linestring.coords[-2],
                                                             node.children[1].linestring.coords[-2])
            new_node_coords = np.array([new_junction, ] + node.linestring.coords[:])
            new_node_coords = watershed_workflow.utils.breakLineStringCollinearity(new_node_coords)
            node.linestring = shapely.geometry.LineString(new_node_coords)

            for child in list(node.children):
                child_coords = child.linestring.coords[:]
                if len(child_coords) > 2:
                    # reach has > 2 points, so we can safely remove the last one
                    child_coords = child_coords[:-1]
                    child_coords[-1] = new_junction
                    child_coords = watershed_workflow.utils.breakLineStringCollinearity(child_coords)
                    child.linestring = shapely.geometry.LineString(child_coords)

                elif len(child_coords) == 2:
                    # reach only has two points
                    # remove it but move children's children to the right point
                    for grandchild in child.children:
                        # update coordinates of children
                        grandchild_ls_coords = grandchild.linestring.coords[:]
                        grandchild_ls_coords[-1] = new_junction
                        grandchild.linestring = shapely.geometry.LineString(grandchild_ls_coords)
                        node.addChild(grandchild)
                    child.remove()

            # we changed things -- return True so this can be called
            # repeatedly until all sections that have the close angle
            # are removed.
            return True
    return False
