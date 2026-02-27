"""This module includes functions to deal with bad angles which can
introduce small triangles or bad quads when meshing.
"""

from typing import List, Optional, Tuple, Any, Iterable, Callable

import logging
import numpy as np
import math
from scipy import interpolate
from scipy.spatial import cKDTree
import shapely
import abc
from matplotlib import pyplot as plt

import watershed_workflow.utils
import watershed_workflow.colors
from watershed_workflow.utils import reverseLineString
from watershed_workflow.river_tree import River
from watershed_workflow.split_hucs import SplitHUCs
from watershed_workflow.sources import standard_names as names


def _getAngles(linestrings : List[shapely.geometry.LineString]) -> np.ndarray:
    """Given a list of linestrings whose terminal coordinate is the same, compute angles between each successive pair of strings."""
    # check preconditions
    assert len(linestrings) > 1
    assert all(watershed_workflow.utils.isClose(linestrings[0].coords[-1], ls.coords[-1]) for ls in linestrings)

    angles = np.array([watershed_workflow.utils.computeAngle(reverseLineString(linestrings[i]),
                                                             linestrings[(i+1) % len(linestrings)]) \
              for i in range(len(linestrings))])

    # check postconditions
    assert(abs(sum(angles) - 360) < 0.01)
    assert(min(angles) > 0)
    return angles
    

def isSharpAngles(hucs : SplitHUCs,
                  rivers : List[River],
                  min_angle = float,
                  junction_min_angle : Optional[float] = None) -> bool:
    """Posterior check for any sharp angles.

    Don't call this first -- just call smoothSharpAngles().  This is
    really for testing and confirming that the algorithm worked -- it
    does almost as much work as smoothSharpAngles().
    """
    if junction_min_angle is None:
        junction_min_angle = min_angle

    if isHUCsSharpAngle(hucs, min_angle, junction_min_angle):
        return True

    for river in rivers:
        if isOutletSharpAngle(hucs, river, junction_min_angle):
            return True
        for reach in river:
            if isInternalSharpAngle(reach.linestring, min_angle):
                return True
            if isUpstreamSharpAngle(hucs, reach, junction_min_angle):
                return True
    return False
    

def smoothSharpAngles(hucs : SplitHUCs,
                      rivers : List[River],                      
                      min_angle : float,
                      junction_min_angle : Optional[float] = None) -> int:
    """Smooth sharp angles across HUC and river linestrings."""
    if junction_min_angle is None:
        junction_min_angle = min_angle

    count = smoothHUCsSharpAngles(hucs, min_angle, junction_min_angle)
    logging.info(f" ... cleaned up {count} sharp angles on HUCs.")
        
    for ri, river in enumerate(rivers):
        lcount = smoothOutletSharpAngles(hucs, river, junction_min_angle)
        logging.info(f" ... cleaned up {lcount} sharp angles at outlets on river {ri}.")
        count += lcount

        rcount = 0
        for reach in river:
            lcount, reach.linestring = smoothInternalSharpAngles(reach.linestring, min_angle)
            rcount += lcount
        for reach in river:
            rcount += smoothUpstreamSharpAngles(hucs, reach, junction_min_angle)
        logging.info(f" ... cleaned up {rcount} internal sharp angles on river {ri}.")
        count += rcount
        
    return count

                
def _isInternalSharpAngle(coords : np.ndarray | shapely.geometry.LineString,
                          i : int,
                          min_angle : float) -> bool:
    if isinstance(coords, shapely.geometry.LineString):
        return _isInternalSharpAngle(np.array(coords.coords), i, min_angle)

    angle = watershed_workflow.utils.computeAngle(coords[i-1] - coords[i],
                                                  coords[i+1] - coords[i])
    return (angle < min_angle) or (angle > (360 - min_angle))

def isInternalSharpAngle(linestring : shapely.geometry.LineString,
                         min_angle : float):
    """Are there any sharp angles at an interior point of a linestring?"""
    coords = np.array(linestring.coords)
    return any(_isInternalSharpAngle(coords, i, min_angle) for i in range(1, len(coords)-1))


def _smoothInternalSharpAngles(linestring : shapely.geometry.LineString,
                          min_angle : float) -> Tuple[int, shapely.geometry.LineString]:
    """Smooths any internal angle less than min_angle."""
    coords = np.array(linestring.coords)
    if len(coords) < 3:
        return 0, linestring

    new_coords = np.copy(coords)

    if len(coords) == 3:
        # cannot move endpoints, so only the middle point can move
        if _isInternalSharpAngle(coords, 1, min_angle):
            # what is the right condition?  Should we remove the point or move it?
            if watershed_workflow.utils.computeDistance(coords[0], coords[2]) \
               < max(watershed_workflow.utils.computeDistance(coords[0], coords[1]),
                     watershed_workflow.utils.computeDistance(coords[1], coords[2])):
                # just remove the extra point
                return 1, shapely.geometry.LineString([coords[0], coords[2]])
            else:
                # move the point to the triangle centroid
                new_coords[1] = watershed_workflow.utils.computeTriangleCentroid(coords[0], coords[1], coords[2])
                return 1, shapely.geometry.LineString(new_coords)
        else:
            return 0, linestring

    else:
        i = 1
        count = 0
        while i < len(coords)-1:
            if _isInternalSharpAngle(coords, i, min_angle):
                count += 1
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
        return count, shapely.geometry.LineString(new_coords2)


def smoothInternalSharpAngles(linestring : shapely.geometry.LineString,
                              min_angle : float) -> Tuple[int, shapely.geometry.LineString]:
    """Smooths any internal angle less than min_angle."""
    # note this just iterates until no points are removed
    if min_angle == 0:
        return 0, linestring
    
    done = False
    count = 0
    while not done:
        lcount, new_linestring = _smoothInternalSharpAngles(linestring, min_angle)
        count += lcount
        done = len(new_linestring.coords) == len(linestring.coords)
        linestring = new_linestring
    return count, linestring


def _spreadAngles(linestrings : List[shapely.geometry.LineString],
                  min_angle : float,
                  can_move : Optional[List[bool]] = None) -> Tuple[int, List[shapely.geometry.LineString], np.ndarray]:
    """Given a list of linestrings, all of whose terminal coordinate
    is the same (origin) and whose ordering is clockwise around that
    origin, nudge the second-to-last point of all linestrings (except
    the first) to ensure that all angles are >= min_angle.
    """
    pre_angles = _getAngles(linestrings)
    logging.debug(f'  smoothing angles pre: {pre_angles}')

    if min(pre_angles) >= min_angle: return 0, linestrings, pre_angles

    if can_move is None:
        can_move = [True for ls in linestrings]
    can_move[0] = False
    for i,ls in enumerate(linestrings):
        if len(ls.coords) == 2:
            # cannot move this one!
            can_move[i] = False

    # compute a valid set of angles, all > min_angle
    n = len(linestrings)
    angles = np.copy(pre_angles)
    itrs = 0
    while min(angles) < min_angle:
        for i in range(len(angles)):
            if angles[i] < min_angle:
                if can_move[i]:
                    # move one
                    angles[i] += 1
                    angles[(i-1)%n] -= 1
                if can_move[(i+1)%n]:
                    angles[i] += 1
                    angles[(i+1)%n] -= 1

        itrs += 1
        if itrs > 1000:
            raise RuntimeError("Cannot smooth sharp angles at junction.")

    logging.debug(f'  smoothing angles post: {angles}')
    assert(abs(sum(angles) - 360) < 0.01)
    assert(min(angles) >= min_angle)

    # now that we have a valid set of angles, we can redistribute the
    # linestrings to those angles by moving the first interior points.
    linestrings_out = [ls for ls in linestrings]
    for i in range(len(linestrings)-1):
        fixed_ls = linestrings_out[i]
        move_ls = linestrings_out[i+1]
        v1 = np.array(fixed_ls.coords[-1]) - np.array(fixed_ls.coords[-2])
        dist = watershed_workflow.utils.computeDistance(move_ls.coords[-1], move_ls.coords[-2])
        v2 = watershed_workflow.utils.projectVectorAtAngle(v1, angles[i], dist)

        new_coords = np.array(linestrings[i+1].coords)
        new_coords[-2] = new_coords[-1] - np.array(v2)
        logging.debug(f'moving: {linestrings[i+1]}')
        new_ls = shapely.geometry.LineString(new_coords)
        logging.debug(f'to: {new_ls}')
        linestrings_out[i+1] = new_ls

    for ls_in, ls_out, ls_can_move in zip(linestrings, linestrings_out, can_move):
        if not ls_can_move:
            assert ls_in is ls_out
        
    return 1, linestrings_out, _getAngles(linestrings_out)


def _getOutletLinestrings(hucs : SplitHUCs,
                          reach : River) -> Tuple[bool, List[Tuple[Any, shapely.geometry.LineString, bool]]]:
    if reach.parent is not None:
        raise RuntimeError('Do not call smoothOutletSharpAngles on an internal outlet -- only on domain outlets.')

    outlet = shapely.geometry.Point(reach.linestring.coords[-1])
    logging.debug(f'outlet: {outlet}')

    # find any HUC segments whose endpoints touches outlet.  Note that
    # these may be boundary segments or internal segments.
    #
    # Save this in a list of touches, consisting of:
    #     [ (handle, linestring, is_reversed), ]
    # Note that, in this list, handle is:
    #   - non-negative integer if it is a HUC linestring handle
    #   - negative integer if it is a child index (incremented by -1 to ensure no 0)
    #   - None for the downstream reach
    touches = [(None, reach.linestring, False),]

    # in this variant, we need only loop over boundary linestrings, not internal linestrings
    for j, spine in hucs.boundaries.items():
        for i, handle in spine.items():
            ls = hucs.linestrings[handle]
            # can only touch at endpoints, as guaranteed by previous simplify call
            if watershed_workflow.utils.isClose(ls.coords[0], outlet):
                touches.append( (handle, reverseLineString(ls), True) )
            if watershed_workflow.utils.isClose(ls.coords[-1], outlet):
                touches.append( (handle, ls, False) )
    
    if len(touches) == 1:
        # internal river, nothing to do
        return False, list()

    # must be exactly 3 -- the reach, plus two boundary linestrings
    assert len(touches) == 3

    if touches[1][0] == touches[2][0]:
        assert touches[1][2] # reversed first entry
        assert not touches[2][2] # not reversed second entry
        # the huc is a single polygon with only one boundary linestring -- split, then call, the remerge
        remerge = True
        huc_ls_coords = touches[1][1].coords
        middle_coord = len(huc_ls_coords) // 2
        assert middle_coord > 1 and middle_coord < len(huc_ls_coords)-2
        touches[1] = (touches[1][0], shapely.geometry.LineString(reversed(huc_ls_coords[0:middle_coord+1])), False)
        touches[2] = (touches[2][0], shapely.geometry.LineString(huc_ls_coords[middle_coord:]), True)
    else:
        remerge = False

    # sort huc boundary linestrings by clockwise angle around the reach
    touches = sorted(touches, key=lambda touch : watershed_workflow.utils.computeAngle(reverseLineString(reach.linestring), touch[1]))

    # confirm we did not mess up and the 0th entry is still the downstream reach
    assert touches[0][0] is None
    return remerge, touches

    
def isOutletSharpAngle(hucs : SplitHUCs,
                          reach : River,
                          min_angle = float) -> bool:
    """Are there sharp angles at the downstream end of a river, which may include HUC boundaries?"""
    _, touches = _getOutletLinestrings(hucs, reach)
    if len(touches) == 0: return False
    angles = _getAngles([touch[1] for touch in touches])
    return min(angles) < min_angle
    


def smoothOutletSharpAngles(hucs : SplitHUCs,
                            reach : River,
                            min_angle : float) -> int:
    """Smooths angles at the downstream end of a river, which may include HUC boundaries."""
    remerge, touches = _getOutletLinestrings(hucs, reach)
    if len(touches) == 0: return 0

    assert len(touches) == 3
    linestrings = [touch[1] for touch in touches]
    count, linestrings, _ = _spreadAngles(linestrings, min_angle)

    # put them back, reversing if needed
    reach.linestring = linestrings[0]
    
    if remerge:
        new_ls = shapely.geometry.LineString(list(reversed(linestrings[1].coords)) + list(linestrings[2].coords[1:]))
        hucs.linestrings[touches[1][0]] = new_ls
    else:
        hucs.linestrings[touches[1][0]] = reverseLineString(linestrings[1]) if touches[1][2] else linestrings[1]
        hucs.linestrings[touches[2][0]] = reverseLineString(linestrings[2]) if touches[2][2] else linestrings[2]
    hucs.update()
    return count


def _getUpstreamLinestrings(hucs : SplitHUCs | None,
                            reach : River) -> List[Tuple[Any, shapely.geometry.LineString, bool]]:
    """Find all linestrings that intersect at the upstream-most coordinate of reach."""
    upstream_point = shapely.geometry.Point(reach.linestring.coords[0])
    logging.debug(f'upstream_point: {upstream_point}')

    # find any HUC segments whose endpoints touches outlet.  Note that
    # these may be boundary segments or internal segments.
    #
    # Save this in a list of touches, consisting of:
    #     [ (handle, linestring, is_reversed), ]
    # Note that, in this list, handle is:
    #   - non-negative integer if it is a HUC linestring handle
    #   - negative integer if it is a child index (incremented by -1 to ensure no 0)
    #   - None for the downstream reach
    touches : List[Tuple[Any, shapely.geometry.LineString, bool]] = [(None, reverseLineString(reach.linestring), True),]
    for i, c in enumerate(reach.children):
        touches.append( (-i-1, c.linestring, False) )
    
    if hucs is not None:
        for i, ls in hucs.linestrings.items():
            # can only touch at endpoints, as guaranteed by previous simplify call
            touches_first = False
            if watershed_workflow.utils.isClose(ls.coords[0], upstream_point):
                touches_first = True
                touches.append( (i, reverseLineString(ls), True) )
            if watershed_workflow.utils.isClose(ls.coords[-1], upstream_point):
                # note, we exclude the possibility of a river beginning at
                # the boundary and ending internally to a HUC.  This could
                # be relaxed with some code that looks like the outlet
                # version of this (e.g. breaking the polygon loop into two
                # parts, then remerging it back), but we don't do that
                # until it is proven necessary.
                assert not touches_first
                touches.append( (i, ls, False) )

    if len(touches) == 1:
        # internal leaf, nothing to do
        return list()
    
    # sort huc boundary linestrings by clockwise angle around the downstream reach
    touches = sorted(touches, key=lambda touch : watershed_workflow.utils.computeAngle(reach.linestring, touch[1]))
    # confirm we did not mess up and the 0th entry is still the downstream reach
    assert touches[0][0] is None
    return touches


def isUpstreamSharpAngle(hucs : SplitHUCs | None,
                         reach : River,
                         min_angle : float) -> bool:
    """Are there sharp angles at the upstream end of a reach, which may include HUC boundaries?"""
    if min_angle == 0: return False
    touches = _getUpstreamLinestrings(hucs, reach)
    if len(touches) == 0: return False
    angles = _getAngles([touch[1] for touch in touches])
    return min(angles) < min_angle
    

def smoothUpstreamSharpAngles(hucs : SplitHUCs | None,
                              reach : River,
                              min_angle : float,
                              zipper_angle : Optional[float] = None) -> int:
    """Smooths sharp angles at the upstream end of a reach, which may include HUC boundaries."""
    if min_angle == 0: return 0
    if zipper_angle is None: zipper_angle = min_angle

    touches = _getUpstreamLinestrings(hucs, reach)
    if len(touches) == 0: return False

    linestrings = [touch[1] for touch in touches]
    try:
        angles = _getAngles(linestrings)
    except AssertionError:
        fig, ax = plt.subplots(1, 1)
        hucs.plot(color='k', ax=ax)
        mls = [reach.linestring,]
        watershed_workflow.plot.linestringWithCoords(reach.linestring, marker='x', color='grey', ax=ax)

        for child, color in zip(reach.children, watershed_workflow.colors.enumerated_palettes[1]):
            mls.append(child.linestring)
            watershed_workflow.plot.linestringWithCoords(child.linestring, marker='x', color=color, ax=ax)

        lss = shapely.geometry.MultiLineString(mls)
        bounds = lss.buffer(1000).bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        plt.show()
        
        raise RuntimeError(f'smoothUpstreamSharpAngles input data is bad -- angles upstream of reach {reach[names.ID]} are invalid')

    if len(angles) > 4:
        logging.info(f"Considering lots of angles at {linestrings[0].coords[-1]}:")
        logging.info(f"  ... angles = {angles}")

    
    # first look to zipper -- look for subsequent reaches that are within angle
    if zipper_angle > 0:
        for i in range(1, len(touches)-1):
            if touches[i][0] < 0 and touches[i+1][0] < 0 and angles[i] < zipper_angle:
                _zipperSiblings([reach.children[-touches[i][0]-1], reach.children[-touches[i+1][0]-1]], zipper_angle)
                # changed the graph, try again
                return 1 + smoothUpstreamSharpAngles(hucs, reach, min_angle, zipper_angle)

    # done zippering, now spread
    count, linestrings, new_angles = _spreadAngles(linestrings, min_angle)
    if len(angles) > 4:
        logging.info(f"  ... spread angles = {new_angles}")    

    for ls, touch in zip(linestrings, touches):
        if touch[0] is None:
            # reach's linestring
            reach.linestring = reverseLineString(ls)

        elif touch[0] < 0:
            # child linestring
            child_index = -touch[0] - 1
            reach.children[child_index].linestring = ls
        else:
            assert hucs is not None
            if touch[2]:
                # reversed!
                ls = reverseLineString(ls)
                hucs.linestrings[touch[0]] = ls

    if hucs is not None: hucs.update()
    return count


def _getHUCsJunctionLinestrings_Generator(hucs : SplitHUCs):
    huc_handles = list(hucs.linestrings.handles())
    coords = [hucs.linestrings[handle].coords[i] for handle in huc_handles for i in [0,-1]]
    kdtree = cKDTree(coords)
    done = np.zeros((len(hucs.linestrings),2), 'i')
    
    for i, (_, ls_i) in enumerate(hucs.linestrings.items()):
        for j,index in enumerate([0,-1]):
            if not done[i,j]:
                touch_points = kdtree.query_ball_point(ls_i.coords[index], 1.e-10)

                # as before, touches is a list of [handle, linestring, is_reversed]
                touches = []
                for c in touch_points:
                    done[c//2, c%2] = True
                    handle = huc_handles[c//2]
                    is_reversed = c%2 == 0
                    ls = hucs.linestrings[handle]
                    if is_reversed:
                        ls = reverseLineString(ls)
                    touches.append( (handle, ls, is_reversed) )
                assert all(watershed_workflow.utils.isClose(touch[1].coords[-1], ls_i.coords[index]) for touch in touches)

                # sort by angle from the first
                touches = sorted(touches, key=lambda touch : watershed_workflow.utils.computeAngle(reverseLineString(touches[0][1]), touch[1]))
                yield touches

    
def isHUCsSharpAngle(hucs : SplitHUCs,
                     min_angle : float,
                     junction_min_angle : Optional[float] = None) -> bool:
    """Are there sharp angles in a HUC boundary or junction?"""
    if junction_min_angle is None:
        junction_min_angle = min_angle
    
    if min_angle == 0: return False
    # check internal angles
    if min_angle > 0:
        for handle, ls in hucs.linestrings.items():
            if isInternalSharpAngle(ls, min_angle):
                return True

    # check junction angles
    if junction_min_angle > 0:
        for touches in _getHUCsJunctionLinestrings_Generator(hucs):
            angles = _getAngles([t[1] for t in touches])
            if min(angles) < junction_min_angle:
                return True
    return False
    

def smoothHUCsSharpAngles(hucs : SplitHUCs,
                          min_angle : float,
                          junction_min_angle : Optional[float] = None) -> int:
    """Smooths sharp angles on HUC linestrings or junctions."""
    if junction_min_angle is None:
        junction_min_angle = min_angle

    # smooth internal angles
    count = 0
    if min_angle > 0:
        for handle, ls in hucs.linestrings.items():
            lcount, hucs.linestrings[handle] = _smoothInternalSharpAngles(ls, min_angle)
            count += lcount

    # smooth junction angles
    if junction_min_angle > 0:
        for touches in _getHUCsJunctionLinestrings_Generator(hucs):
            linestrings = [t[1] for t in touches]
            for ls in linestrings:
                logging.debug(f'ls before: {ls}')
            lcount, linestrings, _  = _spreadAngles(linestrings, junction_min_angle)
            count += lcount
            for ls in linestrings:
                logging.debug(f'ls after: {ls}')

            # put them back
            for touch,ls in zip(touches, linestrings):
                hucs.linestrings[touch[0]] = reverseLineString(ls) if touch[2] else ls
    hucs.update()
    return count


def _zipperSiblings(reaches : List[River],
                   min_angle : float):
    """This function assumes that the list of reaches are siblings,
    share a common endpoint, and the angle between their linestrings
    as they leave the endpoint is less than min_angle.

    It then chomps from the endpoint, removing point by point until
    one of the angles is greater than min_angle.

    Note this assumes that all reaches have been discretized, and with
    a fairly consistent ds.
    """
    logging.info(f'-- zippering children at reach: {reaches[0].parent.index}')

    assert len(reaches) > 1
    linestrings = [r.linestring for r in reaches]
    
    for ls in linestrings:
        assert watershed_workflow.utils.isClose(reaches[0].linestring.coords[-1], ls.coords[-1])
    
    # chomp the reaches until max of angles > min_angle
    angles = _getAngles(linestrings)
    assert max(angles) > min_angle
    
    new_coords = [linestrings[0].coords[-1],]
    done = False
    count = 0
    while not done:
        centroid = np.mean([ls.coords[-2] for ls in linestrings], axis=0)
        new_coords.append(centroid)
        count += 1
        linestrings = [shapely.geometry.LineString(ls.coords[:-2]+[centroid,]) for ls in linestrings]

        if any(len(ls.coords) == 1 for ls in linestrings):
            done = True
        else:
            angles = _getAngles(linestrings)
            if max(angles) > min_angle:
                done = True

    logging.info(f'   zipper up {count} segments to {centroid}')
                
    new_downstream_ls = shapely.geometry.LineString(reversed(new_coords))
    logging.info(f'   introducing a new reach of length {new_downstream_ls.length} with {len(new_downstream_ls.coords)} coords')

    old_downstream_nodes = []
    merged_ds_node = None
    for reach, ls in zip(reaches, linestrings):
        us, ds = reach.split(len(ls.coords)-1)
        us.moveCoordinate(-1, centroid)
        old_downstream_nodes.append(ds)

        if merged_ds_node is None:
            merged_ds_node = ds
        else:
            merged_ds_node = watershed_workflow.river_tree.combineSiblings(merged_ds_node, ds, new_downstream_ls)
