"""This module includes functions to increase the resolution of the
river network and the HUC boundary, using the original river network
and HUC.

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


class ComputeTargetLength(abc.ABC):
    @property
    @abc.abstractmethod
    def is_uniform(self) -> bool:
        pass

    @abc.abstractmethod
    def __call__(self, arg : Any) -> float:
        pass
    

def createStrategyFixed(target_length : float) -> ComputeTargetLength:
    class ComputeTargetLengthFixed(ComputeTargetLength):
        is_uniform = True
        def __init__(self, target_length):
            self._target_length = target_length

        def __call__(self, arg):
            return self._target_length

    return ComputeTargetLengthFixed(target_length)


def createStrategyByProperty() -> ComputeTargetLength:
    class ComputeTargetLengthByProperty(ComputeTargetLength):
        is_uniform = True
        def __call__(self, reach):
            return reach.properties[names.TARGET_SEGMENT_LENGTH]
    return ComputeTargetLengthByProperty()


def createStrategyByCallable(func : Callable[[Any,],float]) -> ComputeTargetLength:
    class ComputeTargetLengthByCallable(ComputeTargetLength):
        is_uniform = True
        def __init__(self, func):
            self._func = func

        def __call__(self, reach):
            return self._func(reach)
    return ComputeTargetLengthByCallable(func)

def createStrategyByDistance(dist_args : List[float], shp : shapely.geometry.base.BaseGeometry) -> ComputeTargetLength:
    class ComputeTargetLengthByDistanceToShape(ComputeTargetLength):
        is_uniform = False
        def __init__(self, dist_args, shp):
            self._d1, self._d2 = dist_args[0], dist_args[2]
            self._l1, self._l2 = dist_args[1], dist_args[3]
            self._shp = shp

        def _fromDistance(self, dist):
            if dist < self._d1:
                return self._l1
            elif dist > self._d2:
                return self._l2
            else:
                return self._l1 + (dist - self._d1) / (self._d2 - self._d1) * (self._l2 - self._l1)

        def __call__(self, point):
            return self._fromDistance(shapely.distance(self._shp, shapely.geometry.Point(point)))

    return ComputeTargetLengthByDistanceToShape(dist_args, shp)
    

def resampleRivers(rivers : List[River],
                   computeTargetLength : Callable[[River,],float]) -> None:
    """Resamples rivers, in place, given a strategy to compute the target segment length."""
    for river in rivers:
        resampleRiver(river, computeTargetLength)


def resampleRiver(river : River,
                  computeTargetLength : Callable[[River,],float]) -> None:
    """Resamples a river, in place, given a strategy to compute the target segment length."""
    for node in river.preOrder():
        node.linestring = resampleLineStringUniform(node.linestring, computeTargetLength(node))

        
def resampleHUCs(huc : SplitHUCs,
                 computeTargetLength : ComputeTargetLength) -> None:
    """Resamples HUC boundaries, in place, given a strategy to compute the target segment length."""
    if computeTargetLength.is_uniform:
        for i, ls in enumerate(huc.linestrings):
            huc.linestrings[i] = resampleLineStringUniform(ls, computeTargetLength(ls))
    else:
        for i, ls in enumerate(huc.linestrings):
            huc.linestrings[i] = resampleLineStringNonuniform(ls, computeTargetLength)
                       
        
        
def _resampleLineStringUniform(linestring : shapely.geometry.LineString,
                              target_length : float) -> shapely.geometry.LineString:
    """Resample a linestring uniformly with no respect for previous discrete coords."""
    count = math.ceil(linestring.length / target_length) + 1
    s = np.linspace(0, 1, count)
    return shapely.line_interpolate_point(linestring, s, normalized=True)
                

def resampleLineStringUniform(linestring : shapely.geometry.LineString,
                              target_length : float) -> shapely.geometry.LineString:
    """Resample a linestring nearly uniformly, but preferring to keep previous discrete coords if possible."""
    assert linestring.length > 0
    coords = np.array(linestring.coords)
    dcoords = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    if np.max(dcoords) > target_length:
        new_coords = []
        i = 0
        j = 1
        while i < len(coords)-1:
            while j < len(coords) and dcoords[j-1] < target_length:
                j += 1
            sublinestring = shapely.geometry.LineString(coords[i:j+1])
            new_coords.extend(_resampleLineStringUniform(sublinestring, target_length))
            i = j
            j += 1
    else:
        new_coords = _resampleLineStringUniform(linestring, target_length)
    return shapely.geometry.LineString(new_coords)


def _resampleLineStringNonuniform(linestring : shapely.geometry.LineString,
                                  computeTargetLength : Callable[[Any,],float]) -> shapely.geometry.LineString:
    """Resample a linestring by distance with no respect for previous discrete coords."""
    coords = [linestring.coords[0]]
    arclens = [0.,]
    length = linestring.length
    s = 0.0
    while s < length:
        d = computeTargetLength(coords[-1])
        s = s + d
        arclens.append(s)
        coords.append(linestring.interpolate(s))

    # we could now have a really short ls at the end... add one more
    # full-size linestring and shrink the whole thing back to length.
    arclens_a = np.array(arclens)
    arclens_a = arclens_a * length / s
    new_linestring_coords = shapely.line_interpolate_point(linestring, arclens_a)
    return new_linestring_coords
    

def resampleLineStringNonuniform(linestring : shapely.geometry.LineString,
                                 computeTargetLength : Callable[[Any,],float]) -> shapely.geometry.LineString:
    """This function resamples coordinates in node's linestrings to a preferred size."""
    assert linestring.length > 0

    targets = [computeTargetLength(c) for c in linestring.coords]
    if max(targets) == min(targets):
        return resampleLineStringUniform(linestring, targets[0])

    if targets[0] > targets[-1]:
        # work from closest to furthest, so flip the linestring
        linestring = shapely.geometry.LineString(reversed(linestring.coords))
        reverse = True
    else:
        reverse = False

    # greedily add coords along the linestring
    coords = np.array(linestring.coords)
    dcoords = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    ml = targets[0]
    if np.max(dcoords) > ml:
        new_coords = []
        i = 0
        j = 1
        while i < len(coords)-1:
            while j < len(coords) and dcoords[j-1] < ml:
                j += 1
                ml = computeTargetLength(coords[j])
            sublinestring = shapely.geometry.LineString(coords[i:j+1])
            new_coords.extend(_resampleLineStringNonuniform(sublinestring, computeTargetLength))
            i = j
            j += 1
            if j < len(coords):
                ml = computeTargetLength(coords[j])

    else:
        new_coords = _resampleLineStringNonuniform(linestring, computeTargetLength)

    if reverse:
        new_coords = list(reversed(new_coords))
    return shapely.geometry.LineString(new_coords)
                                       

















def removeSharpAngles(rivers : List[River],
                      hucs : SplitHUCs,
                      angle_limit : float = 0,
                      junction_angle_limit : float = 0,
                      huc_river_angle_limit : float = 0,
                      merge_limit : float = 0) -> None:
    """
    Parameters
    ----------
    angle_limit: float, optional
      If provided, smooth any angle formed by three consecutive points
      on the river tree smaller thea this value (degrees)
    junction_angle_limit: float, optional
      If provided, remove sections to eliminate angles below this
      tolerance at junctions"""
    for river in rivers:
        _removeSharpAngles(river,
                             hucs,
                             angle_limit=angle_limit,
                             junction_angle_limit=junction_angle_limit,
                             huc_river_angle_limit=huc_river_angle_limit)
        watershed_workflow.hydrography.merge(river, tol = merge_limit * 0.6)


def _removeSharpAngles(river : River,
                       hucs : SplitHUCs,
                       angle_limit : float,
                       junction_angle_limit : float,
                       huc_river_angle_limit : float) -> None:
    """Smooth out any sharp angles in the river tree and between
    river tree and watershed boundary

    Note all angle tolerances are in degrees.

    """
    for node in river.preOrder():
        if angle_limit is not None:
            removeSharpAnglesFromSeg(node, angle_limit=angle_limit)  # from internal linestrings
            assert (node.is_locally_continuous())

        if len(node.children) != 0:
            # at junctions, angle between parent and child node
            if angle_limit is not None:
                treatNodeJunctionsForSharpAngles(node, angle_limit=angle_limit)
                assert (node.is_locally_continuous())

            # angle between two children (how often can we have >2 children??)
            if junction_angle_limit is not None:
                remove = True
                while remove:
                    remove = treatSmallAngleBetweenChildNodes(node,
                                                                   angle_limit=junction_angle_limit)
                    if not node.is_locally_continuous():
                        print(node.properties['ID'])
                    assert (node.is_locally_continuous())

        assert river.isContinuous()

    if huc_river_angle_limit > 0:
        # checks angle betweebn huc linestring and river at the outlets and rotate the part of huc linestring as needed
        treatSmallAngleBtwRiverHUC(river, hucs, angle_limit=huc_river_angle_limit)


def removeSharpAnglesFromSeg(node : River,
                             angle_limit : float = 10) -> None:
    """Smooth out any sharp angles in a reach linestring."""
    seg = node.linestring
    seg_coords = seg.coords[:]
    for i in range(len(seg_coords) - 2):
        seg_up = shapely.geometry.LineString([seg_coords[i], seg_coords[i + 1]])
        seg_down = shapely.geometry.LineString([seg_coords[i + 1], seg_coords[i + 2]])
        angle = watershed_workflow.river_mesh.angleRiversSegs(ref_seg=seg_down, seg=seg_up)
        if angle > 360 - angle_limit or angle < angle_limit:
            logging.info(
                f"removing sharp angle in linestring: {angle} for node {node.properties['ID']}")
            if len(seg_coords) > 3:
                new_point = shapely.geometry.Polygon(
                    [seg_coords[i], seg_coords[i + 1], seg_coords[i + 2]]).centroid
                seg_coords_new = seg_coords[:i] + new_point.coords[:] + seg_coords[i + 3:]
                node.linestring = shapely.geometry.LineString(seg_coords_new)
                removeSharpAnglesFromSeg(node, angle_limit=angle_limit)
            else:
                seg_coords_new = [seg_coords[0], seg_coords[2]]
                node.linestring = shapely.geometry.LineString(seg_coords_new)
            if i == 0:  # upstream end of the linestring moved, update children linestrings
                for child in node.children:
                    child_seg_coords = child.linestring.coords[:]
                    child_seg_coords[-1] = node.linestring.coords[0]
                    child.linestring = shapely.geometry.LineString(child_seg_coords)
            elif i == len(seg_coords
                          ) - 3:  # downstream end if the linestring moved, update parent and siblings
                if node.parent != None:  # update parent
                    parent_seg_coords = node.parent.linestring.coords[:]
                    parent_seg_coords[0] = node.linestring.coords[-1]
                    node.parent.linestring = shapely.geometry.LineString(parent_seg_coords)
                for sibling in node.siblings():  # update siblings
                    sibling_coords = sibling.linestring.coords[:]
                    sibling_coords[-1] = node.linestring.coords[-1]
                    sibling.linestring = shapely.geometry.LineString(sibling_coords)
            assert node.parent.isContinuous()


def treatNodeJunctionsForSharpAngles(node : River, angle_limit : float = 10) -> None:
    """Smooth out junction angles.

    Moves the junction point to the centroid of the triangle formed by
    the junction point and the two neighboring points.  This is done
    recursively until the tolerance is met.
    """
    for child in node.children:
        # note, this must be done here.  If there are multiple
        # children and the first moves the parent, we have to look at
        # the moved parent, not the original parent.
        seg1 = node.linestring
        seg2 = child.linestring
        is_changed, seg1, seg2 = removeSharpAnglesAtReachJunctions(seg1,
                                                                   seg2,
                                                                   angle_limit=angle_limit,
                                                                   id=node.properties['ID'])

        if is_changed:
            node.linestring = seg1
            child.linestring = seg2
            for sibling in child.siblings():
                sibling_coords = sibling.linestring.coords[:]
                sibling_coords[-1] = child.linestring.coords[-1]
                sibling.linestring = shapely.geometry.LineString(sibling_coords)


def removeSharpAnglesAtReachJunctions(seg1 : shapely.geometry.LineString,
                                      seg2 : shapely.geometry.LineString,
                                      angle_limit : float = 10, 
                                      id : Any = None) -> \
                                      Tuple[bool, shapely.geometry.LineString, shapely.geometry.LineString]:
    """Moves the common shared point of seg1 and seg2 to the centroid
    of the triangle formed by the junction point and the two
    neighboring points.  This is done recursively until the tolerance
    is met.
    """
    is_changed = False
    seg_up = shapely.geometry.LineString([seg2.coords[-2], seg2.coords[-1]])
    seg_down = shapely.geometry.LineString([seg1.coords[0], seg1.coords[1]])
    angle = watershed_workflow.river_mesh.angleRiversSegs(ref_seg=seg_down, seg=seg_up)
    if angle > 360 - angle_limit or angle < angle_limit:
        logging.info(f"removing sharp angle at junction: {angle} for node {id}")
        new_point = shapely.geometry.Polygon([seg2.coords[-2], seg2.coords[-1],
                                              seg1.coords[1]]).centroid
        if len(seg1.coords) < 3:
            seg1_coords_new = new_point.coords[:] + seg1.coords[1:]
        else:
            seg1_coords_new = new_point.coords[:] + seg1.coords[2:]
        if len(seg2.coords) < 3:
            seg2_coords_new = seg2.coords[:-1] + new_point.coords[:]
        else:
            seg2_coords_new = seg2.coords[:-2] + new_point.coords[:]
        seg1 = shapely.geometry.LineString(seg1_coords_new)
        seg2 = shapely.geometry.LineString(seg2_coords_new)
        _, seg1, seg2 = removeSharpAnglesAtReachJunctions(seg1, seg2, angle_limit=angle_limit)
        is_changed = True
    return is_changed, seg1, seg2


def treatSmallAngleBetweenChildNodes(node : River, angle_limit : float = 10) -> bool:
    """Zippers up junctions that have small angles.  Returns True if a section was removed.

    \  |       \ /
     ||   -->   |
      \          \

    """
    if len(node.children) > 1 and len(node.children) < 3:
        # note, we don't really have a good implementation for 3
        # children.  Could work on one if it shows up...
        seg1 = node.linestring
        angles = []
        for child in node.children:
            seg2 = child.linestring
            seg_up = shapely.geometry.LineString([seg2.coords[-2], seg2.coords[-1]])
            assert (watershed_workflow.utils.isClose(seg2.coords[-1], seg1.coords[0], 1.e-10))
            seg_down = shapely.geometry.LineString([seg1.coords[0], seg1.coords[1]])
            angle = watershed_workflow.river_mesh.angleRiversSegs(ref_seg=seg_down, seg=seg_up)
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
                        grandchild_seg_coords = grandchild.linestring.coords[:]
                        grandchild_seg_coords[-1] = new_junction
                        grandchild.linestring = shapely.geometry.LineString(grandchild_seg_coords)
                        node.addChild(grandchild)
                    child.remove()

            # we changed things -- return True so this can be called
            # repeatedly until all sections that have the close angle
            # are removed.
            return True
    return False

def treatSmallAngleBtwRiverHUC(river : River, hucs : SplitHUCs, angle_limit : float = 20) -> None:
    """Rotates the huc linestring to increase the angle between huc linestring and the river tree at the outlets as needed
    """

    river_mls = shapely.geometry.MultiLineString(list(river))

    # the intersection_point might be a MultiPoint with intersections at leaf tips
    # find all leaf tip points and exclude coinciding intersection points
    leaf_tips = [shapely.geometry.Point(leaf.linestring.coords[0]) for leaf in river.leaf_nodes()]

    for i, seg in enumerate(hucs.linestrings):

        if seg.intersects(river_mls):  # does huc linestring intersect with the river
            intersection_point = seg.intersection(river_mls)
            intersection_found = True
            if isinstance(intersection_point, shapely.geometry.multipoint.MultiPoint):
                non_coinciding_points = [
                    p for p in intersection_point
                    if not any(p.distance(point) < 1e-6 for point in leaf_tips)
                ]
                # Check if there's exactly one point left after filtering
                if len(non_coinciding_points) == 1:
                    intersection_point = non_coinciding_points[0]
                elif len(non_coinciding_points) > 1:
                    print([point.coords[:] for point in non_coinciding_points])
                    raise ValueError(
                        "The intersection_point must be a single shapely.geometry.Point object")
                elif not non_coinciding_points:
                    intersection_found = False

            if intersection_found:
                parent_node = watershed_workflow.river_mesh.nodeAtIntersection(
                    intersection_point, river)
                if parent_node.parent == None:  # check if it is the outlet node for this river
                    outlet_junction = True
                else:
                    outlet_junction = False

                ind_intersection_point = watershed_workflow.river_mesh._indexPointInSeg(
                    parent_node.linestring, intersection_point)

                # reference linestring for angles
                if outlet_junction:
                    if len(parent_node.linestring.coords) > 2:
                        ref_seg = shapely.geometry.LineString(
                            parent_node.linestring.coords[ind_intersection_point - 2:])
                    else:
                        ref_seg = parent_node.linestring
                else:
                    if len(parent_node.linestring.coords) > 2:
                        ref_seg = shapely.geometry.LineString(
                            parent_node.linestring.coords[ind_intersection_point:ind_intersection_point
                                                       + 2])
                    else:
                        ref_seg = parent_node.linestring

                # angle of huc linestring with the ref linestring
                huc_seg_angle = watershed_workflow.river_mesh.angleRiversSegs(ref_seg, seg)
                # angles of river with the huc linestring
                river_angles = [
                    watershed_workflow.river_mesh.angleRiversSegs(ref_seg, _seg) - huc_seg_angle
                    for _seg in parent_node.children
                ]

                # is any angle between river and huc linestring smaller than the limit?
                angle_check = checkAbsSmaller(river_angles, angle_limit)

                if angle_check[0]:
                    assert isinstance(angle_check[1], int)
                    river_angle = river_angles[angle_check[1]]
                    logging.info(
                        f"removing sharp angle between river and huc: {river_angle} for node {parent_node.properties['ID']}"
                    )
                    if river_angle > 0:
                        rotate_angle = angle_limit - river_angle + 5
                    else:
                        rotate_angle = -(angle_limit - abs(river_angle) + 5)

                    # rotated such that the angle is increased
                    rotated_seg = shapely.affinity.rotate(seg, rotate_angle, origin=intersection_point)  

                    seg_orientation_flag = np.argmin([
                        watershed_workflow.utils.computeDistance(seg_end, intersection_point.coords[0])
                        for seg_end in [seg.coords[0], seg.coords[-1]]
                    ])
                    if seg_orientation_flag == 0:
                        seg_new_coords = seg.coords[:]
                        seg_new_coords[1] = rotated_seg.coords[1]
                        seg = shapely.geometry.LineString(seg_new_coords)
                    elif seg_orientation_flag == 1:
                        seg_new_coords = seg.coords[:]
                        seg_new_coords[-2] = rotated_seg.coords[-2]
                        seg = shapely.geometry.LineString(seg_new_coords)
                    hucs.linestrings[i] = seg


def checkAbsSmaller(numbers : Iterable[float],
                    value : float) -> Tuple[bool, int | None]:
    for index, num in enumerate(numbers):
        if abs(num) < value:
            return True, index
    return False, None
