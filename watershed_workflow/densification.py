"""This module includes functions to increase the resolution of the
river network and the HUC boundary, using the original river network
and HUC.

"""
import logging
import numpy as np
import math
from scipy import interpolate
import shapely

import watershed_workflow.utils


def densify_rivers(rivers, rivers_raw=None, **kwargs):
    """Returns a list for densified rivers"""
    if rivers_raw is None:
        rivers_raw = [None, ] * len(rivers)
    for river, river_raw in zip(rivers, rivers_raw):
        densify_river(river, river_raw, **kwargs)

    mins = []
    for river in rivers:
        for line in river.depthFirst():
            coords = np.array(line.coords[:])
            dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
            mins.append(np.min(dz))
    logging.info(f"  river min seg length: {min(mins)}")
    logging.info(f"  river median seg length: {np.median(np.array(mins))}")


def densify_rivers_new(rivers, limit=50):
    """Returns a list for densified rivers"""
    for river in rivers:
        for node in river.preOrder():
            if isinstance(limit, bool):
                if limit and 'target_length' in node.properties:
                    target_length = node.properties['target_length']
                else: 
                    raise RuntimeError('not a valid option to provide width')
            else:
                    target_length = limit 
            node.segment = resample_linestring_preserve_ends(node.segment, target_length)

    mins = []
    for river in rivers:
        for line in river.depthFirst():
            coords = np.array(line.coords[:])
            dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
            mins.append(np.min(dz))
    logging.info(f"  river min seg length: {min(mins)}")
    logging.info(f"  river median seg length: {np.median(np.array(mins))}")


def densify_river(river, river_raw=None, limit=100):
    """This function traverse in the river tree and densify node.segments in place.
    
    Parameters:
    -----------
    river: watershed_workflow.river_tree.River object
      River tree to be redensified.
    river_raw: watershed_workflow.river_tree.River object, optional
      Original river tree containing all known points.
    limit : float
      Upper bound on section length.
    """
    if 'ID' in river.properties.keys() and river_raw is not None:
        NHD_ids_raw = []
        for node in river_raw.preOrder():
            NHD_ids_raw.append(node.properties['ID'])
    elif river_raw is not None:
        assert (len(river) == len(river_raw))

    for j, node in enumerate(river.preOrder()):
        if river_raw is not None:
            if 'ID' in river.properties.keys():
                node_index_in_raw = NHD_ids_raw.index(node.properties['ID'])
                node_raw = list(river_raw.preOrder())[node_index_in_raw]
            else:
                node_raw = list(river_raw.preOrder())[j]
        else:
            node_raw = None

        node.segment = densify_node_segments(node, node_raw, limit=limit)


def densify_node_segments(node, node_raw, limit=100):
    """This function adds equally-spaced points in the reach-sections longer than the limit at a desired resolution
        potentially using original river tree
     
    Parameters:
     -----------
    node: node of a watershed_workflow.river_tree.River object
        node of a simplifed tree (sparse points) that is to be densified 
    node_raw: nodeof a watershed_workflow.river_tree.River object, optional
        node from the original tree containing all the known points from NHDPlus 
    limit : int
        limit on the section length above which more points are added

    Returns
    -------
    node.segment: node.segment of a watershed_workflow.river_tree.River object
        a densified (inplace) node.segment
    """

    seg_coords = list(node.segment.coords)  # coordinates of node.segment to be densified
    seg_coords_densified = seg_coords.copy()  # segment coordinates densified


    if isinstance(limit, bool):
        if limit and 'target_length' in node.properties:
            target_length = node.properties['target_length']
        else: 
            raise RuntimeError('not a valid option to provide width')
    else:
            target_length = limit
            
    j = 0
    for i in range(len(seg_coords) - 1):
        section_length = watershed_workflow.utils.distance(seg_coords[i], seg_coords[i + 1])
        if section_length > target_length:
            number_new_points = int(section_length // target_length)
            end_points = [seg_coords[i],
                          seg_coords[i + 1]]  # points betwen which more points will be added
            if node_raw is not None:
                seg_coords_raw = list(node_raw.segment.coords
                                      )  # coordinates of node.segment from original river network
                new_points = _interpolate_with_orig(end_points, seg_coords_raw, number_new_points)
            else:
                new_points = _interpolate_simple(end_points, number_new_points)
            seg_coords_densified[j + 1:j + 1] = new_points
            j += number_new_points
        j += 1
    node.segment.coords = seg_coords_densified
    return node.segment


def densify_hucs(huc, huc_raw=None, rivers=None, limit_scales=None):
    """This function densify huc boundaries. The densification length scale either can be a constant value or a refinement 
    function where huc segment refinedment is greater for huc segments closer to the river tree
     
    Parameters:
     -----------
      hucs: watershed_workflow.split_hucs.SplitHUCs object
        huc to be densified 
      huc_raw: watershed_workflow.split_hucs.SplitHUCs object
        original huc with all the known points from NHDPlus 
      river: watershed_workflow.river_tree.River object
        to check the proximity of huc and river for refinement 
      limit_scales: int or List 
        limit of section length above which more points are added, either a constant value or a list for step refinement 
        [near_distance, near_length_scale, far_distance, far_length_scale]

     Returns
    -------
    watershed_densified: watershed_workflow.split_hucs.SplitHUCs object
        a densified huc
    """
    for i, seg in enumerate(huc.segments):  # densifying segment by segment
        # find which original segment is this segment part of, so we can use it to resample
        logging.debug(f"trying to refine huc segment: {i}")
        coords = list(seg.coords)
        if huc_raw is not None:
            seg_raw = None
            for j, seg_orig in enumerate(huc_raw.segments):
                if seg.intersects(seg_orig):
                    intersect_seg = seg.intersection(seg_orig)
                    logging.debug(
                        f"original huc segment {j} intersect huc segment {i} as {type(intersect_seg)}"
                    )
                    if isinstance(intersect_seg, shapely.geometry.MultiPoint) and len(
                            intersect_seg.geoms) == 2:
                        pass
                    elif isinstance(intersect_seg, shapely.geometry.LineString) or \
                         isinstance(intersect_seg, shapely.geometry.MultiPoint) or \
                         isinstance(intersect_seg, shapely.geometry.collection.GeometryCollection):
                        seg_raw = seg_orig
                        logging.debug(f"for huc segment {i}, found original huc segment {j}")
                        coords_raw = list(seg_raw.coords)

                        if type(limit_scales) is list:
                            # basic refine
                            coords_densified_basic = _densify_hucs(coords,
                                                                   coords_raw,
                                                                   rivers,
                                                                   limit_scales=limit_scales[-1])
                            # adaptive refine
                            coords_densified = _densify_hucs(coords_densified_basic,
                                                             coords_raw,
                                                             rivers,
                                                             limit_scales=limit_scales)

                        else:
                            coords_densified = _densify_hucs(coords,
                                                             coords_raw,
                                                             rivers,
                                                             limit_scales=limit_scales)

                        break
                else:
                    logging.debug(f"original huc segment {j} do not intersect huc segment {i}")

            if seg_raw == None:
                logging.debug(
                    "did not find corresponding huc.segment in original, doing simple interpolation"
                )
                coords_raw = None
                coords_densified = _densify_hucs(coords,
                                                 coords_raw,
                                                 rivers,
                                                 limit_scales=limit_scales)

        else:
            coords_raw = None
            coords_densified = _densify_hucs(coords, coords_raw, rivers, limit_scales=limit_scales)

        huc.segments[i] = shapely.geometry.LineString(coords_densified)


def _densify_hucs(coords, coords_raw=None, rivers=None, limit_scales=None):
    """This function increases the resolution of huc boundary by adding equally spaced interpolated points

     Parameters:
     -----------
      coords: List            
        coordinates of the huc segment to be densified
      coords_raw: List              
        coordinates of the original huc segment from which points can be resmapled
      limit_scales: int or List 
        limit of section length above which more points are added, either a constant value or a list for step refinement 
        [near_distance, near_length_scale, far_distance, far_length_scale]
    
      Returns
    -------
    coords_densified: List
        densified coordinates of a huc segment
    """
    adaptive = type(limit_scales) is list  # setting up flag

    coords_densified = coords.copy()
    j = 0
    for i in range(len(coords) - 1):

        # calculation of limit for a set of point
        if adaptive:
            limit = limit_from_river_distance([coords[i], coords[i + 1]], limit_scales, rivers)
        else:
            limit = limit_scales

        section_length = math.dist(coords[i], coords[i + 1])

        if section_length > limit:
            number_new_points = int(section_length // limit)
            end_points = [coords[i],
                          coords[i + 1]]  # points between which more points will be added

            if adaptive:
                new_points = _interpolate_simple(end_points, number_new_points)
            else:
                new_points = _interpolate_with_orig(end_points, coords_raw, number_new_points)

            coords_densified[j + 1:j + 1] = new_points
            j += number_new_points
        j += 1

    return coords_densified


def resample_linestring_preserve_ends(seg, initial_spacing):
    """redensifies linestring at a desired resolution"""
    length = seg.length

    num_segments = max(int(round(length / initial_spacing)), 1)

    adjusted_spacing = length / num_segments
    
    redensified_points = [seg.interpolate(distance).coords[0] for distance in [i * adjusted_spacing for i in range(num_segments + 1)]]
    
    new_seg = shapely.geometry.LineString(redensified_points)
    return new_seg


def _interpolate_with_orig(end_points, interp_data, n):
    """This function adds desired number of new points between end points a segment (huc or river)
    resampling from orinal data

    Parameters
    ----------
    end_points: List
        end points of the segment whihc needs to be densified
    interp_data: List
        coordinates in the original segment, will be used to create an interpolator
    n: int
        number of new "equally" spaced points to be added 

    Returns:
    --------
    new_points: List
        coordinates of the densified segment
    """

    inds = [watershed_workflow.utils.closest_point_ind(point, interp_data) for point in end_points
            ]  # point-indices on original network slicing a section for interpolation
    if inds[1] < inds[0]:  # this is to deal with corner case of interpolation of the last segment
        inds[1] = -2
    if inds[0] == inds[
            1]:  # this is the case when original segment does not offer any data for interpolation here
        section_interp_data = np.array(end_points)
    else:
        section_interp_data = np.array(interp_data[inds[0]:inds[1] + 1])  # coordinates on section
    a = np.array(end_points)
    (dx, dy) = abs(a[0, :] - a[1, :])
    if dx > dy:  # interpolating on x axis
        f = interpolate.interp1d(section_interp_data[:, 0],
                                 section_interp_data[:, 1],
                                 fill_value='extrapolate')  # creating interpolator
        xnew = np.linspace(end_points[0][0], end_points[1][0],
                           n + 2)[1:-1]  # new xs equally space between existing points
        ynew = f(xnew)  # interpolated ys
    else:  # interpolating on y axis
        f = interpolate.interp1d(section_interp_data[:, 1],
                                 section_interp_data[:, 0],
                                 fill_value='extrapolate')  # creating interpolator
        ynew = np.linspace(end_points[0][1], end_points[1][1],
                           n + 2)[1:-1]  # new ys equally space between existing points
        xnew = f(ynew)  # interpolated xs
    new_points = [(xnew[k], ynew[k]) for k in range(n)]
    return new_points


def _interpolate_simple(end_points, n):
    """This function does not use any original data, just adds new equally spaced points based on linear interpolation"""
    xnew = np.linspace(end_points[0][0], end_points[1][0],
                       n + 2)[1:-1]  # new xs equally space between existing points
    ynew = np.linspace(end_points[0][1], end_points[1][1],
                       n + 2)[1:-1]  # new ys equally space between existing points
    new_points = [(xnew[k], ynew[k]) for k in range(n)]
    return new_points


def limit_from_river_distance(segment_ends, limit_scales, rivers):
    """Returns a graded refinement function based upon a distance function from rivers, for use with DensifyHucs function.
    HUC segment resolution must be higher in near_distance when the HUC segment midpoint is within near_distance from the river network.
    Length must be smaller than away_length when the HUC segment midpoint is at least away_distance from the river network.
    Area must be smaller than a linear interpolant between
    """
    near_distance, near_length, away_distance, away_length = limit_scales
    p0 = shapely.geometry.Point(segment_ends[0])
    p1 = shapely.geometry.Point(segment_ends[1])
    p_mid = shapely.geometry.Point([(segment_ends[0][0] + segment_ends[1][0]) / 2,
                                    (segment_ends[0][1] + segment_ends[1][1]) / 2])
    river_multilines = [shapely.geometry.MultiLineString(list(river)) for river in rivers]
    distance = min([
        min(p0.distance(river_multiline), p_mid.distance(river_multiline),
            p1.distance(river_multiline)) for river_multiline in river_multilines
    ])

    if distance > away_distance:
        length = away_length
    elif distance < near_distance:
        length = near_length
    else:
        length = near_length + (distance-near_distance) / (away_distance-near_distance) * (
            away_length-near_length)

    return length


def remove_sharp_angles(rivers,
                        hucs,
                        angle_limit=0,
                        junction_angle_limit=0,
                        huc_seg_river_angle_limit=0,
                        limit=None):
    """ angle_limit: float, optional
      If provided, smooth any angle formed by three consecutive points
      on the river tree smaller thea this value (degrees)
    junction_angle_limit: float, optional
      If provided, remove sections to eliminate angles below this
      tolerance at junctions"""
    for river in rivers:
        _remove_sharp_angles(river,
                             hucs,
                             angle_limit=angle_limit,
                             junction_angle_limit=junction_angle_limit,
                             huc_seg_river_angle_limit=huc_seg_river_angle_limit)
        watershed_workflow.hydrography.merge(river, tol=limit * 0.6)


def _remove_sharp_angles(river,
                         hucs,
                         angle_limit=0,
                         junction_angle_limit=0,
                         huc_seg_river_angle_limit=0):
    """Smooth out any sharp angles in the river tree and between river tree and watershed boundary

    Note all angle tolerances are in degrees.
    """
    for node in river.preOrder():
        if angle_limit is not None:
            remove_sharp_angles_from_seg(node, angle_limit=angle_limit)  # from internal segments
            assert (node.is_locally_continuous())

        if len(node.children) != 0:
            # at junctions, angle between parent and child node
            if angle_limit is not None:
                treat_node_junctions_for_sharp_angles(node, angle_limit=angle_limit)
                assert (node.is_locally_continuous())

            # angle between two children (how often can we have >2 children??)
            if junction_angle_limit is not None:
                remove = True
                while remove:
                    remove = treat_small_angle_between_child_nodes(node,
                                                                   angle_limit=junction_angle_limit)
                    if not node.is_locally_continuous():
                        print(node.properties['ID'])
                    #assert (node.is_locally_continuous())
           

        assert (river.is_continuous())

    if huc_seg_river_angle_limit > 0:
        # checks angle betweebn huc segment and river at the outlets and rotate the part of huc segment as needed
        treat_small_angle_btw_river_huc(river, hucs, angle_limit=huc_seg_river_angle_limit)


def remove_sharp_angles_from_seg(node, angle_limit=10):
    """Smooth out any sharp angles in a reach segment."""
    seg = node.segment
    seg_coords = seg.coords[:]
    for i in range(len(seg_coords) - 2):
        seg_up = shapely.geometry.LineString([seg_coords[i], seg_coords[i + 1]])
        seg_down = shapely.geometry.LineString([seg_coords[i + 1], seg_coords[i + 2]])
        angle = watershed_workflow.river_mesh.angle_rivers_segs(ref_seg=seg_down, seg=seg_up)
        if angle > 360 - angle_limit or angle < angle_limit:
            logging.info(f"removing sharp angle in segment: {angle} for node {node.properties['ID']}")
            if len(seg_coords) > 3:
                new_point = shapely.geometry.Polygon(
                    [seg_coords[i], seg_coords[i + 1], seg_coords[i + 2]]).centroid
                seg_coords_new = seg_coords[:i] + new_point.coords[:] + seg_coords[i + 3:]
                node.segment = shapely.geometry.LineString(seg_coords_new)
                remove_sharp_angles_from_seg(node, angle_limit=angle_limit)
            else:
                seg_coords_new = [seg_coords[0], seg_coords[2]]
                node.segment = shapely.geometry.LineString(seg_coords_new)
            if i == 0:  # upstream end of the segment moved, update children segments
                for child in node.children:
                    child_seg_coords = child.segment.coords[:]
                    child_seg_coords[-1] = node.segment.coords[0]
                    child.segment = shapely.geometry.LineString(child_seg_coords)
            elif i == len(seg_coords
                          ) - 3:  # downstream end if the segment moved, update parent and siblings
                if node.parent != None:  # update parent
                    parent_seg_coords = node.parent.segment.coords[:]
                    parent_seg_coords[0] = node.segment.coords[-1]
                    node.parent.segment = shapely.geometry.LineString(parent_seg_coords)
                for sibling in node.siblings():  # update siblings
                    sibling_coords = sibling.segment.coords[:]
                    sibling_coords[-1] = node.segment.coords[-1]
                    sibling.segment = shapely.geometry.LineString(sibling_coords)
            assert (node.parent.is_continuous())


def treat_node_junctions_for_sharp_angles(node, angle_limit=10):
    """Smooth out junction angles.

    Moves the junction point to the centroid of the triangle formed by
    the junction point and the two neighboring points.  This is done
    recursively until the tolerance is met.
    """
    for child in node.children:
        # note, this must be done here.  If there are multiple
        # children and the first moves the parent, we have to look at
        # the moved parent, not the original parent.
        seg1 = node.segment
        seg2 = child.segment
        is_changed, seg1, seg2 = remove_sharp_angles_at_reach_junctions(seg1,
                                                                        seg2,
                                                                        angle_limit=angle_limit, 
                                                                        id=node.properties['ID'])

        if is_changed:
            node.segment = seg1
            child.segment = seg2
            for sibling in child.siblings():
                sibling_coords = sibling.segment.coords[:]
                sibling_coords[-1] = child.segment.coords[-1]
                sibling.segment = shapely.geometry.LineString(sibling_coords)


def remove_sharp_angles_at_reach_junctions(seg1, seg2, angle_limit=10, id=None):
    """Moves the common shared point of seg1 and seg2 to the centroid
    of the triangle formed by the junction point and the two
    neighboring points.  This is done recursively until the tolerance
    is met.
    """
    is_changed = False
    seg_up = shapely.geometry.LineString([seg2.coords[-2], seg2.coords[-1]])
    seg_down = shapely.geometry.LineString([seg1.coords[0], seg1.coords[1]])
    angle = watershed_workflow.river_mesh.angle_rivers_segs(ref_seg=seg_down, seg=seg_up)
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
        _, seg1, seg2 = remove_sharp_angles_at_reach_junctions(seg1, seg2, angle_limit=angle_limit)
        is_changed = True
    return is_changed, seg1, seg2


def treat_small_angle_between_child_nodes(node, angle_limit=10):
    """Zippers up junctions that have small angles.  Returns True if a section was removed.

    \  |       \ /
     ||   -->   |
      \          \

    """
    if len(node.children) > 1 and len(node.children) < 3:
        # note, we don't really have a good implementation for 3
        # children.  Could work on one if it shows up...
        seg1 = node.segment
        angles = []
        for child in node.children:
            seg2 = child.segment
            seg_up = shapely.geometry.LineString([seg2.coords[-2], seg2.coords[-1]])
            assert (watershed_workflow.utils.close(seg2.coords[-1], seg1.coords[0], 1.e-10))
            seg_down = shapely.geometry.LineString([seg1.coords[0], seg1.coords[1]])
            angle = watershed_workflow.river_mesh.angle_rivers_segs(ref_seg=seg_down, seg=seg_up)
            angles.append(angle)

        if abs(angles[1] - angles[0]) < angle_limit:
            logging.info(
                f"removing sharp angle between children: {abs(angles[1]-angles[0])} for node {node.properties['ID']}"
            )
            # zip up the last section of the two children, and give it to the parent
            new_junction = watershed_workflow.utils.midpoint(node.children[0].segment.coords[-2],
                                                             node.children[1].segment.coords[-2])
            new_node_coords = np.array([new_junction, ] + node.segment.coords[:])
            new_node_coords = watershed_workflow.utils.treat_segment_collinearity(new_node_coords)
            node.segment = shapely.geometry.LineString(new_node_coords)

            for child in node.children:
                child_coords = child.segment.coords[:]
                if len(child_coords) > 2:
                    # reach has > 2 points, so we can safely remove the last one
                    child_coords = child_coords[:-1]
                    child_coords[-1] = new_junction
                    child_coords = watershed_workflow.utils.treat_segment_collinearity(child_coords)
                    child.segment = shapely.geometry.LineString(child_coords)

                elif len(child_coords) == 2:
                    # reach only has two points
                    # remove it but move children's children to the right point
                    for grandchild in child.children:
                        # update coordinates of children
                        grandchild_seg_coords = grandchild.segment.coords[:]
                        grandchild_seg_coords[-1] = new_junction
                        grandchild.segment = shapely.geometry.LineString(grandchild_seg_coords)
                        node.addChild(grandchild)
                        print('this was executed')
                    child.remove()

            # we changed things -- return True so this can be called
            # repeatedly until all sections that have the close angle
            # are removed.
            return True
        else:
            # no change -- move along
            return False


def treat_small_angle_btw_river_huc(river, hucs, angle_limit=20):
    """Rotates the huc segment to increase the angle between huc segment and the river tree at the outlets as needed
    """

    river_mls = shapely.geometry.MultiLineString(list(river))

    # the intersection_point might be a MultiPoint with intersections at leaf tips
    # find all leaf tip points and exclude coinciding intersection points
    leaf_tips = [shapely.geometry.Point(leaf.segment.coords[0]) for leaf in river.leaf_nodes()]
    
    for i, seg in enumerate(hucs.segments):

        if seg.intersects(river_mls):  # does huc segment intersect with the river
            intersection_point = seg.intersection(river_mls)
            intersection_found = True
            if isinstance(intersection_point, shapely.geometry.multipoint.MultiPoint):
                non_coinciding_points = [p for p in intersection_point if not any(p.distance(point) < 1e-6 for point in leaf_tips)]
                # Check if there's exactly one point left after filtering
                if len(non_coinciding_points) == 1:
                    intersection_point = non_coinciding_points[0]
                elif len(non_coinciding_points) > 1:
                    print([point.coords[:] for point in non_coinciding_points])
                    raise ValueError("The intersection_point must be a single shapely.geometry.Point object")
                elif not non_coinciding_points:
                    intersection_found = False

            if intersection_found:
                parent_node = watershed_workflow.river_mesh.node_at_intersection(
                    intersection_point, river)
                if parent_node.parent == None:  # check if it is the outlet node for this river
                    outlet_junction = True
                else:
                    outlet_junction = False

                ind_intersection_point = watershed_workflow.river_mesh._indexPointInSeg(
                    parent_node.segment, intersection_point)

                # reference segment for angles
                if outlet_junction:
                    if len(parent_node.segment.coords) > 2:
                        ref_seg = shapely.geometry.LineString(
                            parent_node.segment.coords[ind_intersection_point - 2:])
                    else:
                        ref_seg = parent_node.segment
                else:
                    if len(parent_node.segment.coords) > 2:
                        ref_seg = shapely.geometry.LineString(
                            parent_node.segment.coords[ind_intersection_point:ind_intersection_point
                                                    + 2])
                    else:
                        ref_seg = parent_node.segment

                # angle of huc segment with the ref segment
                huc_seg_angle = watershed_workflow.river_mesh.angle_rivers_segs(ref_seg, seg)
                # angles of river with the huc segment
                river_angles = [
                    watershed_workflow.river_mesh.angle_rivers_segs(ref_seg, _seg) - huc_seg_angle
                    for _seg in parent_node.children
                ]

                angle_check = check_abs_smaller(
                    river_angles,
                    angle_limit)  # is any angle between river and huc segment smaller than the limit?

                if angle_check[0]:
                    river_angle = river_angles[angle_check[1]]
                    logging.info(f"removing sharp angle between river and huc: {river_angle} for node {parent_node.properties['ID']}")
                    if river_angle > 0:
                        rotate_angle = angle_limit - river_angle + 5
                    else:
                        rotate_angle = -(angle_limit - abs(river_angle) + 5)
                    rotated_seg = shapely.affinity.rotate(
                        seg, rotate_angle,
                        origin=intersection_point)  # rotated such that the angle is increased

                    seg_orientation_flag = np.argmin([
                        watershed_workflow.utils.distance(seg_end, intersection_point.coords[0])
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
                    hucs.segments[i] = seg


def check_abs_smaller(numbers, value):
    for index, num in enumerate(numbers):
        if abs(num) < value:
            return True, index
    return False, None
