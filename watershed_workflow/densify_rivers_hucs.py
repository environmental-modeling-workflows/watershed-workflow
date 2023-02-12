"""this code increases the resolution of river network and huc boundary in a controlled way using original river network and huc"""
import logging
import numpy as np
import math
from scipy import interpolate
import shapely

import watershed_workflow.utils


def densify_rivers(rivers,
                   rivers_raw=None,
                   use_original=False,
                   limit=100,
                   treat_collinearity=False):
    """Returns a list for densified rivers"""
    rivers = [
        densify_river(river,
                      river_raw,
                      use_original=use_original,
                      limit=limit,
                      treat_collinearity=treat_collinearity)
        for river, river_raw in zip(rivers, rivers_raw)
    ]

    mins = []
    for river in rivers:
        for line in river.dfs():
            coords = np.array(line.coords[:])
            dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
            mins.append(np.min(dz))
    logging.info(f"  river min seg length: {min(mins)}")
    logging.info(f"  river median seg length: {np.median(np.array(mins))}")

    return rivers


def densify_river(river, river_raw=None, use_original=False, limit=100, treat_collinearity=False):
    """This function traverse in the river tree and densify node.segments
    
    Parameters:
    -----------
    river: watershed_workflow.river_tree.RiverTree object
        river tree after simplifed (sparse points) that is to be densified 
    river_raw: watershed_workflow.river_tree.RiverTree object, optional
        original tree containing all the known points from NHDPlus 
    limit : int
        limit on the section length above which more points are added
    use_original: boolean
        flag for whether to resample from original river tree or just do simple interpolation
    treat_collinearity: boolean
        flag for whether to enforce non-colinearity. Collinear points in the segment create problem when 
        river corridor polynomial is created 

    Returns
    -------
    river_densified: watershed_workflow.river_tree.RiverTree object
        a densified river tree

    """
    
    river_densified = river.deep_copy()

    if 'NHDPlusID' in river.properties.keys():
  
        NHD_ids_raw = []
        for node in river_raw.preOrder():
            NHD_ids_raw.append(node.properties['NHDPlusID'])
    else:
         assert (len(river) == len(river_raw))

    for j, node in enumerate(river_densified.preOrder()):

        if 'NHDPlusID' in river.properties.keys():
            node_index_in_raw = NHD_ids_raw.index(node.properties['NHDPlusID'])
            node_ = list(river_raw.preOrder())[node_index_in_raw]
        else:
            node_ = list(river_raw.preOrder())[j]
    
        node.segment = densify_node_segments(node,
                                            node_,
                                            limit=limit,
                                            use_original=use_original,
                                            treat_collinearity=treat_collinearity)
    return river_densified


def densify_node_segments(node, node_raw, use_original=False, limit=100, treat_collinearity=False):
    """This function adds equally-spaced points in the reach-sections longer than the limit at a desired resolution
        potentially using original river tree
     
    Parameters:
     -----------
    node: node of a watershed_workflow.river_tree.RiverTree object
        node of a simplifed tree (sparse points) that is to be densified 
    node_raw: nodeof a watershed_workflow.river_tree.RiverTree object, optional
        node from the original tree containing all the known points from NHDPlus 
    limit : int
        limit on the section length above which more points are added
    use_original: boolean
        flag for whether to resample from original river tree or just do simple interpolation
    treat_collinearity: boolean
        flag for whether to enforce non-colinearity. Collinear points in the segment create problem when 
        river corridor polynomial is created 

    Returns
    -------
    node.segment: node.segment of a watershed_workflow.river_tree.RiverTree object
        a densified (inplace) node.segment
    """

    seg_coords = list(node.segment.coords)  # coordinates of node.segment to be densified
    seg_coords_ = list(
        node_raw.segment.coords)  # coordinates of node.segment from original river network
    seg_coords_densified = seg_coords.copy()  # segment coordinates densified
    j = 0
    for i in range(len(seg_coords) - 1):
        section_length = watershed_workflow.utils.distance(seg_coords[i], seg_coords[i + 1])
        if section_length > limit:
            number_new_points = int(section_length // limit)
            end_points = [seg_coords[i],
                          seg_coords[i + 1]]  # points betwen which more points will be added
            if use_original:
                new_points = interpolate_with_orig(end_points, seg_coords_, number_new_points)
            else:
                new_points = interpolate_simple(end_points, number_new_points)
            seg_coords_densified[j + 1:j + 1] = new_points
            j += number_new_points
        j += 1
    if treat_collinearity:
        seg_coords_densified = treat_segment_collinearity(seg_coords_densified)
    node.segment.coords = seg_coords_densified
    return node.segment


def densify_hucs(huc, huc_raw=None, rivers=None, use_original=False, limit_scales=None):
    """This function densify huc boundaries. The densification length scale either can be a constant value or a refinement 
    function where huc segment refinedment is greater for huc segments closer to the river tree
     
    Parameters:
     -----------
      hucs: watershed_workflow.split_hucs.SplitHUCs object
        huc to be densified 
      huc_raw: watershed_workflow.split_hucs.SplitHUCs object
        original huc with all the known points from NHDPlus 
      river: watershed_workflow.river_tree.RiverTree object
        to check the proximity of huc and river for refinement 
      use_original: boolean
        flag for whether to resample from original river tree or just do simple interpolation
      limit_scales: int or List 
        limit of section length above which more points are added, either a constant value or a list for step refinement 
        [near_distance, near_length_scale, far_distance, far_length_scale]

     Returns
    -------
    watershed_densified: watershed_workflow.split_hucs.SplitHUCs object
        a densified huc
    """

    for i, seg in enumerate(huc.segments): # densifying segment by segment 
        # find which original segment is this segment part of, so we can use it to resample
        coords = list(seg.coords)  
        if use_original:
            if huc_raw == None:
                raise RuntimeError('No original hucs found')

            for seg_orig in huc_raw.segments:  
                intersect = seg.intersection(seg_orig)
                if type(intersect) is shapely.geometry.MultiPoint and len(seg.intersection(seg_orig)) > 2:
                    seg_raw = seg_orig

            coords_raw = list(seg_raw.coords)

            if type(limit_scales) is list:
                # basic refine
                coords_densified_basic = densify_hucs_(coords,
                                                    coords_raw,
                                                    rivers,
                                                    limit_scales=limit_scales[-1])
                # adaptive refine
                coords_densified = densify_hucs_(coords_densified_basic,
                                                coords_raw,
                                                rivers,
                                                limit_scales=limit_scales)

            else:
                coords_densified = densify_hucs_(coords, coords_raw, rivers, limit_scales=limit_scales)

        else:
            coords_raw = None
            coords_densified = densify_hucs_(coords, coords_raw, rivers, limit_scales=limit_scales)

        huc.segments[i] = shapely.geometry.LineString(coords_densified)

    return huc
 

def densify_hucs_(coords, coords_raw=None, rivers=None, limit_scales=None):
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
            end_points = [coords[i], coords[i + 1]]  # points between which more points will be added
            if adaptive:
                new_points = interpolate_simple(end_points, number_new_points)
            else:
                new_points = interpolate_with_orig(end_points, coords_raw, number_new_points)

            coords_densified[j + 1:j + 1] = new_points
            j += number_new_points
        j += 1

    return coords_densified


def interpolate_with_orig(end_points, interp_data, n):
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


def interpolate_simple(end_points, n):
    """This function does not use any original data, just adds new equally spaced points based on linear interpolation"""
    xnew = np.linspace(end_points[0][0], end_points[1][0],
                       n + 2)[1:-1]  # new xs equally space between existing points
    ynew = np.linspace(end_points[0][1], end_points[1][1],
                       n + 2)[1:-1]  # new ys equally space between existing points
    new_points = [(xnew[k], ynew[k]) for k in range(n)]
    return new_points


def treat_segment_collinearity(segment_coords, tol=1e-6):
    """This functions removes collinearity from a node segment by making small pertubations orthogonal to the segment"""
    col_checks = []
    for i in range(0,
                   len(segment_coords)
                   - 2):  # traversing along the segment, checking 3 consecutive points at a time
        p0 = segment_coords[i]
        p1 = segment_coords[i + 1]
        p2 = segment_coords[i + 2]
        if check_collinearity(p0, p1, p2,
                              tol=tol):  # treating collinearity through a small pertubation
            del_ortho = 10 * tol  # shift in the middle point
            m = (p2[1] - p0[1]) / (p2[0] - p0[0])
            if abs(m) == float('inf'):
                m = 1e6
            del_y = del_ortho / (1 + m**2)**0.5
            del_x = -1 * del_ortho * m / (1 + m**2)**0.5
            p1 = (p1[0] + del_x, p1[1] + del_y)
            segment_coords[i + 1] = p1
        col_checks.append(check_collinearity(p0, p1, p2))
    assert (sum(col_checks) == 0)
    return segment_coords


def check_collinearity(p0, p1, p2, tol=1e-6):
    """this fucntion checks if three points are collinear for given tolerance value"""
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1*y2 - x2*y1) < tol


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
