"""Functions for manipulating hydrography and river_tree.River objects"""

import math
import copy
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import itertools
import collections

import shapely.ops
import shapely.geometry

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.river_tree
import watershed_workflow.split_hucs
import watershed_workflow.plot

_tol = 0.1  # meters by default


def snap(hucs,
         rivers,
         tol=_tol,
         triple_junctions_tol=None,
         reach_endpoints_tol=None,
         cut_intersections=False):
    """Snap HUCs to rivers.

    Attempts to make rivers that intersect, or are close to
    intersecting with HUC boundaries intersect discretely, in that
    they share a common point on the boundary.

    This is the highest level function -- it calls many of the other
    functions in this namespace.

    """
    if triple_junctions_tol is None and tol is not None:
        triple_junctions_tol = 3 * tol
    if reach_endpoints_tol is None and tol is not None:
        reach_endpoints_tol = 2 * tol

    assert (type(hucs) is watershed_workflow.split_hucs.SplitHUCs)
    assert (type(rivers) is list)
    assert (all(river.isContinuous() for river in rivers))
    list(hucs.polygons())

    if len(rivers) == 0:
        return True
    assert (len(rivers) > 0)
    for r in rivers:
        assert (len(r) > 0)

    assert (all(r.isLocallyContinuous() for r in rivers))

    # snap boundary triple junctions to river endpoints
    if triple_junctions_tol is not None:
        logging.info("  snapping polygon segment boundaries to river endpoints")
        snapPolygonEndpoints(hucs, rivers, triple_junctions_tol)
        if not all(river.isContinuous() for river in rivers):
            logging.info("    ...resulted in inconsistent rivers!")
            return False
        try:
            list(hucs.polygons())
        except AssertionError:
            logging.info("    ...resulted in inconsistent HUCs")
            return False

    assert (all(r.isLocallyContinuous() for r in rivers))

    # snap endpoints of all rivers to the boundary if close
    # note this is a null-op on cases dealt with above
    if reach_endpoints_tol is not None:
        logging.info("  snapping river endpoints to the polygon")
        for tree in rivers:
            snapEndpoints(tree, hucs, reach_endpoints_tol)
        if not all(river.isContinuous() for river in rivers):
            logging.info("    ...resulted in inconsistent rivers!")
            return False
        try:
            list(hucs.polygons())
        except AssertionError:
            logging.info("    ...resulted in inconsistent HUCs")
            return False

    assert (all(r.isLocallyContinuous() for r in rivers))

    if cut_intersections:
        cutAndSnapCrossings(hucs, rivers, tol)
    assert (all(r.isLocallyContinuous() for r in rivers))

    # snapping can result in 0-length reaches
    cleanup(rivers)
    assert (all(r.isLocallyContinuous() for r in rivers))
    return rivers


def snapWaterbodies(hucs, waterbodies, tol=_tol, cut_intersections=True):
    """Snap waterbodies to HUCs.

    Attempts to make waterbodies that intersect or nearly intersect
    hucs intersect discretely, in that they share common point(s).

    """
    assert (type(hucs) is watershed_workflow.split_hucs.SplitHUCs)
    assert (type(waterbodies) is list)
    list(hucs.polygons())

    if len(waterbodies) == 0:
        return True
    assert (len(waterbodies) > 0)

    # snap endpoints of all rivers to the boundary if close
    # note this is a null-op on cases dealt with above
    logging.info("  snapping waterbody points to the HUC boundary")
    for i, wb in enumerate(waterbodies):
        for polygon in hucs.polygons():
            waterbodies[i] = shapely.ops.snap(wb, polygon, tol)


def cutAndSnapCrossings(hucs, rivers, tol=_tol):
    """If any reach crosses a HUC boundary:

    1. If it crosses an external boundary, cut the reach in two and
       discard the portion outside of the domain.

    2. If it crosses an internal boundary, ensure there is a
       coordinate the reach that is on the internal boundary.

    Either way, also ensure there is a coordinate on the HUC
    boundary at the crossing point.
    
    """
    logging.info("  cutting at crossings")
    # Note this is O(M*N) where M is number of huc segments and N is
    # number of reaches, and could be made more efficient.
    for tree in rivers:
        for river_node in tree.preOrder():
            _cutAndSnapCrossing(hucs, river_node, tol)

    cleanup(rivers)
    return rivers


def _cutAndSnapCrossing(hucs, reach_node, tol=_tol):
    """Helper function for cut_and_snap_crossings()"""
    r = reach_node.linestring

    # first deal with crossings of the HUC exterior boundary -- in
    # this case, the reach segment gets split in two and the external
    # one is remoevd.
    for b, spine in hucs.boundaries.items():
        for s, seg_handle in spine.items():
            seg = hucs.linestrings[seg_handle]

            if seg.intersects(r):
                logging.info('intersection found')
                new_spine, new_reach_segs = watershed_workflow.utils.cut(seg, r)
                try:
                    assert (len(new_reach_segs) == 1 or len(new_reach_segs) == 2)
                    assert (len(new_spine) == 1 or len(new_spine) == 2)
                    logging.info("  - cutting reach at external boundary of HUCs:")
                    logging.info(f"      split HUC boundary seg into {len(new_spine)} pieces")
                    logging.info(f"      split reach seg into {len(new_reach_segs)} pieces")

                    # which piece of the reach are we keeping?
                    if hucs.exterior.buffer(-tol).contains(
                            shapely.geometry.Point(new_reach_segs[0].coords[0])):
                        # keep the upstream (or only) reach seg
                        if len(new_reach_segs) == 2:
                            # confirm other/downstream reach is outside
                            assert (not hucs.exterior.contains(
                                shapely.geometry.Point(new_reach_segs[1].coords[-1])))
                        reach_node.linestring = new_reach_segs[0]

                    elif len(new_reach_segs) == 2:
                        if hucs.exterior.buffer(-tol).contains(
                                shapely.geometry.Point(new_reach_segs[1].coords[-1])):
                            # keep the downstream reach seg, confirm upstream is outside
                            assert (not hucs.exterior.contains(
                                shapely.geometry.Point(new_reach_segs[0].coords[0])))
                            reach_node.linestring = new_reach_segs[1]

                    # keep both pieces of a split huc boundary segment
                    # -- rename the first
                    hucs.linestrings[seg_handle] = new_spine[0]
                    if len(new_spine) > 1:
                        # -- add the first
                        assert (len(new_spine) == 2)
                        new_handle = hucs.linestrings.append(new_spine[1])
                        spine.append(new_handle)

                except AssertionError:
                    print('Error:')
                    reachc = np.array(reach_node.linestring.coords)
                    segc = np.array(seg.coords)
                    plt.plot(reachc[:, 0], reachc[:, 1], 'k--x')
                    plt.plot(segc[:, 0], segc[:, 1], 'k--+')

                    print(f'Reach split into {len(new_spine)} segments')
                    if len(new_spine) > 0:
                        r1c = np.array(new_spine[0].coords)
                        plt.plot(r1c[:, 0], r1c[:, 1], 'rx', markersize=40)
                    if len(new_spine) > 1:
                        r2c = np.array(new_spine[1].coords)
                        plt.plot(r2c[:, 0], r2c[:, 1], 'm+', markersize=40)

                    print(f'Reach split into {len(new_reach_segs)} segments')
                    if len(new_reach_segs) > 0:
                        r1c = np.array(new_reach_segs[0].coords)
                        plt.plot(r1c[:, 0], r1c[:, 1], 'b+', markersize=40)
                        inter = watershed_workflow.utils.non_point_intersection(
                            hucs.exterior, new_reach_segs[0])
                        print(r1c)
                        print(f'  r1 intersects with boundary? {inter}')
                    if len(new_reach_segs) > 1:
                        r2c = np.array(new_reach_segs[1].coords)
                        plt.plot(r2c[:, 0], r2c[:, 1], 'cx', markersize=40)
                        inter = watershed_workflow.utils.non_point_intersection(
                            hucs.exterior, new_reach_segs[1])
                        print(f'  r2 intersects with boundary? {inter}')
                        print(r2c)
                        inter = hucs.exterior.intersection(new_reach_segs[1])
                        print(f'  r2 intersection = {inter}')

                    plt.show()
                    raise RuntimeError('Problem in cut_intersection')

                break

    # now deal with crossings of the HUC interior boundary -- in this
    # case, the reach segment is kept as one but the segment geometry
    # is snapped to make sure the intersection is exact
    for i, spine in hucs.intersections.items():
        for s, seg_handle in spine.items():
            seg = hucs.linestrings[seg_handle]

            if seg.intersects(r):
                new_spine = watershed_workflow.utils.cut(seg, r, tol)
                new_reach_segs = watershed_workflow.utils.cut(r, seg, tol)
                assert (len(new_reach_segs) == 1 or len(new_reach_segs) == 2)
                assert (len(new_spine) == 1 or len(new_spine) == 2)
                logging.info("  - snapping reach at internal boundary of HUCs")
                if (len(new_reach_segs) == 2):
                    reach_node.linestring = shapely.geometry.LineString(
                        list(new_reach_segs[0].coords) + list(new_reach_segs[1].coords)[1:])
                else:
                    reach_node.linestring = new_reach_segs[0]

                hucs.linestrings[seg_handle] = new_spine[0]
                if len(new_spine) > 1:
                    assert (len(new_spine) == 2)
                    new_handle = hucs.linestrings.append(new_spine[1])
                    spine.append(new_handle)
                break


def snapPolygonEndpoints(hucs, rivers, tol=_tol):
    """Snaps the endpoints of HUC segments to endpoints of rivers."""
    # make the kdTree of endpoints of all reaches
    coords1 = np.array([reach.linestring.coords[-1] for river in rivers for reach in river.preOrder()])
    coords2 = np.array([reach.linestring.coords[0] for river in rivers for reach in river.leaf_nodes])
    coords = np.concatenate([coords1, coords2], axis=0)

    # limit to x,y
    if (coords.shape[1] != 2):
        coords = coords[:, 0:2]

    debug_point = shapely.geometry.Point([-581678.5238123547, -378867.813358335])

    kdtree = cKDTree(coords)
    # for each segment of the HUC spine, find the river outlet that is
    # closest.  If within tolerance, move it
    for seg_handle, seg in hucs.linestrings.items():
        # check point 0, -1
        endpoints = np.array([seg.coords[0], seg.coords[-1]])
        # limit to x,y
        if (endpoints.shape[1] != 2):
            endpoints = endpoints[:, 0:2]
        dists, inds = kdtree.query(endpoints)

        #### DEBUG CODE #####
        if debug_point.distance(shapely.geometry.Point(seg.coords[0])) < 10000:
            dist = debug_point.distance(shapely.geometry.Point(seg.coords[0]))
            print(
                f'found a huc seg beginpoint: {seg.coords[0]} with distance {dist} < tol = {tol}?')

        elif debug_point.distance(shapely.geometry.Point(seg.coords[-1])) < 10000:
            dist = debug_point.distance(shapely.geometry.Point(seg.coords[-1]))
            print(f'found a huc seg endpoint: {seg.coords[-1]} with distance {dist} < tol = {tol}?')
        #### END DEBUG CODE #####

        if dists.min() < tol:
            new_seg = list(seg.coords)
            if dists[0] < tol:
                new_seg[0] = coords[inds[0]]
                logging.debug(
                    f"  Moving HUC segment point 0,1: {list(seg.coords)[0]}, {list(seg.coords)[-1]}"
                )
                logging.debug("        point 0 to river at %r" % list(new_seg[0]))

            if dists[1] < tol:
                new_seg[-1] = coords[inds[1]]
                logging.debug(
                    f"  Moving HUC segment point 0,1: {list(seg.coords)[0]}, {list(seg.coords)[-1]}"
                )
                logging.debug("        point -1 to river at %r" % list(new_seg[-1]))
            hucs.linestrings[seg_handle] = shapely.geometry.LineString(new_seg)


def _closestPoint(point, line, tol=_tol):
    """Determine the closest location on line to point.  If that point is
    further than tol or already a coordinate in the line, returns
    None.  Otherwise, return the location.
    """
    if watershed_workflow.utils.inNeighborhood(shapely.geometry.Point(point), line, tol):
        logging.debug("  - in neighborhood")
        nearest_p = watershed_workflow.utils.findNearestPoint(point, line)
        dist = watershed_workflow.utils.computeDistance(nearest_p, point)
        logging.debug("  - nearest p = {0}, dist = {1}, tol = {2}".format(nearest_p, dist, tol))
        if dist < tol:
            if dist < 1.e-7:
                # filter case where the point is already there
                if any(watershed_workflow.utils.isClose(point, c) for c in line.coords):
                    return None
            return nearest_p
    return None


def snapEndpoints(tree, hucs, tol=_tol):
    """Snap river endpoints to huc segments and insert that point into
    the boundary.

    Note this is O(n^2), and could be made more efficient.
    """
    to_add = []
    for node in tree.preOrder():
        river = node.linestring
        for b, component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):

            # note, this is done in two stages to allow it deal with both endpoints touching
            for s, seg_handle in component.items():
                seg = hucs.linestrings[seg_handle]
                #logging.debug("SNAP P0:")
                #logging.debug("  huc seg: %r"%seg.coords[:])
                #logging.debug("  river: %r"%river.coords[:])
                altered = False
                logging.debug("  - checking river coord: %r" % list(river.coords[0]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = _closestPoint(river.coords[0], seg, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord != None:
                    logging.info("    snapped river: %r to %r" % (river.coords[0], new_coord))

                    # move new_coord onto an existing segment coord
                    dist = np.linalg.norm(np.array(seg.coords) - np.expand_dims(new_coord, 0),
                                          2,
                                          axis=1)
                    assert (len(dist) == len(seg.coords))
                    assert (len(dist.shape) == 1)
                    i = int(np.argmin(dist))
                    if (dist[i] < tol):
                        new_coord = seg.coords[i]

                    # remove points that are closer
                    coords = list(river.coords)
                    done = False
                    while len(coords) > 2 and watershed_workflow.utils.computeDistance(new_coord, coords[1]) < \
                          watershed_workflow.utils.computeDistance(new_coord, coords[0]):
                        coords.pop(0)
                    coords[0] = new_coord
                    river = shapely.geometry.LineString(coords)
                    node.linestring = river
                    to_add.append((seg_handle, component, 0, node))
                    break

            # second stage
            for s, seg_handle in component.items():
                seg = hucs.linestrings[seg_handle]
                # logging.debug("SNAP P1:")
                # logging.debug("  huc seg: %r"%seg.coords[:])
                # logging.debug("  river: %r"%river.coords[:])
                altered = False
                logging.debug("  - checking river coord: %r" % list(river.coords[-1]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = _closestPoint(river.coords[-1], seg, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord != None:
                    logging.info("  - snapped river: %r to %r" % (river.coords[-1], new_coord))

                    # move new_coord onto an existing segment coord
                    dist = np.linalg.norm(np.array(seg.coords) - np.expand_dims(new_coord, 0),
                                          2,
                                          axis=1)
                    assert (len(dist) == len(seg.coords))
                    assert (len(dist.shape) == 1)
                    i = int(np.argmin(dist))
                    if (dist[i] < tol):
                        new_coord = seg.coords[i]

                    # remove points that are closer
                    coords = list(river.coords)
                    done = False
                    while len(coords) > 2 and \
                       watershed_workflow.utils.computeDistance(new_coord, coords[-2]) < watershed_workflow.utils.computeDistance(new_coord, coords[-1]):
                        coords.pop(-1)
                    coords[-1] = new_coord
                    river = shapely.geometry.LineString(coords)
                    node.linestring = river
                    to_add.append((seg_handle, component, -1, node))
                    break

    # find the list of points to add to a given segment
    to_add_dict = dict()
    for seg_handle, component, endpoint, node in to_add:
        if seg_handle not in to_add_dict.keys():
            to_add_dict[seg_handle] = list()
        to_add_dict[seg_handle].append((component, endpoint, node))

    # find the set of points to add to each given segment
    def equal(p1, p2):
        if watershed_workflow.utils.isClose(p1[2].linestring.coords[p1[1]], p2[2].linestring.coords[p2[1]],
                                          1.e-5):
            assert (p1[0] == p2[0])
            return True
        else:
            return False

    to_add_dict2 = dict()
    for seg_handle, insert_list in to_add_dict.items():
        new_list = []
        for p1 in insert_list:
            if (all(not equal(p1, p2) for p2 in new_list)):
                new_list.append(p1)
        to_add_dict2[seg_handle] = new_list

    # add these points to the segment
    for seg_handle, insert_list in to_add_dict2.items():
        seg = hucs.linestrings[seg_handle]
        # make a list of the coords and a flag to indicate a new
        # coord, then sort it by arclength along the segment.
        #
        # Note this needs special care if the seg is a loop, or else the endpoint gets sorted twice
        if not watershed_workflow.utils.isClose(seg.coords[0], seg.coords[-1]):
            new_coords = [[p[2].linestring.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in seg.coords
                if not any(watershed_workflow.utils.isClose(c, nc, tol) for nc in new_coords)
            ]
            new_seg_coords = sorted(new_coords + old_coords,
                                    key=lambda a: seg.project(shapely.geometry.Point(a[0])))

            # determine the new coordinate indices
            breakpoint_inds = [i for i, (c, f) in enumerate(new_seg_coords) if f == 1]

        else:
            new_coords = [[p[2].linestring.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in seg.coords[:-1]
                if not any(watershed_workflow.utils.isClose(c, nc, tol) for nc in new_coords)
            ]
            new_seg_coords = sorted(new_coords + old_coords,
                                    key=lambda a: seg.project(shapely.geometry.Point(a[0])))
            breakpoint_inds = [i for i, (c, f) in enumerate(new_seg_coords) if f == 1]
            assert (len(breakpoint_inds) > 0)
            new_seg_coords = new_seg_coords[breakpoint_inds[0]:] + new_seg_coords[
                0:breakpoint_inds[0] + 1]
            new_seg_coords[0][1] = 0
            new_seg_coords[-1][1] = 0
            breakpoint_inds = [i for i, (c, f) in enumerate(new_seg_coords) if f == 1]

        # now break into new segments
        new_segs = []
        ind_start = 0
        for ind_end in breakpoint_inds:
            assert (ind_end != 0)
            new_segs.append(
                shapely.geometry.LineString([c
                                             for (c, f) in new_seg_coords[ind_start:ind_end + 1]]))
            ind_start = ind_end

        assert (ind_start < len(new_seg_coords) - 1)
        new_segs.append(
            shapely.geometry.LineString([tuple(c) for (c, f) in new_seg_coords[ind_start:]]))

        # put all new_segs into the huc list.  Note insert_list[0][0] is the component
        hucs.linestrings[seg_handle] = new_segs.pop(0)
        new_handles = hucs.linestrings.extend(new_segs)
        insert_list[0][0].extend(new_handles)

    return river


def cleanup(rivers, simp_tol=None, prune_tol=_tol, merge_tol=_tol, preserve_catchments=False):
    """Cleans rivers in place by:

    1. simplifying to tol
    2. pruning all leaf nodes of length < prune_tol
    3. merging all internal nodes of length < merge_tol
    """
    # simplify
    if simp_tol is not None:
        for tree in rivers:
            watershed_workflow.river_tree.simplify(tree, simp_tol)

    assert (all([river.isConsistent() for river in rivers]))
    for river in rivers:
        assert (river.isContinuous())

    # prune short leaf branches and merge short interior reaches
    for tree in rivers:
        if merge_tol is not None:
            watershed_workflow.river_tree.mergeShortReaches(tree, merge_tol)
        if merge_tol != prune_tol and prune_tol is not None:
            watershed_workflow.river_tree.pruneBySegmentLength(tree, prune_tol, preserve_catchments)

    assert (all(river.isContinuous() for river in rivers))

    tols = [t for t in [prune_tol, merge_tol] if t is not None]
    if len(tols) > 0:
        tol = min(tols)
        for river in rivers:
            for r in river:
                assert (r.linestring.length > tol)


