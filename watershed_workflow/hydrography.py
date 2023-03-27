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


def make_global_tree(reaches, method='geometry', tol=_tol):
    """Constructs River objects from a list of reaches."""
    if method == 'hydroseq':
        return watershed_workflow.river_tree.River.construct_rivers_by_hydroseq(reaches)
    elif method == 'geometry':
        return watershed_workflow.river_tree.River.construct_rivers_by_geometry(reaches, tol)
    else:
        raise ValueError(
            "Invalid method for making Rivers, must be one of 'hydroseq' or 'geometry'")


def snap(hucs, rivers, tol=_tol, tol_triple_junctions=None, cut_intersections=False):
    """Snap HUCs to rivers."""
    assert (type(hucs) is watershed_workflow.split_hucs.SplitHUCs)
    assert (type(rivers) is list)
    assert (all(river.is_continuous() for river in rivers))
    list(hucs.polygons())

    if len(rivers) == 0:
        return True
    assert (len(rivers) > 0)
    for r in rivers:
        assert (len(r) > 0)

    if tol_triple_junctions is None:
        tol_triple_junctions = tol

    # snap boundary triple junctions to river endpoints
    logging.info("  snapping polygon segment boundaries to river endpoints")
    snap_polygon_endpoints(hucs, rivers, tol_triple_junctions)
    if not all(river.is_continuous() for river in rivers):
        logging.info("    ...resulted in inconsistent rivers!")
        return False
    try:
        list(hucs.polygons())
    except AssertionError:
        logging.info("    ...resulted in inconsistent HUCs")
        return False

    # snap endpoints of all rivers to the boundary if close
    # note this is a null-op on cases dealt with above
    logging.info("  snapping river endpoints to the polygon")
    for tree in rivers:
        snap_endpoints(tree, hucs, tol)
    if not all(river.is_continuous() for river in rivers):
        logging.info("    ...resulted in inconsistent rivers!")
        return False
    try:
        list(hucs.polygons())
    except AssertionError:
        logging.info("    ...resulted in inconsistent HUCs")
        return False

    if cut_intersections:
        cut_and_snap_crossings(hucs, rivers, tol)
    return rivers


def snap_waterbodies(hucs, waterbodies, tol=_tol, cut_intersections=True):
    """Snap waterbodies to HUCs."""
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


def cut_and_snap_crossings(hucs, rivers, tol=_tol):
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
            _cut_and_snap_crossing(hucs, river_node, tol)

    assert (all(river.is_consistent() for river in rivers))
    return rivers


def _cut_and_snap_crossing(hucs, reach_node, tol=_tol):
    """Helper function for cut_and_snap_crossings()"""
    r = reach_node.segment

    # first deal with crossings of the HUC exterior boundary -- in
    # this case, the reach segment gets split in two and the external
    # one is remoevd.
    for b, spine in hucs.boundaries.items():
        for s, seg_handle in spine.items():
            seg = hucs.segments[seg_handle]

            if seg.intersects(r):
                logging.info('intersection found')
                new_spine = watershed_workflow.utils.cut(seg, r, tol)
                new_reach_segs = watershed_workflow.utils.cut(r, seg, tol)
                try:
                    assert (len(new_reach_segs) == 1 or len(new_reach_segs) == 2)
                    assert (len(new_spine) == 1 or len(new_spine) == 2)
                    logging.info("  - cutting reach at external boundary of HUCs:")
                    logging.info(f"      split HUC boundary seg into {len(new_spine)} pieces")
                    logging.info(f"      split reach seg into {len(new_reach_segs)} pieces")

                    # which piece of the reach are we keeping?
                    if hucs.exterior().buffer(-tol).contains(
                            shapely.geometry.Point(new_reach_segs[0].coords[0])):
                        # keep the upstream (or only) reach seg
                        if len(new_reach_segs) == 2:
                            # confirm other/downstream reach is outside
                            assert (not hucs.exterior().contains(
                                shapely.geometry.Point(new_reach_segs[1].coords[-1])))
                        reach_node.segment = new_reach_segs[0]

                    elif len(new_reach_segs) == 2:
                        if hucs.exterior().buffer(-tol).contains(
                                shapely.geometry.Point(new_reach_segs[1].coords[-1])):
                            # keep the downstream reach seg, confirm upstream is outside
                            assert (not hucs.exterior().contains(
                                shapely.geometry.Point(new_reach_segs[0].coords[0])))
                            reach_node.segment = new_reach_segs[1]

                    # keep both pieces of a split huc boundary segment
                    # -- rename the first
                    hucs.segments[seg_handle] = new_spine[0]
                    if len(new_spine) > 1:
                        # -- add the first
                        assert (len(new_spine) == 2)
                        new_handle = hucs.segments.add(new_spine[1])
                        spine.add(new_handle)

                except AssertionError:
                    print('Error:')
                    reachc = np.array(reach_node.segment.coords)
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
                            hucs.exterior(), new_reach_segs[0])
                        print(r1c)
                        print(f'  r1 intersects with boundary? {inter}')
                    if len(new_reach_segs) > 1:
                        r2c = np.array(new_reach_segs[1].coords)
                        plt.plot(r2c[:, 0], r2c[:, 1], 'cx', markersize=40)
                        inter = watershed_workflow.utils.non_point_intersection(
                            hucs.exterior(), new_reach_segs[1])
                        print(f'  r2 intersects with boundary? {inter}')
                        print(r2c)
                        inter = hucs.exterior().intersection(new_reach_segs[1])
                        print(f'  r2 intersection = {inter}')

                    plt.show()
                    raise RuntimeError('Problem in cut_intersection')

                break

    # now deal with crossings of the HUC interior boundary -- in this
    # case, the reach segment is kept as one but the segment geometry
    # is snapped to make sure the intersection is exact
    for i, spine in hucs.intersections.items():
        for s, seg_handle in spine.items():
            seg = hucs.segments[seg_handle]

            if seg.intersects(r):
                new_spine = watershed_workflow.utils.cut(seg, r, tol)
                new_reach_segs = watershed_workflow.utils.cut(r, seg, tol)
                assert (len(new_reach_segs) == 1 or len(new_reach_segs) == 2)
                assert (len(new_spine) == 1 or len(new_spine) == 2)
                logging.info("  - snapping reach at internal boundary of HUCs")
                if (len(new_reach_segs) == 2):
                    reach_node.segment = shapely.geometry.LineString(
                        list(new_reach_segs[0].coords) + list(new_reach_segs[1].coords)[1:])
                else:
                    reach_node.segment = new_reach_segs[0]

                hucs.segments[seg_handle] = new_spine[0]
                if len(new_spine) > 1:
                    assert (len(new_spine) == 2)
                    new_handle = hucs.segments.add(new_spine[1])
                    spine.add(new_handle)
                break


def snap_polygon_endpoints(hucs, rivers, tol=_tol):
    """Snaps the endpoints of HUC segments to endpoints of rivers."""
    # make the kdTree of endpoints of all reaches
    coords1 = np.array([reach.coords[-1] for river in rivers for reach in river.depthFirst()])
    coords2 = np.array([reach.coords[0] for river in rivers for reach in river.leaves()])
    coords = np.concatenate([coords1, coords2], axis=0)

    # limit to x,y
    if (coords.shape[1] != 2):
        coords = coords[:, 0:2]
    # kdtree = scipy.spatial.cKDTree(coords)
    kdtree = cKDTree(coords)
    # for each segment of the HUC spine, find the river outlet that is
    # closest.  If within tolerance, move it
    for seg_handle, seg in hucs.segments.items():
        # check point 0, -1
        endpoints = np.array([seg.coords[0], seg.coords[-1]])
        # limit to x,y
        if (endpoints.shape[1] != 2):
            endpoints = endpoints[:, 0:2]
        dists, inds = kdtree.query(endpoints)
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
            hucs.segments[seg_handle] = shapely.geometry.LineString(new_seg)


def _closest_point(point, line, tol=_tol):
    """Determine the closest location on line to point.  If that point is
    further than tol or already a coordinate in the line, returns
    None.  Otherwise, return the location.
    """
    if watershed_workflow.utils.in_neighborhood(shapely.geometry.Point(point), line, tol):
        logging.debug("  - in neighborhood")
        nearest_p = watershed_workflow.utils.nearest_point(line, point)
        dist = watershed_workflow.utils.distance(nearest_p, point)
        logging.debug("  - nearest p = {0}, dist = {1}, tol = {2}".format(nearest_p, dist, tol))
        if dist < tol:
            if dist < 1.e-7:
                # filter case where the point is already there
                if any(watershed_workflow.utils.close(point, c) for c in line.coords):
                    return None
            return nearest_p
    return None


def snap_endpoints(tree, hucs, tol=_tol):
    """Snap river endpoints to huc segments and insert that point into
    the boundary.

    Note this is O(n^2), and could be made more efficient.
    """
    to_add = []
    for node in tree.preOrder():
        river = node.segment
        for b, component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):

            # note, this is done in two stages to allow it deal with both endpoints touching
            for s, seg_handle in component.items():
                seg = hucs.segments[seg_handle]
                #logging.debug("SNAP P0:")
                #logging.debug("  huc seg: %r"%seg.coords[:])
                #logging.debug("  river: %r"%river.coords[:])
                altered = False
                logging.debug("  - checking river coord: %r" % list(river.coords[0]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = _closest_point(river.coords[0], seg, tol)
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
                    while len(coords) > 2 and watershed_workflow.utils.distance(new_coord, coords[1]) < \
                          watershed_workflow.utils.distance(new_coord, coords[0]):
                        coords.pop(0)
                    coords[0] = new_coord
                    river = shapely.geometry.LineString(coords)
                    node.segment = river
                    to_add.append((seg_handle, component, 0, node))
                    break

            # second stage
            for s, seg_handle in component.items():
                seg = hucs.segments[seg_handle]
                # logging.debug("SNAP P1:")
                # logging.debug("  huc seg: %r"%seg.coords[:])
                # logging.debug("  river: %r"%river.coords[:])
                altered = False
                logging.debug("  - checking river coord: %r" % list(river.coords[-1]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = _closest_point(river.coords[-1], seg, tol)
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
                       watershed_workflow.utils.distance(new_coord, coords[-2]) < watershed_workflow.utils.distance(new_coord, coords[-1]):
                        coords.pop(-1)
                    coords[-1] = new_coord
                    river = shapely.geometry.LineString(coords)
                    node.segment = river
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
        if watershed_workflow.utils.close(p1[2].segment.coords[p1[1]], p2[2].segment.coords[p2[1]],
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
        seg = hucs.segments[seg_handle]
        # make a list of the coords and a flag to indicate a new
        # coord, then sort it by arclength along the segment.
        #
        # Note this needs special care if the seg is a loop, or else the endpoint gets sorted twice
        if not watershed_workflow.utils.close(seg.coords[0], seg.coords[-1]):
            new_coords = [[p[2].segment.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in seg.coords
                if not any(watershed_workflow.utils.close(c, nc, tol) for nc in new_coords)
            ]
            new_seg_coords = sorted(new_coords + old_coords,
                                    key=lambda a: seg.project(shapely.geometry.Point(a)))

            # determine the new coordinate indices
            breakpoint_inds = [i for i, (c, f) in enumerate(new_seg_coords) if f == 1]

        else:
            new_coords = [[p[2].segment.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in seg.coords[:-1]
                if not any(watershed_workflow.utils.close(c, nc, tol) for nc in new_coords)
            ]
            new_seg_coords = sorted(new_coords + old_coords,
                                    key=lambda a: seg.project(shapely.geometry.Point(a)))
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
        hucs.segments[seg_handle] = new_segs.pop(0)
        new_handles = hucs.segments.add_many(new_segs)
        insert_list[0][0].add_many(new_handles)

    return river


def simplify_and_merge(reaches, tol=_tol):
    """First pass to clean up hydro data"""
    logging.info("  quick simplify of reaches")
    reaches = shapely.ops.linemerge(reaches).simplify(tol)
    return list(reaches.geoms)


def cleanup(rivers, simp_tol=_tol, prune_tol=10, merge_tol=10):
    """Some hydrography data seems to get some random branches, typically
    quite short, that are nearly perfectly parallel to other, longer
    branches.  Surely this is a data error -- remove them.

    This returns rivers in a forest, not in a list.
    """
    # simplify
    if simp_tol is not None:
        for tree in rivers:
            simplify(tree, simp_tol)

    # prune short leaf branches and merge short interior reaches
    for tree in rivers:
        if merge_tol is not None:
            merge(tree, merge_tol)
        if merge_tol != prune_tol and prune_tol is not None:
            prune_by_segment_length(tree, prune_tol)


def prune_by_segment_length(tree, prune_tol=10):
    """Removes any leaf segments that are shorter than prune_tol"""
    for leaf in tree.leaf_nodes():
        if leaf.segment.length < prune_tol:
            logging.info("  ...cleaned leaf segment of length: %g at centroid %r" %
                         (leaf.segment.length, leaf.segment.centroid.coords[0]))
            leaf.removePreserveProperties()


def prune_river_by_area(river, area, prop='TotalDrainageAreaSqKm'):
    """Removes, IN PLACE, reaches whose total contributing area is less than area km^2.

    Note this requires NHDPlus data to have been used and the
    'TotalDrainageAreaSqKm' property to have been set.
    """
    count = 0
    for node in tree.preOrder():
        if node.properties[prop] < area:
            count += 1
            node.prune()
    return count


def prune_by_area(rivers, area):
    """Removes, IN PLACE, reaches whose total contributing area is less than area km^2.

    Note this requires NHDPlus data to have been used and the
    'TotalDrainageAreaSqKm' property to have been set.
    """
    logging.info(f"Pruning by total contributing area < {area}")
    count = 0
    sufficiently_big_rivers = []
    for river in rivers:
        if river.properties['TotalDrainageAreaSqKm'] >= area:
            count += prune_river_by_area(river, area)
            sufficiently_big_rivers.append(river)
    logging.info(f"... pruned {count}")
    return sufficiently_big_rivers


def prune_river_by_fractional_contributing_area(river, area_fraction, total_area=None):
    """Removes, IN PLACE, reaches whose CA/total_area < area_fraction.

    Note this requires NHDPlus data to have been used and the
    'TotalDrainageAreaSqKm' property to have been set.
    """
    if total_area is None:
        total_area = river.properties['TotalDrainageAreaSqKm']
    logging.info(f'... total contributing area = {total_area}')

    catchments_available = 'catchment' in river.properties.keys()
    count = 0
    first = True
    for node in river.preOrder():
        if first:
            # removing the top reach kills the whole river...
            first = False
            continue

        if node.properties['TotalDrainageAreaSqKm'] / total_area < area_fraction:
            logging.info(
                f"... removing: {node.properties['TotalDrainageAreaSqKm']} of {total_area}")
            count += 1

            node.prune()
    return count


def prune_by_fractional_contributing_area(rivers,
                                          area_fraction,
                                          total_area=None,
                                          prune_whole_rivers=True):
    """Removes, IN PLACE, reaches whose CA/total_area < area_fraction.

    Note this requires NHDPlus data to have been used and the
    'TotalDrainageAreaSqKm' property to have been set.
    """
    logging.info(f"Pruning by fractional contributing area < {area_fraction}")
    count = 0
    sufficiently_big_rivers = []
    if total_area is None:
        total_area = sum(river.properties['TotalDrainageAreaSqKm'] for river in rivers)

    for river in rivers:
        if (not prune_whole_rivers
            ) or river.properties['TotalDrainageAreaSqKm'] >= area_fraction * total_area:
            count += prune_river_by_fractional_contributing_area(river, area_fraction, total_area)
            sufficiently_big_rivers.append(river)
        else:
            logging.info(
                f"... pruning river with area {river.properties['TotalDrainageAreaSqKm']} of {total_area}"
            )
    logging.info(f"... pruned {count}")
    return sufficiently_big_rivers


def remove_divergences(rivers, remove_diversions=True, remove_braids=True):
    """Uses the NHDPlus property DIVERGENCE to remove divergences, IN PLACE.

    These can be of two types:

    Diversions are divergences that do not return to the stream
    network.  These will consist of individual rivers, typically with
    only one leaf reach, whose leaf reaches are all divergences.

    Braids are divergences that return to the river network, and so
    look like branches of a river tree, all of whose leaves are
    divergences.
    """
    if remove_diversions:
        logging.info("Remove diversions.")
        non_diversion = []
        for river in rivers:
            if all(leaf.properties['DivergenceCode'] == 2 for leaf in river.leaf_nodes()):
                logging.info(f"  ...remove river with {len(river)} reaches.")
            else:
                non_diversion.append(river)
        rivers = non_diversion

    if remove_braids:
        logging.info("Pruning divergences.")
        for river in rivers:
            for reach in river.leaf_nodes():
                if reach.properties['DivergenceCode'] == 2:
                    upstream_hydroseq = reach.properties['UpstreamMainPathHydroSeq']
                    try:
                        upstream = next(
                            r for r in river
                            if r.properties['UpstreamMainPathHydroSeq'] == upstream_hydroseq)
                    except StopIteration:
                        # a diversion, not a divergence
                        pass
                    else:
                        # a divergence, start removing until we meet a junction
                        done = reach.parent is None
                        count = 1
                        while not done:
                            parent = reach.parent
                            reach.removePreserveProperties()
                            count += 1
                            reach = parent
                            done = len(parent.children) > 0 or reach.parent is None
                        logging.info(f"  ... pruning braid of length {count}")
    return rivers


def filter_small_rivers(rivers, count):
    """Remove any rivers with fewer than count reaches."""
    logging.info(f"Removing rivers with fewer than {count} reaches.")
    new_rivers = []
    for river in rivers:
        ltree = len(river)
        if ltree < count:
            logging.info("  ...removing river with %d reaches" % ltree)
        else:
            new_rivers.append(river)
            logging.info("  ...keeping river with %d reaches" % ltree)
    return new_rivers


def merge(river, tol=_tol):
    """Remove inner branches that are short, combining branchpoints as needed. This function
    merge the "short" segment into the parent segment"""
    for node in list(river.preOrder()):
        if node.segment.length < tol and node.parent is not None:
            logging.info("  ...cleaned inner segment of length %g at centroid %r" %
                         (node.segment.length, node.segment.centroid.coords[0]))
            num_siblings = len(list(node.siblings()))
            node.parent.segment = shapely.geometry.LineString([node.segment.coords[0], ]
                                                              + node.parent.segment.coords[1:])
            if num_siblings != 0:
                for sibling in node.siblings():
                    sibling.segment = shapely.geometry.LineString(sibling.segment.coords[:-1]
                                                                  + [node.segment.coords[0], ])
            if 'area' in node.properties and 'area' in node.parent.properties:
                node.parent.properties['area'] += node.properties['area']

            if 'catchment' in node.properties and 'catchment' in node.parent.properties:
                node.parent.properties['catchment'] = shapely.ops.unary_union(
                    [node.properties['catchment'], node.parent.properties['catchment']])
            for child in node.children:
                node.parent.addChild(child)
            node.remove()


def simplify(river, tol=_tol):
    """Simplify, IN PLACE, all reaches."""
    for node in river.preOrder():
        if node.segment is not None:
            new_seg = node.segment.simplify(tol)
            assert (watershed_workflow.utils.close(new_seg.coords[0], node.segment.coords[0]))
            assert (watershed_workflow.utils.close(new_seg.coords[-1], node.segment.coords[-1]))
            node.segment = new_seg
