"""Functions for manipulating combinations of RiverTree and SplitHUCs objects"""

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

_tol = 0.1  # meters by default


def findOutletsByCrossings(hucs, river, tol=None, debug_plot=False):
    """For each HUC, find all outlets using a river network's crossing points."""
    if tol is None:
        tol = 10
    # next determine the outlet, and all boundary edges within x m of that outlet
    polygons = list(hucs.polygons())
    poly_crossings = []
    for i_sub, poly in enumerate(polygons):
        my_crossings = []
        for reach in river.preOrder():
            if poly.exterior.intersects(reach.segment):
                my_crossings.append(poly.exterior.intersection(reach.segment))

        # cluster my_crossings to make sure that multiple crossings are only counted once
        my_crossing_centroids = []
        for crossing in my_crossings:
            my_crossing_centroids.append([crossing.centroid.xy[0][0], crossing.centroid.xy[1][0]])
        my_crossing_centroids = np.array(my_crossing_centroids)
        if len(my_crossing_centroids) > 1:
            clusters, cluster_centroids = watershed_workflow.utils.cluster(
                my_crossing_centroids, tol)
        else:
            cluster_centroids = my_crossing_centroids
        poly_crossings.append(cluster_centroids)

    logging.info("Crossings by Polygon:")
    for i, c in enumerate(poly_crossings):
        logging.info(f'  Polygon {i}')
        for p in c:
            logging.info(f'    crossing: {p}')

    # unravel the clusters
    all_crossings = [c for p in poly_crossings for c in p]

    # cluster crossings that are within tolerance across polygons
    crossings_clusters_indices, crossings_clusters_centroids = \
        watershed_workflow.utils.cluster(all_crossings, tol)

    # now group cluster ids by polygon and polygon ids by cluster
    poly_cluster_indices = dict()
    cluster_poly_indices = collections.defaultdict(list)
    lcv = 0
    for lcv_poly, pc in enumerate(poly_crossings):
        my_inds = []
        for c in pc:
            my_inds.append(crossings_clusters_indices[lcv])
            lcv += 1
        poly_cluster_indices[lcv_poly] = my_inds
        for ci in my_inds:
            cluster_poly_indices[ci].append(lcv_poly)

    # assert equivalent
    for pi, clusters in poly_cluster_indices.items():
        for ci in clusters:
            assert (pi in cluster_poly_indices[ci])
    for ci, polys in cluster_poly_indices.items():
        for pi in polys:
            assert (ci in poly_cluster_indices[pi])

    # create a tree, recursively finding all polygons with only
    # one crossing -- this must be an outlet -- then removing it
    # from the list, hopefully leaving a downstream polygon with
    # only one outlet.  This must be done N iterations, where N is
    # the maximal number of polygons crossed from 0th order to
    # maximal order.
    logging.info('Constructing outlet list')
    outlets = dict()
    inlets = collections.defaultdict(list)
    itercount = 0
    done = False
    while not done:
        logging.info(f'Iteration = {itercount}')
        logging.info(f'-----------------')
        new_outlets = dict()

        # look for polygons with only one crossing -- this must be an outlet.
        for pi, clusters in poly_cluster_indices.items():
            if len(clusters) == 1 and pi not in outlets:
                # only one crossing cluster, this is the outlet
                cluster_id = clusters[0]
                new_outlets[pi] = cluster_id
                cluster_poly_indices[cluster_id].remove(pi)
                logging.info(
                    f' poly outlet {pi} : {cluster_id}, {crossings_clusters_centroids[cluster_id]}')
                last_outlet = cluster_id
                last_outlet_poly = pi

        # look for clusters with only one poly -- this must be an inlet
        to_remove = []
        for ci, polys in cluster_poly_indices.items():
            if len(polys) == 1:
                poly_id = polys[0]
                poly_cluster_indices[poly_id].remove(ci)
                logging.info(f' poly inlet {poly_id} : {ci}, {crossings_clusters_centroids[ci]}')
                to_remove.append(ci)
                inlets[poly_id].append(ci)
        for ci in to_remove:
            cluster_poly_indices.pop(ci)

        if debug_plot and len(new_outlets) > 0:
            import watershed_workflow.plot
            fig, ax = watershed_workflow.plot.get_ax(None)
            watershed_workflow.plot.shplys(polygons, None, color='k', ax=ax)
            watershed_workflow.plot.rivers([river, ], None, color='b', ax=ax)
            for pi, ci in outlets.items():
                outlet = crossings_clusters_centroids[ci]
                ax.scatter([outlet[0], ], [outlet[1], ], s=100, c='b', marker='o')
            for pi, ci in new_outlets.items():
                outlet = crossings_clusters_centroids[ci]
                ax.scatter([outlet[0], ], [outlet[1], ], s=100, c='r', marker='o')
            for ci in range(len(crossings_clusters_centroids)):
                if ci not in outlets.values() and ci not in new_outlets.values():
                    crossing = crossings_clusters_centroids[ci]
                    ax.scatter([crossing[0], ], [crossing[1], ], s=100, c='k', marker='o')
            from matplotlib import pyplot as plt
            ax.set_title(f'Outlets after iteration {itercount}')
            plt.show()

        outlets.update(new_outlets)
        itercount += 1
        done = itercount > 50 or len(outlets) == len(polygons) or len(new_outlets) == 0

    logging.info(
        f'last outlet is {last_outlet} in polygon {last_outlet_poly} at {crossings_clusters_centroids[last_outlet]}'
    )

    # create the output
    outlet_locs = {}
    inlet_locs = {}
    for pi, ci in outlets.items():
        outlet = crossings_clusters_centroids[ci]
        outlet_locs[pi] = shapely.geometry.Point(outlet[0], outlet[1])
    for pi, cis in inlets.items():
        my_inlet_locs = []
        for ci in cis:
            inlet = crossings_clusters_centroids[ci]
            my_inlet_locs.append(shapely.geometry.Point(inlet[0], inlet[1]))
        inlet_locs[pi] = my_inlet_locs

    last_outlet_p = crossings_clusters_centroids[last_outlet]
    last_outlet_loc = shapely.geometry.Point(last_outlet_p[0], last_outlet_p[1])

    hucs.exterior_outlet = last_outlet_loc
    hucs.polygon_outlets = outlet_locs


def findOutletsByElevation(hucs, crs, elev_raster, elev_raster_profile):
    """Find outlets by the minimum elevation on the boundary."""
    import watershed_workflow
    exterior = hucs.exterior.exterior
    mesh_points = np.array([exterior.coords])[0, :, :]
    mesh_points = watershed_workflow.elevate(mesh_points, crs, elev_raster, elev_raster_profile)
    i = np.argmin(mesh_points[:, 2])
    hucs.exterior_outlet = shapely.geometry.Point(mesh_points[i, 0], mesh_points[i, 1])

    outlets = []
    for poly in hucs.polygons():
        mesh_points = np.array([poly.exterior.coords])[0, :, :]
        mesh_points = watershed_workflow.elevate(mesh_points, crs, elev_raster, elev_raster_profile)
        i = np.argmin(mesh_points[:, 2])
        outlets.append(shapely.geometry.Point(mesh_points[i, 0], mesh_points[i, 1]))
    hucs.polygon_outlets = outlets


def findOutletsByHydroseq(hucs, river, tol=0):
    """Find outlets using the HydroSequence VAA of NHDPlus.

    Finds the minimum hydroseq reach in each HUC, and intersects that
    with the boundary to find the outlet.
    """
    polygons = list(hucs.polygons())
    polygon_outlets = [None for poly in hucs.polygons()]

    # iterate over the reaches, sorted by hydrosequence, looking for
    # the first one that intersects the polygon boundary.
    assert (river.is_hydroseq_consistent())
    reaches = sorted(river.preOrder(), key=lambda r: r.properties['HydrologicSequence'])
    if tol > 0:
        reaches = [r.segment.buffer(tol) for r in reaches]
    else:
        reaches = [r.segment for r in reaches]
    first = True

    poly_ids = [(i, poly) for (i, poly) in enumerate(polygons)]
    for lcv, reach in enumerate(reaches):
        try:
            j, (poly_i, poly) = next((j,(i,poly)) for (j,(i,poly)) in enumerate(poly_ids) \
                                     if poly.intersects(reach))
        except StopIteration:
            continue
        else:
            # find the intersection
            logging.debug(f'hydroseq {lcv} is a match for polygon {poly_i}')
            intersect = poly.exterior.intersection(reach)
            if intersect.is_empty:
                # find the nearest point instead
                intersect = shapely.ops.nearest_points(poly.exterior, reach)[0]
            else:
                intersect = intersect.centroid

            if first:
                hucs.exterior_outlet = intersect
                first = False
            polygon_outlets[poly_i] = intersect
            poly_ids.pop(j)
        if len(poly_ids) == 0:
            break

    hucs.polygon_outlets = polygon_outlets


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
    hucs.update()

    if len(rivers) == 0:
        return True
    assert (len(rivers) > 0)
    for r in rivers:
        assert (len(r) > 0)

    assert(all(r.isLocallyContinuous() for r in rivers))
        
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

    assert(all(r.isLocallyContinuous() for r in rivers))

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

    assert(all(r.isLocallyContinuous() for r in rivers))

    if cut_intersections:
        cutAndSnapCrossings(hucs, rivers, tol)
    assert(all(r.isLocallyContinuous() for r in rivers))

    # snapping can result in 0-length reaches
    cleanup(rivers)
    assert(all(r.isLocallyContinuous() for r in rivers))
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
    """Helper function for cutAndSnapCrossings()"""
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
                    if hucs.exterior.buffer(-tol).contains(
                            shapely.geometry.Point(new_reach_segs[0].coords[0])):
                        # keep the upstream (or only) reach seg
                        if len(new_reach_segs) == 2:
                            # confirm other/downstream reach is outside
                            assert (not hucs.exterior.contains(
                                shapely.geometry.Point(new_reach_segs[1].coords[-1])))
                        reach_node.segment = new_reach_segs[0]

                    elif len(new_reach_segs) == 2:
                        if hucs.exterior.buffer(-tol).contains(
                                shapely.geometry.Point(new_reach_segs[1].coords[-1])):
                            # keep the downstream reach seg, confirm upstream is outside
                            assert (not hucs.exterior.contains(
                                shapely.geometry.Point(new_reach_segs[0].coords[0])))
                            reach_node.segment = new_reach_segs[1]

                    # keep both pieces of a split huc boundary segment
                    # -- rename the first
                    hucs.segments[seg_handle] = new_spine[0]
                    if len(new_spine) > 1:
                        # -- add the first
                        assert (len(new_spine) == 2)
                        new_handle = hucs.segments.append(new_spine[1])
                        spine.append(new_handle)

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
                    new_handle = hucs.segments.append(new_spine[1])
                    spine.append(new_handle)
                break


def snapPolygonEndpoints(hucs, rivers, tol=_tol):
    """Snaps the endpoints of HUC segments to endpoints of rivers."""
    # make the kdTree of endpoints of all reaches
    coords1 = np.array([reach.segment.coords[-1] for river in rivers for reach in river.preOrder()])
    coords2 = np.array([reach.segment.coords[0] for river in rivers for reach in river.leaf_nodes])
    coords = np.concatenate([coords1, coords2], axis=0)

    # limit to x,y
    if (coords.shape[1] != 2):
        coords = coords[:, 0:2]

    debug_point = shapely.geometry.Point([-581678.5238123547, -378867.813358335])

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
            hucs.segments[seg_handle] = shapely.geometry.LineString(new_seg)


def snapEndpoints(tree, hucs, tol=_tol):
    """Snap river endpoints to huc segments and insert that point into
    the boundary.

    Note this is O(n^2), and could be made more efficient.
    """
    to_add = []
    for node in tree.preOrder():
        reach = node.segment
        for b, component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):

            # note, this is done in two stages to allow it deal with both endpoints touching
            for s, seg_handle in component.items():
                seg = hucs.segments[seg_handle]
                #logging.debug("SNAP P0:")
                #logging.debug("  huc seg: %r"%seg.coords[:])
                #logging.debug("  reach: %r"%reach.coords[:])
                altered = False
                logging.debug("  - checking reach coord: %r" % list(reach.coords[0]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = watershed_workflow.utils.findNearestPoint(reach.coords[0], seg, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord != None:
                    logging.debug("    snapped reach: %r to %r" % (reach.coords[0], new_coord))

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
                    coords = list(reach.coords)
                    done = False
                    while len(coords) > 2 and watershed_workflow.utils.computeDistance(new_coord, coords[1]) < \
                          watershed_workflow.utils.computeDistance(new_coord, coords[0]):
                        coords.pop(0)
                    coords[0] = new_coord
                    reach = shapely.geometry.LineString(coords)
                    node.segment = reach
                    to_add.append((seg_handle, component, 0, node))
                    break

            # second stage
            for s, seg_handle in component.items():
                seg = hucs.segments[seg_handle]
                # logging.debug("SNAP P1:")
                # logging.debug("  huc seg: %r"%seg.coords[:])
                # logging.debug("  reach: %r"%reach.coords[:])
                altered = False
                logging.debug("  - checking reach coord: %r" % list(reach.coords[-1]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = watershed_workflow.utils.findNearestPoint(reach.coords[-1], seg, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord != None:
                    logging.debug("  - snapped reach: %r to %r" % (reach.coords[-1], new_coord))

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
                    coords = list(reach.coords)
                    done = False
                    while len(coords) > 2 and \
                       watershed_workflow.utils.computeDistance(new_coord, coords[-2]) < watershed_workflow.utils.computeDistance(new_coord, coords[-1]):
                        coords.pop(-1)
                    coords[-1] = new_coord
                    reach = shapely.geometry.LineString(coords)
                    node.segment = reach
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
        if watershed_workflow.utils.isClose(p1[2].segment.coords[p1[1]], p2[2].segment.coords[p2[1]],
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
        if not watershed_workflow.utils.isClose(seg.coords[0], seg.coords[-1]):
            new_coords = [[p[2].segment.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in seg.coords
                if not any(watershed_workflow.utils.isClose(c, nc, tol) for nc in new_coords)
            ]
            new_seg_coords = sorted(new_coords + old_coords,
                                    key=lambda a: seg.project(shapely.geometry.Point(a[0])))

            # determine the new coordinate indices
            breakpoint_inds = [i for i, (c, f) in enumerate(new_seg_coords) if f == 1]

        else:
            new_coords = [[p[2].segment.coords[p[1]], 1] for p in insert_list]
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
        hucs.segments[seg_handle] = new_segs.pop(0)
        new_handles = hucs.segments.extend(new_segs)
        insert_list[0][0].extend(new_handles)


def cleanup(rivers, simp_tol=None, prune_tol=_tol, merge_tol=_tol, preserve_catchments=False):
    """Cleans rivers in place by:

    1. simplifying to tol
    2. pruning all leaf nodes of length < prune_tol
    3. merging all internal nodes of length < merge_tol
    """
    # simplify
    if simp_tol is not None:
        for tree in rivers:
            simplify(tree, simp_tol)

    assert (all([river.isConsistent() for river in rivers]))
    for river in rivers:
        assert (river.isContinuous())

    # prune short leaf branches and merge short interior reaches
    for tree in rivers:
        if merge_tol is not None:
            merge(tree, merge_tol)
        if merge_tol != prune_tol and prune_tol is not None:
            pruneBySegmentLength(tree, prune_tol, preserve_catchments)

    assert (all(river.isContinuous() for river in rivers))

    tols = [t for t in [prune_tol, merge_tol] if t is not None]
    if len(tols) > 0:
        tol = min(tols)
        for river in rivers:
            for r in river:
                assert (r.segment.length > tol)


def pruneBySegmentLength(tree, prune_tol=10, preserve_catchments=False):
    """Removes any leaf segments that are shorter than prune_tol"""
    for leaf in tree.leaf_nodes:
        if leaf.segment.length < prune_tol:
            logging.info("  ...cleaned leaf segment of length: %g at centroid %r" %
                         (leaf.segment.length, leaf.segment.centroid.coords[0]))
            leaf.prune(preserve_catchments)


def pruneRiverByArea(river, area, prop='DivergenceRoutedDrainAreaSqKm', preserve_catchments=False):
    """Removes, IN PLACE, reaches whose total contributing area is less than area km^2.

    Note this requires NHDPlus data to have been used and the
    'DivergenceRoutedDrainAreaSqKm' property (or whatever is selected) to have been set.
    """
    count = 0
    for node in river.preOrder():
        # note, we only ever prune children, to avoid unneeded recursive pruning
        #
        # make a copy of the children, as this list will be modified by potential prune calls
        children = node.children[:]
        for child in children:
            if child.properties[prop] < area:
                logging.debug(
                    f"... removing trib with {len(child)} reaches of area: {child.properties[prop]}"
                )
                count += len(child)
                child.prune(preserve_catchments)

    return count


def pruneByArea(rivers, area, prop='DivergenceRoutedDrainAreaSqKm', preserve_catchments=False):
    """Removes, IN PLACE, reaches whose total contributing area is less than area km^2.

    Note this requires NHDPlus data to have been used and the
    'DivergenceRoutedDrainAreaSqKm' property to have been set.
    """
    logging.info(f"Pruning by total contributing area < {area}")
    count = 0
    sufficiently_big_rivers = []
    for river in rivers:
        if river.properties[prop] >= area:
            count += pruneRiverByArea(river, area, prop, preserve_catchments)
            sufficiently_big_rivers.append(river)
    logging.info(f"... pruned {count}")
    return sufficiently_big_rivers


def removeDiversions(rivers, preserve_catchments=False):
    """Removes diversions, but not braids."""
    logging.info("Remove diversions...")
    non_diversions = []
    for river in rivers:
        keep_river = True
        count_tribs = 0
        count_reaches = 0
        for leaf in river.leaf_nodes:
            if leaf.properties['DivergenceCode'] == 2:
                # is a braid or a diversion
                if river.getNode(leaf.properties['UpstreamMainPathHydroSeq']) is None:
                    # diversion!
                    try:
                        joiner = next(n for n in leaf.pathToRoot()
                                      if n.parent is not None and len(n.parent.children) > 1)
                    except StopIteration:
                        # no joiner means kill the whole tree
                        logging.info(f'  ... remove diversion river with {len(river)} reaches.')
                        keep_river = False
                        break
                    else:
                        count_tribs += 1
                        count_reaches += len(joiner)
                        joiner.prune(preserve_catchments)

        if keep_river:
            logging.info(
                f'  ... removed {count_tribs} diversion tributaries with {count_reaches} total reaches.'
            )
            non_diversions.append(river)

    return non_diversions


def removeBraids(rivers, preserve_catchments=False):
    """Remove braids, but not diversions."""
    logging.debug("Removing braided sections...")
    for river in rivers:
        count_tribs = 0
        count_reaches = 0

        for leaf in river.leaf_nodes:
            if leaf.properties['DivergenceCode'] == 2:
                # is a braid or a diversion?
                logging.info(f"  Found a braid with upstream = {leaf.properties['UpstreamMainPathHydroSeq']}")
                upstream_hydroseq = leaf.properties['UpstreamMainPathHydroSeq']
                if river.findNode(lambda n : n.properties['HydrologicSequence'] == upstream_hydroseq) is not None:
                    # braid!
                    try:
                        joiner = next(n for n in leaf.pathToRoot()
                                      if n.parent is not None and len(n.parent.children) > 1)
                    except StopIteration:
                        assert (False)
                        # this should not be possible, because our braid must come back somewhere
                    else:
                        count_tribs += 1
                        count_reaches += len(joiner)
                        joiner.prune()

        logging.debug(
            f'... removed {count_tribs} braids with {count_reaches} reaches from a river of length {len(river)}'
        )
    return rivers


def removeDivergences(rivers, preserve_catchments=False):
    """Removes both diversions and braids.

    Braids are divergences that return to the river network, and so
    look like branches of a river tree whose upstream entity is in the
    river (in another branch).

    Diversions are divergences that do not return to the stream
    network, and so their upstream entity is in another river.

    """
    logging.info("Removing divergent sections...")
    non_divergences = []

    for river in rivers:
        keep_river = True
        count_tribs = 0
        count_reaches = 0
        for leaf in river.leaf_nodes:
            if leaf.properties['DivergenceCode'] == 2:
                # diversion!
                try:
                    joiner = next(n for n in leaf.pathToRoot()
                                  if n.parent is not None and len(n.parent.children) > 1)
                except StopIteration:
                    # no joiner means kill the whole tree
                    logging.info(f'  ... remove divergence river with {len(river)} reaches.')
                    keep_river = False
                    break
                else:
                    count_tribs += 1
                    count_reaches += len(joiner)
                    joiner.prune(preserve_catchments)

        if keep_river:
            logging.info(
                f'  ... removed {count_tribs} divergence tributaries with {count_reaches} total reaches.'
            )
            non_divergences.append(river)

    return non_divergences


def filterSmallRivers(rivers, count):
    """Remove any rivers with fewer than count reaches."""
    logging.info(f"Removing rivers with fewer than {count} reaches.")
    new_rivers = []
    for river in rivers:
        ltree = len(river)
        if ltree < count:
            logging.debug("  ...removing river with %d reaches" % ltree)
        else:
            new_rivers.append(river)
            logging.debug("  ...keeping river with %d reaches" % ltree)
    logging.info(f'... removed {len(rivers) - len(new_rivers)} rivers')
    return new_rivers


def merge(river, tol=_tol):
    """Remove inner branches that are short, combining branchpoints as needed.

    This function merges the "short" segment into the child segment if it is a junction tributary with one child
    or into the parent segment otherwise

    """
    for node in list(river.preOrder()):
        if node.segment.length < tol and node.parent is not None:
            logging.info(
                "  ...cleaned inner segment of length %g at centroid %r with id %r" %
                (node.segment.length, node.segment.centroid.coords[0], node.properties['ID']))

            if len(list(node.siblings())) > 0 and len(node.children) == 1:
                # junction tributary with one child
                node.merge(to='child')
            elif len(node.children) == 0:
                # if the leaf node is too small
                node.remove()
            else:
                for sibling in list(node.siblings()):
                    sibling.moveCoordinate(-1, node.segment.coords[0])
                    sibling.remove()
                    node.addChild(sibling)

                assert (len(list(node.siblings())) == 0)
                node.merge()


def simplify(river, tol=_tol):
    """Simplify, IN PLACE, all reaches."""
    for node in river.preOrder():
        if node.segment is not None:
            new_seg = node.segment.simplify(tol)
            assert (watershed_workflow.utils.isClose(new_seg.coords[0], node.segment.coords[0]))
            assert (watershed_workflow.utils.isClose(new_seg.coords[-1], node.segment.coords[-1]))
            node.segment = new_seg
