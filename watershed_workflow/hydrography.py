"""Functions for manipulating combinations of River and SplitHUCs objects"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple

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
import xarray

import watershed_workflow.config
import watershed_workflow.utils
from watershed_workflow.river_tree import River
from watershed_workflow.split_hucs import SplitHUCs
from watershed_workflow.sources import standard_names as names


#
# findOutlets functions
#
def findOutletsByCrossings(hucs : SplitHUCs,
                           river : River,
                           tol : float = 10,
                           debug_plot : bool = False) -> None:
    """For each HUC, find all outlets using a river network's crossing points."""
    # next determine the outlet, and all boundary edges within x m of that outlet
    polygons = list(hucs.polygons())
    poly_crossings = []
    for i_sub, poly in enumerate(polygons):
        my_crossings = []
        for reach in river.preOrder():
            if poly.exterior.intersects(reach.linestring):
                my_crossings.append(poly.exterior.intersection(reach.linestring))

        # cluster my_crossings to make sure that multiple crossings are only counted once
        mccl = []
        for crossing in my_crossings:
            mccl.append([crossing.centroid.xy[0][0], crossing.centroid.xy[1][0]])
        my_crossing_centroids = np.array(mccl)
        clusters, cluster_centroids = watershed_workflow.utils.cluster(
            my_crossing_centroids, tol)
        poly_crossings.append(cluster_centroids)

    logging.info("Crossings by Polygon:")
    for i, c in enumerate(poly_crossings):
        logging.info(f'  Polygon {i}')
        for p in c:
            logging.info(f'    crossing: {p}')

    # unravel the clusters
    all_crossings_l = [c for p in poly_crossings for c in p]
    all_crossings = np.array(all_crossings_l)

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

    # create a tree, recursively finding all polygons with only
    # one crossing -- this must be an outlet -- then removing it
    # from the list, hopefully leaving a downstream polygon with
    # only one outlet.  This must be done N iterations, where N is
    # the maximal number of polygons crossed from 0th order to
    # maximal order.
    logging.info('Constructing outlet list')
    outlets : Dict[int,int] = dict()
    inlets = collections.defaultdict(list)
    itercount = 0
    done = False
    while not done:
        logging.info(f'Iteration = {itercount}')
        logging.info(f'-----------------')
        new_outlets : Dict[int,int] = dict()

        # look for polygons with only one crossing -- this must be an outlet.
        for pi, clusters2 in poly_cluster_indices.items():
            if len(clusters2) == 1 and pi not in outlets:
                # only one crossing cluster, this is the outlet
                cluster_id = clusters2[0]
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
            fig, ax = plt.subplots(1,1)
            hucs.plot(color='k', ax=ax)
            river.plot(color='b', ax=ax)
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
    hucs.df['outlet'] = outlet_locs.values()


def findOutletsByElevation(hucs : SplitHUCs,
                           elev_raster : xarray.Dataset):
    """Find outlets by the minimum elevation on the boundary."""
    import watershed_workflow
    def _findOutletsByElevation_helper(polygon, crs, elev_raster):
        mesh_points = np.array(polygon.exterior.coords)
        mesh_points_remapped = watershed_workflow.warp.points(mesh_points, crs, elev_raster.crs) 
        elevs = watershed_workflow.utils.valuesFromRaster(mesh_points_remapped, elev_raster)
        i = np.argmin(elevs)
        return shapely.geometry.Point(mesh_points[i])

    hucs.exterior_outlet = _findOutletsByElevation_helper(hucs.exterior, hucs.crs, elev_raster)

    outlets = [_findOutletsByElevation_helper(poly, hucs.crs, elev_raster) for poly in hucs.polygons()]
    hucs.df['outlet'] = outlets


def findOutletsByHydroseq(hucs : SplitHUCs,
                          river : River,
                          tol : float = 0.0):
    """Find outlets using the HydroSequence VAA of NHDPlus.

    Finds the minimum hydroseq reach in each HUC, and intersects that
    with the boundary to find the outlet.
    """
    assert river.isHydroseqConsistent()

    polygons = list(hucs.polygons())
    polygon_outlets = [None for poly in hucs.polygons()]

    # iterate over the reaches, sorted by hydrosequence, looking for
    # the first one that intersects the polygon boundary.
    reaches = sorted(river.preOrder(), key=lambda r: r.properties[names.HYDROSEQ])
    if tol > 0:
        reaches = [r.linestring.buffer(tol) for r in reaches]
    else:
        reaches = [r.linestring for r in reaches]
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

    hucs.df['outlet'] = polygon_outlets


def snapWaterbodies(waterbodies : List[shapely.geometry.base.Geometry],
                    hucs : SplitHUCs,
                    rivers : List[River],
                    tol : float):
    """Snap waterbodies to HUCs and river linestrings.

    Attempts to make waterbodies that intersect or nearly intersect
    hucs intersect discretely, in that they share common point(s).

    """
    # note, this is a fairly fragile algorithm that looks only at
    # discrete points.  A more robust one could be developed if
    # needed.
    mls = shapely.geometry.MultiLineString(
        [r.linestring for river in rivers for r in river]
        + [polygon.exterior for polygon in hucs.polygons()])

    for i, wb in enumerate(waterbodies):
        waterbodies[i] = shapely.snap(wb, mls, tol)


def cutAndSnapCrossings(hucs : SplitHUCs,
                        rivers : List[River]) -> None:
    """If any reach crosses a HUC boundary:

    1. If it crosses an external boundary, cut the reach in two and
       discard the portion outside of the domain.

    2. If it crosses an internal boundary, ensure there is a
       coordinate of the reach that is on the internal boundary.

    Either way, also ensure there is a coordinate on the HUC
    boundary at the crossing point.
    
    """
    # precondition: only trunk or leaf nodes may touch the exterior boundary of hucs
    exterior = hucs.exterior
    for river in rivers:
        for reach in river:
            # if is trunk or leaf
            if reach.parent is None or len(reach.children) == 0:
                assert watershed_workflow.utils.isNonPointIntersection(reach.linestring, exterior)
            else:
                assert exterior.contains(reach.linestring)

    # check exterior crossings on trunk and leaf
    for river in rivers:
        _cutAndSnapExteriorCrossing(hucs, river)
        for leaf in river.leaf_nodes:
            _cutAndSnapExteriorCrossing(hucs, leaf)
        
    # check interior crossings on all
    for river in rivers:
        for reach in river:
            _cutAndSnapInteriorCrossing(hucs, reach)


def _cutAndSnapExteriorCrossing(hucs : SplitHUCs,
                                reach : River,
                                merge_tol : float) -> None:
    """Helper function for cutAndSnapCrossings()"""
    r = reach.linestring

    # first deal with crossings of the HUC exterior boundary -- in
    # this case, the reach linestring gets split in two and the external
    # one is removed.
    for b, spine in hucs.boundaries.items():
        for s, ls_handle in spine.items():
            ls = hucs.linestrings[ls_handle]

            if watershed_workflow.utils.intersects(ls, r):
                logging.info('intersection found')
                new_spine, new_reach = watershed_workflow.utils.cut(ls, r)
                assert len(new_reach) == 1 or len(new_reach) == 2
                assert len(new_spine) == 1 or len(new_spine) == 2

                logging.info("  - cutting reach at external boundary of HUCs:")
                logging.info(f"      split HUC boundary ls into {len(new_spine)} pieces")
                logging.info(f"      split reach ls into {len(new_reach)} pieces")

                # which piece of the reach are we keeping?
                if hucs.exterior.buffer(-1).contains(shapely.geometry.Point(new_reach[0].coords[0])):
                    # keep the upstream (or only) reach ls
                    if len(new_reach) == 2:
                    # confirm other/downstream reach is outside
                    assert not hucs.exterior.contains(shapely.geometry.Point(new_reach[1].coords[-1]))
                    reach.linestring = new_reach[0]

                elif len(new_reach) == 2:
                    if hucs.exterior.buffer(-1).contains(shapely.geometry.Point(new_reach[1].coords[-1])):
                        # keep the downstream reach ls, confirm upstream is outside
                        assert not hucs.exterior.contains(shapely.geometry.Point(new_reach[0].coords[0]))
                        reach.linestring = new_reach[1]

                # keep both pieces of a split huc boundary linestring
                # -- rename the first
                hucs.linestrings[ls_handle] = new_spine[0]
                if len(new_spine) > 1:
                    # -- add the first
                    new_handle = hucs.linestrings.append(new_spine[1])
                    spine.append(new_handle)


def _cutAndSnapInteriorCrossing(hucs : SplitHUCs,
                                reach : River,
                                merge_tol : float) -> None:
    """Helper function for cutAndSnapCrossings()"""
    r = reach.linestring

    # now deal with crossings of the HUC interior boundary -- in this
    # case, the reach linestring cut, then potentially merged to neighbors
    for i, spine in hucs.intersections.items():
        for s, ls_handle in spine.items():
            ls = hucs.linestrings[ls_handle]
                    
            if watershed_workflow.utils.intersects(ls, r):
                new_spine, new_reach = watershed_workflow.utils.cut(ls, r)
                assert len(new_reach) == 1 or len(new_reach) == 2
                assert len(new_spine) == 1 or len(new_spine) == 2
                logging.info("  - snapping reach at internal boundary of HUCs")
                if len(new_reach) == 1:
                    reach.linestring = new_reach[0]
                elif len(new_reach) == 2:
                    reach.linestring = shapely.geometry.LineString(
                        list(new_reach[0].coords) + list(new_reach[1].coords)[1:])
                    us, ds = reach.split(len(new_reach[0].coords)-1)
                    # potentially merge
                    if us.linestring.length < 
                else:

                hucs.linestrings[ls_handle] = new_spine[0]
                if len(new_spine) > 1:
                    assert (len(new_spine) == 2)
                    new_handle = hucs.linestrings.append(new_spine[1])
                    spine.append(new_handle)
                break


def snapHUCsJunctions(hucs : SplitHUCs,
                      rivers : List[River],
                      tol : float):
    """Snaps the junctions of HUC linestrings to endpoints of rivers."""
    # make the kdTree of endpoints of all reaches
    coords1 = np.array([reach.linestring.coords[-1] for river in rivers for reach in river.preOrder()])
    coords2 = np.array([reach.linestring.coords[0] for river in rivers for reach in river.leaf_nodes])
    coords = np.concatenate([coords1, coords2], axis=0)

    # limit to x,y
    if (coords.shape[1] != 2):
        coords = coords[:, 0:2]

    kdtree = cKDTree(coords)
    # for each linestring of the HUC spine, find the river outlet that is
    # closest.  If within tolerance, move it
    for ls_handle, ls in hucs.linestrings.items():
        # check point 0, -1
        endpoints = np.array([ls.coords[0], ls.coords[-1]])
        # limit to x,y
        if (endpoints.shape[1] != 2):
            endpoints = endpoints[:, 0:2]
        dists, inds = kdtree.query(endpoints)

        if dists.min() < tol:
            new_ls = list(ls.coords)
            if dists[0] < tol:
                new_ls[0] = coords[inds[0]]
                logging.debug(
                    f"  Moving HUC linestring point 0,1: {list(ls.coords)[0]}, {list(ls.coords)[-1]}"
                )
                logging.debug("        point 0 to river at %r" % list(new_ls[0]))

            if dists[1] < tol:
                new_ls[-1] = coords[inds[1]]
                logging.debug(
                    f"  Moving HUC linestring point 0,1: {list(ls.coords)[0]}, {list(ls.coords)[-1]}"
                )
                logging.debug("        point -1 to river at %r" % list(new_ls[-1]))
            hucs.linestrings[ls_handle] = shapely.geometry.LineString(new_ls)


def snapEndpoints(river : River,
                  hucs : SplitHUCs,
                  tol : float) -> None:
    """Snap river endpoints to huc linestrings and insert that point into
    the boundary.

    Note this is O(n^2), and could be made more efficient.
    """
    to_add = []
    for node in river.preOrder():
        reach = node.linestring
        for b, component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):

            # note, this is done in two stages to allow it deal with both endpoints touching
            for s, ls_handle in component.items():
                ls = hucs.linestrings[ls_handle]
                #logging.debug("SNAP P0:")
                #logging.debug("  huc ls: %r"%ls.coords[:])
                #logging.debug("  reach: %r"%reach.coords[:])
                altered = False
                logging.debug("  - checking reach coord: %r" % list(reach.coords[0]))
                logging.debug("  - ls coords: {0}".format(list(ls.coords)))
                new_coord = watershed_workflow.utils.findNearestPoint(reach.coords[0], ls, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord is not None:
                    logging.debug(f"    snapped reach: {reach.coords[0]} to {new_coord}")

                    # move new_coord onto an existing linestring coord
                    new_coord_c = np.expand_dims(new_coord, 0) # type: ignore
                    dist = np.linalg.norm(np.array(ls.coords) - new_coord_c, axis=1)
                    assert (len(dist) == len(ls.coords))
                    assert (len(dist.shape) == 1)
                    i = int(np.argmin(dist))
                    if (dist[i] < tol):
                        new_coord = ls.coords[i]

                    # remove points that are closer
                    coords = list(reach.coords)
                    done = False
                    while len(coords) > 2 and \
                          watershed_workflow.utils.computeDistance(new_coord, coords[1]) \
                          < watershed_workflow.utils.computeDistance(new_coord, coords[0]):
                        coords.pop(0)
                    coords[0] = new_coord
                    reach = shapely.geometry.LineString(coords)
                    node.linestring = reach
                    to_add.append((ls_handle, component, 0, node))
                    break

            # second stage
            for s, ls_handle in component.items():
                ls = hucs.linestrings[ls_handle]
                # logging.debug("SNAP P1:")
                # logging.debug("  huc ls: %r"%ls.coords[:])
                # logging.debug("  reach: %r"%reach.coords[:])
                altered = False
                logging.debug("  - checking reach coord: %r" % list(reach.coords[-1]))
                logging.debug("  - ls coords: {0}".format(list(ls.coords)))
                new_coord = watershed_workflow.utils.findNearestPoint(reach.coords[-1], ls, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord is not None:
                    logging.debug("  - snapped reach: %r to %r" % (reach.coords[-1], new_coord))

                    # move new_coord onto an existing linestring coord
                    new_coord_c = np.expand_dims(new_coord, 0) # type: ignore
                    dist = np.linalg.norm(np.array(ls.coords) - new_coord_c, axis=1)
                    assert (len(dist) == len(ls.coords))
                    assert (len(dist.shape) == 1)
                    i = int(np.argmin(dist))
                    if (dist[i] < tol):
                        new_coord = ls.coords[i]

                    # remove points that are closer
                    coords = list(reach.coords)
                    done = False
                    while len(coords) > 2 and \
                          watershed_workflow.utils.computeDistance(new_coord, coords[-2]) \
                          < watershed_workflow.utils.computeDistance(new_coord, coords[-1]):
                        coords.pop(-1)
                    coords[-1] = new_coord
                    reach = shapely.geometry.LineString(coords)
                    node.linestring = reach
                    to_add.append((ls_handle, component, -1, node))
                    break

    # find the list of points to add to a given linestring
    to_add_dict : Dict[int, List[Any]] = dict()
    for ls_handle, component, endpoint, node in to_add:
        if ls_handle not in to_add_dict.keys():
            to_add_dict[ls_handle] = list()
        to_add_dict[ls_handle].append((component, endpoint, node))

    # find the set of points to add to each given linestring
    def isEqual(p1, p2):
        if watershed_workflow.utils.isClose(p1[2].linestring.coords[p1[1]],
                                            p2[2].linestring.coords[p2[1]],
                                            1.e-5):
            assert (p1[0] == p2[0])
            return True
        else:
            return False

    to_add_dict2 = dict()
    for ls_handle, insert_list in to_add_dict.items():
        new_list : List[Any] = []
        for p1 in insert_list:
            if (all(not isEqual(p1, p2) for p2 in new_list)):
                new_list.append(p1)
        to_add_dict2[ls_handle] = new_list

    # add these points to the linestring
    for ls_handle, insert_list in to_add_dict2.items():
        ls = hucs.linestrings[ls_handle]
        # make a list of the coords and a flag to indicate a new
        # coord, then sort it by arclength along the linestring.
        #
        # Note this needs special care if the ls is a loop, or else the endpoint gets sorted twice
        if not watershed_workflow.utils.isClose(ls.coords[0], ls.coords[-1]):
            new_coords = [[p[2].linestring.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in ls.coords
                if not any(watershed_workflow.utils.isClose(c, nc, tol) for nc in new_coords)
            ]
            new_ls_coords = sorted(new_coords + old_coords,
                                    key=lambda a: ls.project(shapely.geometry.Point(a[0])))

            # determine the new coordinate indices
            breakpoint_inds = [i for i, (c, f) in enumerate(new_ls_coords) if f == 1]

        else:
            new_coords = [[p[2].linestring.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in ls.coords[:-1]
                if not any(watershed_workflow.utils.isClose(c, nc, tol) for nc in new_coords)
            ]
            new_ls_coords = sorted(new_coords + old_coords,
                                    key=lambda a: ls.project(shapely.geometry.Point(a[0])))
            breakpoint_inds = [i for i, (c, f) in enumerate(new_ls_coords) if f == 1]
            assert (len(breakpoint_inds) > 0)
            new_ls_coords = new_ls_coords[breakpoint_inds[0]:] + new_ls_coords[
                0:breakpoint_inds[0] + 1]
            new_ls_coords[0][1] = 0
            new_ls_coords[-1][1] = 0
            breakpoint_inds = [i for i, (c, f) in enumerate(new_ls_coords) if f == 1]

        # now break into new linestrings
        new_lss = []
        ind_start = 0
        for ind_end in breakpoint_inds:
            assert (ind_end != 0)
            new_lss.append(
                shapely.geometry.LineString([c
                                             for (c, f) in new_ls_coords[ind_start:ind_end + 1]]))
            ind_start = ind_end

        assert (ind_start < len(new_ls_coords) - 1)
        new_lss.append(
            shapely.geometry.LineString([tuple(c) for (c, f) in new_ls_coords[ind_start:]]))

        # put all new_lss into the huc list.  Note insert_list[0][0] is the component
        hucs.linestrings[ls_handle] = new_lss.pop(0)
        new_handles = hucs.linestrings.extend(new_lss)
        insert_list[0][0].extend(new_handles)









