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
def findOutletsByCrossings(hucs: SplitHUCs,
                           river: River,
                           tol: float = 10,
                           debug_plot: bool = False) -> None:
    """For each HUC, find all outlets using a river network's crossing points.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object to find outlets for.
    river : River
        River network to use for finding crossing points.
    tol : float, optional
        Tolerance in map units for clustering crossings, by default 10.
    debug_plot : bool, optional
        Whether to create debug plots showing outlets, by default False.
    """
    # next determine the outlet, and all boundary edges within x m of that outlet

    # note, we avoid recomputing the polygon geometry because it has a
    # notch removed at the outlet, which will cause an error.
    polygons = list(hucs.df['geometry'])
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
        clusters, cluster_centroids = watershed_workflow.utils.cluster(my_crossing_centroids, tol)
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
    outlets: Dict[int, int] = dict()
    inlets = collections.defaultdict(list)
    itercount = 0
    done = False
    last_outlet = None

    # First of all, check whether inlet dict (cluster_poly_indices) has a value
    # with length of 1, if so, assign that cluster as the outlet of that poly and 
    # meanwhile delete the corresponding {c: [polys]} from the inlet dict.
    for ci, polys in cluster_poly_indices.items():
        if len(polys) == 1:
            outlets[polys[0]] = ci
    cluster2rm = list(outlets.values())
    cluster_poly_indices = {c: p for c, p in cluster_poly_indices.items() if c not in cluster2rm}

    while not done:
        logging.info(f'Iteration = {itercount}')
        logging.info(f'-----------------')
        new_outlets: Dict[int, int] = dict()

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
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1)
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
            ax.set_title(f'Outlets after iteration {itercount}')
            plt.show()

        outlets.update(new_outlets)
        itercount += 1
        done = itercount > 50 or len(outlets) == len(polygons) or len(new_outlets) == 0

    if last_outlet is not None:
        logging.info(f'last outlet is {last_outlet} in polygon {last_outlet_poly} '
                     'at {crossings_clusters_centroids[last_outlet]}')
    else:
        logging.info(f'did not find a domain outlet')

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
    hucs.df[names.OUTLET] = outlet_locs.values()


def findOutletsByElevation(hucs: SplitHUCs, elev_raster: xarray.Dataset) -> None:
    """Find outlets by the minimum elevation on the boundary.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object to find outlets for.
    elev_raster : xarray.Dataset
        Elevation raster dataset for determining minimum elevations.
    """
    import watershed_workflow

    def _findOutletsByElevation_helper(polygon, crs, elev_raster):
        mesh_points = np.array(polygon.exterior.coords)
        mesh_points_remapped = watershed_workflow.warp.points(mesh_points, crs, elev_raster.crs)
        elevs = watershed_workflow.utils.valuesFromRaster(mesh_points_remapped, elev_raster)
        i = np.argmin(elevs)
        return shapely.geometry.Point(mesh_points[i])

    hucs.exterior_outlet = _findOutletsByElevation_helper(hucs.exterior, hucs.crs, elev_raster)

    outlets = [
        _findOutletsByElevation_helper(poly, hucs.crs, elev_raster) for poly in hucs.polygons()
    ]
    hucs.df[names.OUTLET] = outlets


def findOutletsByHydroseq(hucs: SplitHUCs, river: River, tol: float = 0.0) -> None:
    """Find outlets using the HydroSequence VAA of NHDPlus.

    Finds the minimum hydroseq reach in each HUC, and intersects that
    with the boundary to find the outlet.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object to find outlets for.
    river : River
        River network with HydroSequence properties.
    tol : float, optional
        Tolerance for buffering reaches, by default 0.0.
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

    hucs.df[names.OUTLET] = polygon_outlets


def snapWaterbodies(waterbodies: List[shapely.geometry.base.BaseGeometry], hucs: SplitHUCs,
                    rivers: List[River], tol: float) -> None:
    """Snap waterbodies to HUCs and river linestrings.

    Attempts to make waterbodies that intersect or nearly intersect
    hucs intersect discretely, in that they share common point(s).

    Parameters
    ----------
    waterbodies : List[shapely.geometry.base.BaseGeometry]
        List of waterbody geometries to snap.
    hucs : SplitHUCs
        Split HUCs object containing boundary linestrings.
    rivers : List[River]
        List of river networks to snap to.
    tol : float
        Snapping tolerance in map units.
    """
    # note, this is a fairly fragile algorithm that looks only at
    # discrete points.  A more robust one could be developed if
    # needed.
    mls = shapely.geometry.MultiLineString([r.linestring for river in rivers for r in river]
                                           + [polygon.exterior for polygon in hucs.polygons()])

    for i, wb in enumerate(waterbodies):
        waterbodies[i] = shapely.snap(wb, mls, tol)


def cutAndSnapCrossings(hucs: SplitHUCs, rivers: List[River], tol: float) -> None:
    """Aligns river and HUC objects.

    1. where a reach crosses an external boundary, cut in two and keep
       only internal portion.

    2. where a reach crosses an internal boundary, either:
       - snap the internal boundary to the reach endpoint
       - cut the reach in two

    At the end of the day, we ensure that any place where a river
    crosses a HUC boundary is a discrete point at a both a reach
    endpoint and a HUC boundary segment.  This ensures that those
    points will never move and will always be coincident.

    """
    # check exterior crossings on trunk and leaf
    for river in rivers:
        _cutAndSnapExteriorCrossing(hucs, river, tol)
        for leaf in river.leaf_nodes:
            _cutAndSnapExteriorCrossing(hucs, leaf, tol)

    for river in rivers:
        assert river.isContinuous()
            
    # check interior crossings on all
    for river in rivers:
        # DO NOT make a copy -- we want the iterator in this case to be modified in place!
        for reach in river:
            try:
                assert reach.isLocallyContinuous(), 'prior'
                _cutAndSnapInteriorCrossing(hucs, reach, tol)
                assert reach.isLocallyContinuous(), 'after'
            except AssertionError as err:
                fig, ax = plt.subplots(1,1)
                import watershed_workflow.plot
                ax = watershed_workflow.plot.linestringWithCoords(reach.linestring, marker='x', color='r')
                for child in reach.children:
                    watershed_workflow.plot.linestringWithCoords(child.linestring, marker='x', color='m', ax=ax)
                watershed_workflow.plot.linestringWithCoords(reach.parent.linestring, marker='x', color='b', ax=ax)
                watershed_workflow.plot.linestringWithCoords(reach.parent.parent.linestring, marker='x', color='c', ax=ax)
                hucs.plot(ax=ax, color='k')
                plt.show()
                raise RuntimeError(f'Failed locally continuous on reach {reach[names.ID]} in cutAndSnapCrossings due to check {err}')

                

def _cutAndSnapExteriorCrossing(hucs: SplitHUCs, reach: River, merge_tol: float) -> None:
    """Cut and snap reach at exterior HUC boundary crossings.

    If any reach crosses a HUC boundary:

    1. If it crosses an external boundary, cut the reach in two and
       discard the portion outside of the domain.

    2. If it crosses an internal boundary, ensure there is a
       coordinate of the reach that is on the internal boundary.

    Either way, also ensure there is a coordinate on the HUC
    boundary at the crossing point.

    Modifies no geometry, only topology.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object containing boundary information.
    reach : River
        River reach to process for boundary crossings.
    merge_tol : float
        Tolerance for merging operations.
    """
    r = reach.linestring

    # first deal with crossings of the HUC exterior boundary -- in
    # this case, the reach linestring gets split in two and the external
    # one is removed.
    for b, spine in hucs.boundaries.items():
        # make a copy as spine is modified in-place
        spine_list = list(spine.items())
        for s, ls_handle in spine_list:
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
                if hucs.exterior.buffer(-1).contains(shapely.geometry.Point(
                        new_reach[0].coords[0])):
                    # keep the upstream (or only) reach ls
                    if len(new_reach) == 2:
                        # confirm other/downstream reach is outside
                        assert not hucs.exterior.contains(
                            shapely.geometry.Point(new_reach[1].coords[-1]))
                    reach.linestring = new_reach[0]

                elif len(new_reach) == 2:
                    if hucs.exterior.buffer(-1).contains(
                            shapely.geometry.Point(new_reach[1].coords[-1])):
                        # keep the downstream reach ls, confirm upstream is outside
                        assert not hucs.exterior.contains(
                            shapely.geometry.Point(new_reach[0].coords[0]))
                        reach.linestring = new_reach[1]

                # keep both pieces of a split huc boundary linestring
                # -- rename the first
                hucs.linestrings[ls_handle] = new_spine[0]
                if len(new_spine) > 1:
                    # -- add the second
                    new_handle = hucs.linestrings.append(new_spine[1])
                    spine.append(new_handle)


def _cutAndSnapInteriorCrossing(hucs: SplitHUCs, reach: River, merge_tol: float) -> None:
    """Cut and snap reach at interior HUC boundary crossings.

    Helper function for cutAndSnapCrossings().
    Modifies no geometry, only topology.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object containing interior boundary information.
    reach : River
        River reach to process for interior crossings.
    merge_tol : float
        Tolerance for merging operations.

    Returns
    -------
    return_code : int
        0 if no snap
        1 if endpoint was snapped
        2 if interior was snapped
    """
    r = reach.linestring
    return_code = 0

    # now deal with crossings of the HUC interior boundary -- in this
    # case, the reach linestring cut, then potentially merged to neighbors
    for i, spine in hucs.intersections.items():
        # make a copy as spine is modified in-place
        spine_list = list(spine.items())
        for s, ls_handle in spine_list:
            ls = hucs.linestrings[ls_handle]

            if watershed_workflow.utils.intersects(ls, r):
                new_spine, new_reach = watershed_workflow.utils.cut(ls, r)
                assert len(new_reach) == 1 or len(new_reach) == 2
                assert len(new_spine) == 1 or len(new_spine) == 2
                logging.info("  - snapping reach at internal boundary of HUCs")
                if len(new_reach) == 1:
                    reach.linestring = new_reach[0]
                    logging.info(f'  branch1 on reach {reach[names.ID]}')
                    return_code = max(return_code, 1)
                elif len(new_reach) == 2:
                    logging.info(f'  branch2 on reach {reach[names.ID]}')
                    reach.linestring = shapely.geometry.LineString(
                        list(new_reach[0].coords) + list(new_reach[1].coords)[1:])
                    logging.info(f'  old_r: {list(r.coords)}')
                    logging.info(f'  seg1: {list(new_reach[0].coords)}')
                    logging.info(f'  seg2: {list(new_reach[1].coords)}')
                    logging.info(
                        f'  splitting at coord: {len(new_reach[0].coords)-1} of {len(reach.linestring.coords)}'
                    )
                    us, ds = reach.split(len(new_reach[0].coords) - 1)
                    split = True
                    logging.info(
                        f'  into reach (upstream) {us[names.ID]} and reach (downstream) {ds[names.ID]}')
                    assert us.isLocallyContinuous()
                    assert ds.isLocallyContinuous()
                    return_code = max(return_code, 2)

                    # continue with the downstream segment -- upstream
                    # will be continued in the next iteration
                    reach = ds
                    r = reach.linestring

                hucs.linestrings[ls_handle] = new_spine[0]
                if len(new_spine) > 1:
                    assert (len(new_spine) == 2)
                    new_handle = hucs.linestrings.append(new_spine[1])
                    spine.append(new_handle)

    return return_code

def snapHUCsJunctions(hucs: SplitHUCs, rivers: List[River], tol: float) -> None:
    """Snaps the junctions of HUC linestrings to endpoints of rivers.

    Modifies HUCs geometry.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object to modify.
    rivers : List[River]
        List of river networks to snap to.
    tol : float
        Snapping tolerance in map units.
    """
    # make the kdTree of endpoints of all reaches
    coords1 = np.array(
        [reach.linestring.coords[-1] for river in rivers for reach in river.preOrder()])
    coords2 = np.array(
        [reach.linestring.coords[0] for river in rivers for reach in river.leaf_nodes])
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
        # delete z coord
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

                # zap points until we are outside of the ball -- otherwise we can get weird crossings
                count = 0
                while watershed_workflow.utils.computeDistance(new_ls[1], new_ls[0]) < tol:
                    new_ls.pop(1)
                    count += 1
                logging.debug(f"    also removed {count} other coords")

            if dists[1] < tol:
                new_ls[-1] = coords[inds[1]]
                logging.debug(
                    f"  Moving HUC linestring point 0,1: {list(ls.coords)[0]}, {list(ls.coords)[-1]}"
                )
                logging.debug("        point -1 to river at %r" % list(new_ls[-1]))

                # zap points until we are outside of the ball -- otherwise we can get weird crossings
                count = 0
                while watershed_workflow.utils.computeDistance(new_ls[-2], new_ls[-1]) < tol:
                    new_ls.pop(-2)
                    count += 1
                logging.debug(f"    also removed {count} other coords")

            hucs.linestrings[ls_handle] = shapely.geometry.LineString(new_ls)


def _findContainingPolygon(hucs: SplitHUCs, linestring: shapely.geometry.LineString) -> int:
    """Find the polygon that contains the largest portion of linestring.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object containing polygons.
    linestring : shapely.geometry.LineString
        Linestring to find containing polygon for.

    Returns
    -------
    int
        Index of the polygon containing the largest portion of the linestring.
    """
    return max((i for i in range(len(hucs))),
               key=lambda i: linestring.intersection(hucs.computePolygon(i)).length)


def snapReachEndpoints(hucs: SplitHUCs, river: River, tol: float) -> None:
    """Snap river endpoints to HUC linestrings and insert that point into the boundary.

    Note this is O(n^2), and could be made more efficient.
    Modifies reach geometry.

    Parameters
    ----------
    hucs : SplitHUCs
        Split HUCs object containing boundary linestrings.
    river : River
        River network to snap endpoints for.
    tol : float
        Snapping tolerance in map units.
    """
    to_add = []
    reaches = list(river)

    # check the river oulet
    reach_ls = river.linestring
    done = False
    for b, component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):
        for s, huc_ls_handle in component.items():
            huc_ls = hucs.linestrings[huc_ls_handle]
            logging.debug("  - checking reach coord: %r" % list(reach_ls.coords[-1]))
            logging.debug("  - huc_ls coords: {0}".format(list(huc_ls.coords)))

            # find the nearest point to the endpoint if it is within tol
            new_coord = watershed_workflow.utils.findNearestPoint(reach_ls.coords[-1], huc_ls, tol)
            if new_coord is not None:
                if any(watershed_workflow.utils.isClose(new_coord, c) for c in huc_ls.coords):
                    # the endpoint is already discretely in the HUC
                    # boundary, likely done in a junction snap
                    done = True
                    break

                logging.debug("  - new coord: {0}".format(new_coord))
                logging.debug("  - snapped reach: %r to %r" % (reach_ls.coords[-1], new_coord))

                # keep a list of all points to add, which are all added at once
                to_add.append((huc_ls_handle, component, -1, river))

                # remove points on the reach that are
                # closer to the huc -- this deals with the
                # case that multiple discrete points are
                # on the "wrong" side of the internal
                # boundary.
                coords = list(reach_ls.coords)
                while len(coords) > 2 and \
                      watershed_workflow.utils.computeDistance(new_coord, coords[-2]) \
                      < watershed_workflow.utils.computeDistance(new_coord, coords[-1]):
                    coords.pop(-1)
                coords[-1] = new_coord
                reach_ls = shapely.geometry.LineString(coords)
                river.linestring = reach_ls

                # if we add a point on this huc linestring, don't
                # add it to any other
                done = True
                break

        if done: break  # break out of both for loops

    # now check the upstream end of all reaches
    for reach in reaches:
        reach_ls = reach.linestring
        done = False

        # only consider endpoints for whom the touching reaches span multiple polygons
        reach_linestrings = [reach_ls, ] + [
            watershed_workflow.utils.reverseLineString(c.linestring) for c in reach.children
        ]
        touching_polygons = set(_findContainingPolygon(hucs, r) for r in reach_linestrings)

        if len(touching_polygons) > 1:
            # find the component it touches
            for b, component in itertools.chain(hucs.boundaries.items(),
                                                hucs.intersections.items()):
                for s, huc_ls_handle in component.items():
                    huc_ls = hucs.linestrings[huc_ls_handle]
                    logging.debug("  - checking reach coord: %r" % list(reach_ls.coords[0]))
                    logging.debug("  - huc_ls coords: {0}".format(list(huc_ls.coords)))

                    # only consider endpoints for whom one reach
                    # intersects the HUC boundary.  If the endpoint is
                    # close but no reaches intersect the boundary, it
                    # is likely that the endpoint is fully contained
                    # in the polygon.  We shrink the linestring at the
                    # opposite endpoint to avoid counting an
                    # intersection at the other end of the reach.
                    def _shrinkLS(ls):
                        coords = [c for c in ls.coords]
                        coords[-1] = watershed_workflow.utils.computeMidpoint(
                            coords[-2], coords[-1])
                        return shapely.geometry.LineString(coords)

                    #
                    # this is the upstream point of this reach, so
                    # consider this and all children
                    if any(
                            watershed_workflow.utils.intersects(huc_ls, _shrinkLS(ls))
                            for ls in reach_linestrings):
                        # find the nearest point to the endpoint if it is within tol
                        new_coord = watershed_workflow.utils.findNearestPoint(
                            reach_ls.coords[0], huc_ls, tol)
                        if new_coord is not None:
                            if any(
                                    watershed_workflow.utils.isClose(new_coord, c)
                                    for c in huc_ls.coords):
                                # the endpoint is already discretely in the
                                # HUC boundary, likely done as a junction
                                done = True
                                break

                            logging.debug("  - new coord: {0}".format(new_coord))
                            logging.debug(f"    snapped reach: {reach_ls.coords[0]} to {new_coord}")

                            # keep a list of all points to add, which are all added at once
                            to_add.append((huc_ls_handle, component, 0, reach))

                            for ls, r in zip(reach_linestrings, [reach, ] + list(reach.children)):
                                # remove points on the reach that are
                                # closer to the huc -- this deals with the
                                # case that multiple discrete points are
                                # on the "wrong" side of the internal
                                # boundary.
                                coords = list(ls.coords)
                                while len(coords) > 2 and \
                                      watershed_workflow.utils.computeDistance(new_coord, coords[1]) \
                                      < watershed_workflow.utils.computeDistance(new_coord, coords[0]):
                                    coords.pop(0)
                                coords[0] = new_coord
                                ls = shapely.geometry.LineString(coords)
                                if r is reach:
                                    r.linestring = ls
                                else:
                                    r.linestring = watershed_workflow.utils.reverseLineString(ls)

                            assert reach.isLocallyContinuous()

                            # if we add a point on this huc linestring, don't
                            # add it to any other
                            done = True
                            break

                if done: break  # break out of two for loops

        if reach.linestring.length == 0.:
            # snapped both endpoints to the same point on the internal
            # boundary, remove the reach note we can safely merge this
            # with either parent or child
            if len(list(reach.siblings)) == 0:
                reach.merge()
            else:
                assert len(reach.children) == 1
                reach.children[0].merge()

    # find the list of points to add to a given linestring
    to_add_dict: Dict[int, List[Any]] = dict()
    for huc_ls_handle, component, endpoint, reach in to_add:
        if huc_ls_handle not in to_add_dict.keys():
            to_add_dict[huc_ls_handle] = list()
        to_add_dict[huc_ls_handle].append((component, endpoint, reach))

    # find the set of points to add to each given linestring
    def isEqual(p1, p2):
        if watershed_workflow.utils.isClose(p1[2].linestring.coords[p1[1]],
                                            p2[2].linestring.coords[p2[1]], 1.e-5):
            assert (p1[0] == p2[0])
            return True
        else:
            return False

    to_add_dict2 = dict()
    for huc_ls_handle, insert_list in to_add_dict.items():
        new_list: List[Any] = []
        for p1 in insert_list:
            if (all(not isEqual(p1, p2) for p2 in new_list)):
                new_list.append(p1)
        to_add_dict2[huc_ls_handle] = new_list

    # add these points to the linestring
    for huc_ls_handle, insert_list in to_add_dict2.items():
        ls = hucs.linestrings[huc_ls_handle]
        # make a list of the coords and a flag to indicate a new
        # coord, then sort it by arclength along the linestring.
        #
        # Note this needs special care if the ls is a loop, or else the endpoint gets sorted twice
        if not watershed_workflow.utils.isClose(ls.coords[0], ls.coords[-1]):
            new_coords = [[p[2].linestring.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in ls.coords
                if not any(watershed_workflow.utils.isClose(c, nc[0], tol) for nc in new_coords)
            ]
            new_ls_coords = sorted(new_coords + old_coords,
                                   key=lambda a: ls.project(shapely.geometry.Point(a[0])))

            # determine the new coordinate indices
            breakpoint_inds = [i for i, (c, f) in enumerate(new_ls_coords) if f == 1]

        else:
            new_coords = [[p[2].linestring.coords[p[1]], 1] for p in insert_list]
            old_coords = [
                [c, 0] for c in ls.coords[:-1]
                if not any(watershed_workflow.utils.isClose(c, nc[0], tol) for nc in new_coords)
            ]
            new_ls_coords = sorted(new_coords + old_coords,
                                   key=lambda a: ls.project(shapely.geometry.Point(a[0])))
            breakpoint_inds = [i for i, (c, f) in enumerate(new_ls_coords) if f == 1]
            assert (len(breakpoint_inds) > 0)
            new_ls_coords = new_ls_coords[breakpoint_inds[0]:] + new_ls_coords[0:breakpoint_inds[0]
                                                                               + 1]
            new_ls_coords[0][1] = 0
            new_ls_coords[-1][1] = 0
            breakpoint_inds = [i for i, (c, f) in enumerate(new_ls_coords) if f == 1]

        # now break into new linestrings
        new_lss = []
        ind_start = 0
        for ind_end in breakpoint_inds:
            assert (ind_end != 0)
            new_lss.append(
                shapely.geometry.LineString([c for (c, f) in new_ls_coords[ind_start:ind_end + 1]]))
            ind_start = ind_end

        assert (ind_start < len(new_ls_coords) - 1)
        new_lss.append(
            shapely.geometry.LineString([tuple(c) for (c, f) in new_ls_coords[ind_start:]]))

        # put all new_lss into the huc list.  Note insert_list[0][0] is the component
        watershed_workflow.utils.logMinMaxMedianSegment(new_lss, "new_lss")

        hucs.linestrings[huc_ls_handle] = new_lss.pop(0)
        new_handles = hucs.linestrings.extend(new_lss)
        insert_list[0][0].extend(new_handles)
