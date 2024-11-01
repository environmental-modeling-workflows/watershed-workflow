"""These high-level functions attempt to "do what the user means."
They group other, lower-level functions into common sets of
operations.  The "default workflow" is nearly entirely built from
calls to this namespace.  If the user needs to deviate from the
default workflow, they can then call the lower level functions.

This top level module provides functionality for getting shapes and
rasters representing watershed boundaries, river networks, digital
elevation models, and other GIS datasets and then processing those
data sets for use in simulations.

Functions here that get shapes and rasters should be used instead of
directly using the source managers, because they additionally do
things like convert coordinate systems, get the data into common
data structures, etc.

"""

from . import _version

__version__ = _version.get_versions()['version']

import os
from watershed_workflow.config import rcParams

if rcParams['DEFAULT']['proj_network'] == "True":
    os.environ['PROJ_NETWORK'] = 'ON'
elif rcParams['DEFAULT']['proj_network'] == "False":
    os.environ['PROJ_NETWORK'] = 'OFF'

import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import scipy

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.warp
import watershed_workflow.river_tree
import watershed_workflow.split_hucs
import watershed_workflow.hydrography
import watershed_workflow.triangulation
import watershed_workflow.densification
import watershed_workflow.river_mesh

#import watershed_workflow.sources.utils
#import watershed_workflow.sources.manager_shape


#
# functions for relating objects
# -----------------------------------------------------------------------------
def findHUC(source, shape, in_crs, hint, shrink_factor=1.e-5):
    """Finds the smallest HUC containing shp.

    Parameters
    ----------
    source : source-type
        Source object providing a get_hucs() method.
    shape : Polygon
        Shapely or fiona polygon on which to get the raster.
    in_crs : crs-type
        CRS of shape.
    hint : str
        HUC in which to start searching.  This should be at least as long as
        the indexing file size -- HUC 2 or longer for WBD, 4 or longer for NHD
        Plus, or 8 or longer for NHD.
    shrink_factor : float, optional
        A fraction of the radius of shape to shrink prior for checking
        containment within HUCs.  This fixes cases where shape is on a HUC
        boundary with potentially some numerical error.

    Returns
    ------- 
    out : str
        The smallest containing HUC.

    """
    def _in_huc(shply, huc_shply):
        """Checks whether shp is in HUC"""
        if huc_shply.contains(shply):
            return 2
        elif huc_shply.intersects(shply):
            return 1
        else:
            return 0

    def _findHUC(source, shply, crs, hint):
        """Searches in hint to find shp."""
        logging.debug('searching: %s' % hint)
        hint_level = len(hint)
        search_level = hint_level + 2
        if search_level > source.lowest_level:
            return hint

        _, subhus = get_hucs(source, hint, search_level, crs)

        for subhu in subhus:
            inhuc = _in_huc(shply, subhu)

            if inhuc == 2:
                # fully contained in try_huc, recurse
                hname = watershed_workflow.sources.utils.get_code(subhu, search_level)
                logging.debug('  subhuc: %s contains' % hname)
                return _findHUC(source, shply, crs, hname)
            elif inhuc == 1:
                hname = watershed_workflow.sources.utils.get_code(subhu, search_level)
                logging.debug('  subhuc: %s partially contains' % hname)
                # partially contained in try_huc, return this
                return hint
            else:
                hname = watershed_workflow.sources.utils.get_code(subhu, search_level)
                logging.debug('  subhuc: %s does not contain' % hname)
        assert (False)

    if type(shape) is shapely.geometry.Polygon:
        shply = shape
    else:
        shply = watershed_workflow.utils.create_shply(shape)

    # must shrink the poly a bit in case it is close to or on a boundary
    radius = np.sqrt(shply.area / np.pi)
    shply_s = shply.buffer(-shrink_factor * radius)

    hint = watershed_workflow.sources.utils.huc_str(hint)

    _, hint_hu = get_huc(source, hint, in_crs)
    inhuc = _in_huc(shply_s, hint_hu)
    if inhuc != 2:
        raise RuntimeError("{}: shape not found in hinted HUC '{}'".format(source.name, hint))

    result = _findHUC(source, shply_s, in_crs, hint)
    return result


def reduceRivers(rivers,
                  ignore_small_rivers=0,
                  prune_by_area=None,
                  area_property='DivergenceRoutedDrainAreaSqKm',
                  remove_diversions=False,
                  remove_braided_divergences=False,
                  tol=0.1):
    """Create a river, which is a tree of reaches.
    
    Note, HUCs and rivers must be in the same crs.

    Parameters
    ----------
    rivers : list(river_tree.River)
        A list of rivers to reduce.
    ignore_small_rivers : int, optional
        If provided and positive, removes rivers whose number of
        reaches is less than this value.  If negative, keeps the N
        biggest (in number of reaches) rivers, where N is the negative
        of the provided value (e.g. -2 keeps the biggest 2 rivers).
    prune_by_area : float, optional
        If provided, remove reaches whose total contributing area is
        less than this tol.  NOTE: only valid for reaches that include
        a contributing area property (e.g. NHDPlus).
    area_property : str, optional='DivergenceRoutedDrainAreaSqKm'
        Name of the area property to use for determining reach CA.
        Note that this defines the units of prune_by_area value.
    remove_diversions : bool, optional=False
        If true, remove diversions (see documentation of
        modify_rivers_remove_divergences()).
    remove_braided_divergences : bool, optional=False
        If true, remove braided divergences (see documentation of
        modify_rivers_remove_divergences()).
    tol : float, optional=0.1
        Defines what close is in the case of method == 'geometry'

    Returns
    ------- 
    out : list(river_tree.River)
        A list of rivers, as River objects.

    """
    if ignore_small_rivers < 0:
        rivers = sorted(rivers, key=lambda a: len(a), reverse=True)
        rivers = rivers[0:-ignore_small_rivers]
        logging.info(f"Removing all but the biggest {-ignore_small_rivers} rivers")
    elif ignore_small_rivers > 0:
        rivers = watershed_workflow.hydrography.filterSmallRivers(rivers, ignore_small_rivers)
        if len(rivers) == 0:
            return rivers

    # note it is faster to remove all rivers with small area first
    if prune_by_area is not None:
        logging.info(f"Removing rivers with area < {prune_by_area}")
        rivers = [r for r in rivers if r.properties[area_property] > prune_by_area]
        if len(rivers) == 0:
            return rivers

    if remove_diversions and remove_braided_divergences:
        rivers = watershed_workflow.hydrography.removeDivergences(rivers)
    elif remove_diversions:
        rivers = watershed_workflow.hydrography.removeDiversions(rivers)
    elif remove_braided_divergences:
        rivers = watershed_workflow.hydrography.removeBraids(rivers)
    if len(rivers) == 0:
        return rivers

    if prune_by_area is not None:
        rivers = watershed_workflow.hydrography.pruneByArea(rivers, prune_by_area, area_property)

    if ignore_small_rivers > 0:
        rivers = watershed_workflow.hydrography.filterSmallRivers(rivers, ignore_small_rivers)
        if len(rivers) == 0:
            return rivers

    return rivers


def simplify(hucs,
             rivers,
             waterbodies=None,
             simplify_hucs=0,
             simplify_rivers=None,
             simplify_waterbodies=None,
             prune_tol=None,
             merge_tol=None,
             snap_tol=None,
             snap_triple_junctions_tol=None,
             snap_reach_endpoints_tol=None,
             snap_waterbodies_tol=None,
             cut_intersections=False):
    """Simplifies the HUC and river shapes.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object containing all reaches.
    rivers : list(River)
        A list of river objects.
    waterbodies : list(shply), optional
        A list of waterbodies.
    simplify_hucs : float, optional
        If provided, simply the hucs by moving points at most this
        many units (see also shapely.simplify).  Units are that of the
        CRS of shapes.
    simplify_rivers : float, optional
        If provided, simply the rivers by moving points at most this
        many units (see also shapely.simplify).  Units are that of the
        CRS of shapes.  If not provided, use the simplify_hucs value.
        Provide 0 to make no changes to the rivers.
    simplify_waterbodies : float, optional
        Simplify the waterbodies.  If not provided, uses the
        simplify_hucs value.  Provide 0 to make no changes to the
        waterbodies.
    prune_tol : float, optional = simplify_rivers
        Prune leaf reaches that are smaller than this tolerance.  If
        not provided, uses simplify_rivers value.  Provide 0 to not do
        this step.
    merge_tol : float, optional = simplify_rivers
        Merges reaches that are smaller than this tolerance with their
        downstream parent reach.  Note that if there is a branchpoint
        at the downstream node of the small reach, it will get moved
        to the upstream node.  If not provided, uses simplify_rivers
        value.  Provide 0 to not do this step.
    snap_tol : float, optional = 0.75 * simplify_rivers
        Tolerance used for snapping rivers to nearby huc boundaries.
        Provide 0 to not snap.
    snap_triple_junctions_tol : float, optional = 3 * snap_tol
        Tolerance used for snapping river triple junctions to huc
        triple junctions.
    snap_reach_endpoints_tol : float, optional = 2 * snap_tol
        Tolerance used for snapping reach junctions to huc boundaries.
    snap_waterbodies_tol : float, optional = snap_tol
        If not 0, snaps waterbody and HUC nodes that are nearly
        coincident to be discretely coincident.  Note this is not
        recommended; prefer to include major waterbodies in the HUC
        network.
    cut_intersections : bool, optional = False
        If true, force intersections of the river network and the HUC
        boundary to occur at a coincident node by adding nodes as
        needed.

    Returns
    -------
    rivers : list(River)
       Snap may change the rivers, so this returns the list of updated
       rivers.

    .. note: 
        This also may modify the hucs and waterbody objects in-place.

    """
    assert (type(hucs) is watershed_workflow.split_hucs.SplitHUCs)
    assert (type(rivers) is list)
    assert (all(type(r) is watershed_workflow.river_tree.River for r in rivers))

    if simplify_rivers is None:
        simplify_rivers = simplify_hucs
    if simplify_waterbodies is None:
        simplify_waterbodies = simplify_hucs
    if prune_tol is None:
        prune_tol = simplify_rivers
    if merge_tol is None:
        merge_tol = simplify_rivers
    if snap_tol is None:
        snap_tol = 0.75 * simplify_rivers
    if snap_triple_junctions_tol is None:
        snap_triple_junctions_tol = 3 * snap_tol
    if snap_reach_endpoints_tol is None:
        snap_reach_endpoints_tol = 2 * snap_tol
    if snap_waterbodies_tol is None:
        snap_waterbodies_tol = snap_tol

    logging.info("")
    logging.info("Simplifying")
    logging.info("-" * 30)

    if simplify_rivers > 0:
        logging.info("Simplifying rivers")
        watershed_workflow.hydrography.cleanup(rivers, simplify_rivers, prune_tol, merge_tol)

    if simplify_hucs > 0:
        logging.info("Simplifying HUCs")
        watershed_workflow.split_hucs.simplify(hucs, simplify_hucs)

    if snap_tol > 0 or snap_triple_junctions_tol > 0 or snap_reach_endpoints_tol > 0 or cut_intersections:
        logging.info("Snapping river and HUC (nearly) coincident nodes")
        rivers = watershed_workflow.hydrography.snap(hucs, rivers, snap_tol,
                                                     snap_triple_junctions_tol,
                                                     snap_reach_endpoints_tol, cut_intersections)

    if simplify_waterbodies > 0 and waterbodies is not None:
        for i, wb in enumerate(waterbodies):
            wb = wb.simplify(simplify_waterbodies)
            wb = hucs.exterior().intersection(wb)
            waterbodies[i] = wb

    if snap_waterbodies_tol > 0 and waterbodies is not None:
        logging.info("Snapping waterbodies and HUC (nearly) coincident nodes")
        watershed_workflow.hydrography.snap_waterbodies(hucs, waterbodies, snap_waterbodies_tol)

    assert (all(r.is_locally_continuous() for r in rivers))

    if cut_intersections:
        logging.info("Cutting crossings and removing external segments")
        watershed_workflow.hydrography.cut_and_snap_crossings(hucs, rivers, snap_tol)

    assert (all(r.is_locally_continuous() for r in rivers))

    logging.info("")
    logging.info("Simplification Diagnostics")
    logging.info("-" * 30)
    if len(rivers) != 0:
        mins = []
        for river in rivers:
            for line in river.depthFirst():
                coords = np.array(line.coords[:])
                dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
                mins.append(np.min(dz))
        logging.info(f"  river min seg length: {min(mins)}")
        logging.info(f"  river median seg length: {np.median(np.array(mins))}")

    mins = []
    watershed_workflow.split_hucs.simplify(hucs, 0)
    for line in hucs.segments:
        coords = np.array(line.coords[:])
        dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
        mins.append(np.min(dz))
    logging.info(f"  HUC min seg length: {min(mins)}")
    logging.info(f"  HUC median seg length: {np.median(np.array(mins))}")

    mins = []
    if waterbodies is not None:
        for wb in waterbodies:
            coords = np.array(wb.exterior.coords[:])
            dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
            mins.append(np.min(dz))
        if len(mins) > 0:
            logging.info(f"  Waterbody min seg length: {min(mins)}")
            logging.info(f"  Waterbody median seg length: {np.median(np.array(mins))}")
    return rivers


def densify(objct, target, objct_orig=None, rivers=None, **kwargs):
    """Redensify a river, huc, or waterbodies object, meeting a provided target or target resolution function.

    Parameters
    ----------
    objct : SplitHUCs, list(River), or list(shapely.Polygon)
      The object to be densified.
    target : float, list[float]
      Parameters for the target density -- either a float target
      length or a list of floats used in
      watershed_workflow.densification.limit_from_river_distance
      object.
    objct_orig : same as objct, optional
      The object with original coordinates.  The original,
      unsimplified object, if provided, allows better interpolation
      between the coarsened coordinates.
    rivers : optional
      If target is a list of floats, the rivers used in the signed
      distance function.
    **kwargs : optional
      Passed along to the densify function.
    """
    if isinstance(objct, watershed_workflow.split_hucs.SplitHUCs):
        return watershed_workflow.densification.densify_hucs(objct, objct_orig, rivers, target,
                                                             **kwargs)
    elif isinstance(objct[0], watershed_workflow.river_tree.River):
        return watershed_workflow.densification.densify_rivers(objct, objct_orig, target, **kwargs)
    else:
        raise ValueError("densify() currently only supports list(River) and SplitHUC objects.")


def triangulate(hucs,
                rivers=None,
                river_corrs=None,
                internal_boundaries=None,
                hole_points=None,
                diagnostics=True,
                verbosity=1,
                tol=1,
                refine_max_area=None,
                refine_distance=None,
                refine_max_edge_length=None,
                refine_min_angle=None,
                enforce_delaunay=False,
                river_region_dist=None):
    """Triangulates HUCs and rivers.

    Note, refinement of a given triangle is done if any of the provided
    criteria is met.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs()
    rivers : list[watershed_workflow.river_tree.River], optional
        List of rivers, used to refine the triangulation in conjunction with refine_distance.
    river_corrs : list[shapely.geometry.Polygon], optional
        List of rivers corridor polygons.
    internal_boundaries : list[shapely.geometry.Polygon, watershed_workflow.river_tree.River], optional
        List of objects, whose boundary (in the case of
        polygons/waterbodies) or reaches (in the case of River) will
        be present in the edges of the triangulation.
    hole_points : list(shapely.Point), optional
        List of points inside the polygons to be left as holes/voids (excluded from mesh)
    diagnostics : bool, optional
        Plot diagnostics graphs of the triangle refinement.
    tol : float, optional
        Set tolerance for minimum distance between two nodes. The unit is the same as 
        that of the watershed's CRS. The default is 1.
    refine_max_area : float, optional
        Refine a triangle if its area is greater than this area.
    refine_distance : list(float), optional
        Refine a triangle if its area is greater than a function of its
        centroid's distance from the nearest point on the river network.  The
        argument is given by:

        [near_distance, near_area, far_distance, far_area]

        Defining d as the distance from triangle centroid to the nearest point
        on the river network and area as the area of the triangle in question,
        refinement occurs if:

        * d < near_distance and area > near_area
        * d > far_distance and area > far_area
        * otherwise, defining 
          d' = (d - near_distance) / (far_distance - near_distance),
          refining occurs if
          area > near_area + (far_area - near_area) * d'

        Effectively this simply writes a piecewise linear function of triangle
        distance from centroid and uses that as a max area criteria.
    refine_max_edge_length : float, optional
        Refine a triangle if its max edge length is greater than this length.
    refine_min_angle : float, optional
        Try to ensure that all triangles have a minimum edge length greater
        than this value.
    enforce_delaunay : bool,optional, experimental
        Attempt to ensure all triangles are proper Delaunay triangles.

        .. note:
            This requires a hacked version of meshpy.triangle that
            supports this option.  See the patch available at
            workflow_tpls/meshpy_triangle.patch
    river_region_dist: float, optional
        Create river region based on the distance from river networks. This is useful if explicit 
        representation of riverbed is desired. Default is None.

    Returns
    -------
    vertices : np.array((n_points, 2), 'd')
        Array of triangle vertices.
    triangles : np.array((n_tris, 3), 'i')
        For each triangle, a list of 3 indices into the vertex array that make
        up that triangle.
    areas : _only if diagnostics=True_, np.array((n_tris), 'd')
        Array of triangle areas.

    """
    verbose = verbosity > 2

    logging.info("")
    logging.info("Triangulation")
    logging.info("-" * 30)

    refine_funcs = []
    if refine_max_area != None:
        refine_funcs.append(watershed_workflow.triangulation.refine_from_max_area(refine_max_area))
    if refine_distance != None:
        if river_corrs != None:
            refine_funcs.append(
                watershed_workflow.triangulation.refine_from_river_distance(
                    *refine_distance, river_corrs))
        else:
            refine_funcs.append(
                watershed_workflow.triangulation.refine_from_river_distance(
                    *refine_distance, rivers))
    if refine_max_edge_length != None:
        refine_funcs.append(
            watershed_workflow.triangulation.refine_from_max_edge_length(refine_max_edge_length))

    def my_refine_func(*args):
        return any(rf(*args) for rf in refine_funcs)

    vertices, triangles = watershed_workflow.triangulation.triangulate(
        hucs,
        river_corrs,
        internal_boundaries=internal_boundaries,
        hole_points=hole_points,
        tol=tol,
        verbose=verbose,
        refinement_func=my_refine_func,
        min_angle=refine_min_angle,
        enforce_delaunay=enforce_delaunay,
        allow_boundary_steiner=(river_corrs is None))

    if diagnostics or river_region_dist is not None:
        logging.info("Plotting triangulation diagnostics")
        river_multiline = shapely.geometry.MultiLineString([r for river in rivers for r in river])
        distances = []
        areas = []
        needs_refine = []
        for tri in triangles:
            verts = vertices[tri]
            bary = np.sum(np.array(verts), axis=0) / 3
            bary_p = shapely.geometry.Point(bary[0], bary[1])
            distances.append(bary_p.distance(river_multiline))
            areas.append(watershed_workflow.utils.triangle_area(verts))
            needs_refine.append(my_refine_func(verts, areas[-1]))
        areas = np.array(areas)
        distances = np.array(distances)
        logging.info("  min area = {}".format(areas.min()))
        logging.info("  max area = {}".format(areas.max()))

        if verbosity > 0:
            plt.figure()
            plt.subplot(121)
            plt.hist(distances)
            plt.xlabel("distance from river of triangle centroids [m]")
            plt.ylabel("count [-]")
            plt.subplot(122)
            plt.scatter(distances, areas, c=needs_refine, marker='x')
            plt.xlabel("distance [m]")
            plt.ylabel("triangle area [m^2]")

        if river_region_dist is not None:
            river_idx = distances < river_region_dist
            river_tris = triangles[river_idx]

            plt.figure()
            plt.tripcolor(vertices[:, 0],
                          vertices[:, 1],
                          triangles,
                          facecolors=np.array([0] * len(triangles)),
                          cmap=None,
                          edgecolors='w',
                          linewidth=0.01)
            plt.tripcolor(vertices[:, 0],
                          vertices[:, 1],
                          river_tris,
                          facecolors=np.array([1] * len(river_tris)),
                          cmap='jet',
                          edgecolors='w',
                          linewidth=0.1)
            plt.title("river region")

            return vertices, triangles, areas, distances, river_idx
        return vertices, triangles, areas, distances

    return vertices, triangles


def tessalateRiverAligned(hucs,
                            rivers,
                            river_width,
                            river_n_quads=1,
                            internal_boundaries=None,
                            hole_points=None,
                            diagnostics=False,
                            ax=None,
                            **kwargs):
    """Tessalate HUCs using river-aligned quads along the corridor and triangles away from it.

    Parameters
    ----------
    hucs : SplitHUCs
       The huc geometry to tessalate.  Note this will be adjusted if
       required by the river corridor.
    rivers : list[River]
       The rivers to mesh with quads
    river_width : float or dict or callable or boolean 
       Width of the quads, either a float or a dictionary providing a
       {StreamOrder : width} mapping.
       Or a function (callable) that computer width using node properties
       Or boolean, where True means, width for each reach is explicitely provided properties as "width"
    river_n_quads : int, optional
       Number of quads across the river.  Currently only 1 is
       supported (the default).
    hole_points : list(shapely.Point), optional
        List of points inside the polygons to be left as holes/voids (excluded from mesh)
    internal_boundaries : list[shapely.Polygon], optional
       List of internal boundaries to embed in the domain, e.g. waterbodies.
    diagnostics : bool, optional
       If true, prints extra diagnostic info.
    ax : matplotlib Axes object, optional
       For debugging -- plots troublesome reaches as quad elements are
       generated to find tricky areas.
    kwargs :
       All other arguments are passed to the triangulation function for refinement.

    Returns
    -------
    vertices : np.array((n_vertices, 2), 'd')
        Array of triangle vertices.
    cell_conn : list[list[int]]
        For each cell, an ordered list of indices into the vertices
        array that make up that cell.
    areas : _only if diagnostics=True_, np.array((n_cell_vertices), 'd')
        Array of areas.

    """
    logging.info("")
    logging.info("Stream-aligned Meshing")
    logging.info("-" * 30)

    # generate the quads
    logging.info('Creating stream-aligned mesh...')
    quad_conn, corrs = watershed_workflow.river_mesh.createRiverMeshes(rivers=rivers,
                                                                          widths=river_width,
                                                                          enforce_convexity=True,
                                                                          ax=ax,
                                                                          label=False)

    # adjust the HUC to match the corridor at the boundary
    logging.info('Adjusting rivers at the watershed boundaries...')
    hucs_without_outlet = hucs.deep_copy()
    watershed_workflow.river_mesh.adjustHUCsForRiverCorridors(hucs_without_outlet,
                                                                  rivers,
                                                                  corrs,
                                                                  integrate_rc=False,
                                                                  ax=ax)

    # triangulate the rest
    tri_res = watershed_workflow.triangulate(hucs_without_outlet, rivers, corrs,
                                             internal_boundaries, hole_points, diagnostics,
                                             **kwargs)
    tri_verts = tri_res[0]
    tri_conn = tri_res[1]

    # merge into a single output
    tri_conn_list = [conn.tolist() for conn in tri_conn]
    conn_list = tri_conn_list + quad_conn

    river_gid_start = len(tri_conn_list)
    for river in rivers:
        river.properties['gid_start'] = river_gid_start
        all_elems = [elem for node in river.preOrder() for elem in node.elements]
        river_gid_start = river_gid_start + len(all_elems)

    # note, all quad verts are in the tri_verts, and hopefully in the right order!
    if len(tri_res) > 2:
        return (tri_verts, conn_list) + tuple(tri_res[2:])
    else:
        return tri_verts, conn_list


def elevate(mesh_points, mesh_crs, dem, algorithm='piecewise bilinear'):
    """Elevate mesh_points onto the provided dem.

    Parameters
    ----------
    mesh_points : np.array((n_points, 2), 'd')
        Array of triangle vertices.
    mesh_crs : crs-type
        Mesh coordinate system.
    dem : xarray.DataArray
        2D array forming an elevation raster.
    algorithm : str, optional
        Algorithm used for interpolation.  One of:
        * "nearest" for nearest-neighbor pixels
        * "piecewise bilinear" for interpolation (default)

    Returns
    -------
    out : np.array((n_points, 3), 'd')
        Array of triangle vertices, including a z-dimension.

    """
    # index the i,j of the points, pick the elevations
    elev = interpolateFromArray(mesh_points, mesh_crs, dem, algorithm)

    # create the 3D points
    out = np.zeros((len(mesh_points), 3), 'd')
    out[:, 0:2] = mesh_points
    out[:, 2] = elev
    return out


def interpolateFromArray(array, points, points_crs=None, method='nearest'):
    """Interpolate a raster onto a collection of unstructured points.

    Parameters
    ----------
    array : xarray.DataArray
        2D array forming the raster.
    points : np.array((n_points, 2), 'd')
        Array of points to interpolate onto.
    points_crs : crs-type, optional
        Coordinate system of the points.  If not provided, it is
        assumed the same as array.
    algorithm : str, optional
        Algorithm used for interpolation.  One of:
        * "nearest" for nearest neighbor pixels (default)
        * "linear" for interpolation

    Returns
    -------
    out : np.array((n_points,))
        Array of raster values interpolated onto the points.

    """
    if points_crs is None:
        points_crs = watershed_workflow.crs.from_xarray(array)

    array_crs = watershed_workflow.crs.from_xarray(array)
    points_raster_crs = np.array(watershed_workflow.warp.xy(
        points[:, 0], points[:, 1], points_crs, array_crs))

    x = xarray.DataArray(points_raster_crs[0], dims='points')
    y = xarray.DataArray(points_raster_crs[1], dims='points')
    return array.interp(x=x, y=y, method=method).values


def colorRasterFromShapes(shapes,
                          shape_color_column,
                          raster_bounds,
                          raster_dx,
                          raster_crs=None,
                          nodata=None):
    """Color in a raster by filling in a collection of shapes.

    Given a canvas specified by bounds and pixel size, color a raster by, for
    each shape, finding the intersection of that shape with the canvas and
    coloring it by a provided value.  Paint by numbers.

    Note, if the shapes overlap, the last shape containing a pixel gives the
    color of that pixel.

    Parameters
    ----------
    shapes : geopandas.GeoDataFrame
        Collection of shapes (likely) overlapping the canvas.
    shape_color_column : string
        Column of shapes that is the color.
    raster_bounds : [xmin, ymin, xmax, ymax]
        Bounding box for the output raster, in the given CRS.
    raster_dx : float
        Pixel size (assumed the same in both x and y).
    raster_crs : crs-type, optional=shapes_crs
        Coordinate system of the raster.
    nodata : dtype, optional={-1 (int), nan (float)}
        Value to place in pixels which intersect no shape.  Note the type of
        this should be the same as the type of shape_colors.

    Returns
    -------
    xarray 

    """
    assert (len(shapes) == len(shape_colors))
    if len(shapes) == 0:
        raise ValueError("Cannot generate raster for empty set of shapes")

    logging.info('Coloring shapes onto raster:')

    if not watershed_workflow.crs.isEqual(shapes_crs, raster_crs):
        shapes = watershed_workflow.warp.shplys(shapes, shapes_crs, raster_crs)

    dtype = np.dtype(type(shape_colors[0]))

    if nodata is None:
        try:
            nodata = dtype(np.nan)
        except ValueError:
            nodata = dtype(-1)

    raster_profile, raster = watershed_workflow.utils.create_empty_raster(
        raster_bounds, raster_crs, raster_dx, nodata)
    assert (len(raster.shape) == 3 and raster.shape[0] == 1)
    raster = raster[0, :, :]
    logging.info(f'  of shape: {raster.shape}')
    logging.info(f'  and {len(set(shape_colors))} independent colors')

    for p, p_id in zip(shapes, shape_colors):
        if not p.is_empty:
            p_list = watershed_workflow.utils.flatten([p, ])
            mask = rasterio.features.geometry_mask(p_list,
                                                   raster.shape,
                                                   raster_profile['transform'],
                                                   invert=True)
            raster[mask] = p_id
    return raster_profile, raster


def colorExistingRasterFromShapes(shapes_df, shape_colors, raster, raster_profile):
    """Color in a raster by filling in a collection of shapes.

    Given a canvas, find the intersection of that shape with the canvas and
    coloring it by a provided value.  Paint by numbers.

    Note, if the shapes overlap, the last shape containing a pixel gives the
    color of that pixel.

    Parameters
    ----------
    shapes : list(Polygon)
        Collection of shapes (likely) overlapping the canvas.
    shapes_crs : crs-type
        Coordinate system of the shapes.
    shapes_colors : iterable[]
        Color to label the interior of each polygon with.
    raster : np.ndarray
        The canvas to color on.
    raster_profile : dict
        Rasterio style profile including at least CRS, nodata, and
        transform.

    """
    assert (len(shapes) == len(shape_colors))
    if len(shapes) == 0:
        raise ValueError("Cannot generate raster for empty set of shapes")

    logging.info('Coloring shapes onto raster:')
    logging.info(f'  and {len(set(shape_colors))} independent colors')

    if not watershed_workflow.crs.isEqual(shapes_crs, raster_profile['crs']):
        shapes = watershed_workflow.warp.shplys(shapes, shapes_crs, raster_profile['crs'])

    for p, p_id in zip(shapes, shape_colors):
        if not p.is_empty:
            p_list = watershed_workflow.utils.flatten([p, ])
            mask = rasterio.features.geometry_mask(p_list,
                                                   raster.shape,
                                                   raster_profile['transform'],
                                                   invert=True)
            raster[mask] = p_id
