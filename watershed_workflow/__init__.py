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
from __future__ import annotations

from . import _version
__version__ = _version.get_versions()['version']

from typing import Any, Optional, Iterable, List
import logging
import math
import numpy as np
import geopandas as gpd
import shapely.geometry
import xarray
from matplotlib import pyplot as plt
import folium
import folium.plugins

import watershed_workflow.crs
import watershed_workflow.utils
import watershed_workflow.sources.standard_names as names
import watershed_workflow.source_list

import watershed_workflow.river_tree
from watershed_workflow.river_tree import River
import watershed_workflow.split_hucs
from watershed_workflow.split_hucs import SplitHUCs

import watershed_workflow.hydrography
import watershed_workflow.resampling
import watershed_workflow.angles
import watershed_workflow.triangulation
import watershed_workflow.river_mesh
import watershed_workflow.condition
import watershed_workflow.datasets


def _coerceShapes(df : gpd.GeoDataFrame,
                  crs : Optional[watershed_workflow.crs.CRS] = None,
                  digits : Optional[int] = None) -> gpd.GeoDataFrame:
    # often we end up with mixed data -- some 2D, some 3D, which makes
    # it hard to deal with intersections.  Remove all z coordinates.
    df['geometry'] = df.geometry.apply(watershed_workflow.utils.removeThirdDimension)

    # often data APIs provide all MultiGeometries, when in fact they are a single LineString/Polygon
    def _combine(shp : shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
        if isinstance(shp, shapely.geometry.MultiLineString):
            return shapely.line_merge(shp)
        elif isinstance(shp, shapely.geometry.MultiPolygon):
            return shapely.union_all(shp.geoms)
        return shp
    df['geometry'] = df.geometry.apply(_combine)

    # change the crs
    if crs is not None:
        old_geo = df.active_geometry_name
        for col in df.select_dtypes('geometry'):
            df = df.set_geometry(col).to_crs(crs)
        df = df.set_geometry(old_geo)

    # round to s fixed number of digits -- this may get deprecated in
    # favor of using the grid option to shapely 2.0's
    # union/intersection
    if digits is not None:
        df = df.set_precision(10**-digits)
    return df


def getShapes(source : Any,
              crs : Optional[watershed_workflow.crs.CRS] = None,
              digits : Optional[int] = None,
              **kwargs) -> gpd.GeoDataFrame:
    if isinstance(source, str):
        source = watershed_workflow.source_list.ManagerShapefile(source)
    df = source.getShapes()
    df = _coerceShapes(df, crs, digits)
    return df
              

def getShapesByID(source : Any,
                  ids : Any,
                  crs : Optional[watershed_workflow.crs.CRS] = None,
                  digits : Optional[int] = None,
                  **kwargs) -> gpd.GeoDataFrame:
    if isinstance(ids, str):
        ids = [ids,]
    elif not isinstance(ids, Iterable):
        ids = [ids,]

    if len(kwargs) > 0:
        source.set(**kwargs)
    df = source.getShapesByID(ids)
    df = _coerceShapes(df, crs, digits)
    return df


def getShapesByGeometry(source : Any,
                        geom : shapely.geometry.base.BaseGeometry,
                        geom_crs : watershed_workflow.crs.CRS,
                        crs : Optional[watershed_workflow.crs.CRS] = None,
                        digits : Optional[int] = None,
                        **kwargs) -> gpd.GeoDataFrame:
    df = source.getShapesByGeometry(geom, geom_crs, **kwargs)
    df = _coerceShapes(df, crs, digits)
    return df


#
# functions for relating objects
# -----------------------------------------------------------------------------
def findHUC(source : Any,
            shape : shapely.geometry.Polygon,
            in_crs : watershed_workflow.crs.CRS,
            hint : str,
            shrink_factor : float = 1.e-5) -> str:
    """Finds the smallest HUC containing shape.

    Parameters
    ----------
    source : source-type
        Source object providing a getShapes() method that gets HUCs by ID.
    shape : shapely.geometry.Polygon
        Find this shape in a HUC.
    in_crs : watershed_workflow.crs.CRS
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

        subhus = getShapesByID(source, hint, crs, level=search_level)

        for index, subhu in zip(subhus.index, subhus.geometry):
            inhuc = _in_huc(shply, subhu)

            if inhuc == 2:
                # fully contained in try_huc, recurse
                logging.debug(f'  subhuc: {index} contains')
                return _findHUC(source, shply, crs, index)
            elif inhuc == 1:
                logging.debug(f'  subhuc: {index} partially contains')
                # partially contained in try_huc, return this
                return hint
            else:
                logging.debug(f'  subhuc: {index} does not contain')
        assert False

    # must shrink the poly a bit in case it is close to or on a boundary
    radius = math.sqrt(shape.area / math.pi)
    shape_s = shape.buffer(-shrink_factor * radius)

    hint_hu = getShapesByID(source, hint, in_crs)
    inhuc = _in_huc(shape_s, hint_hu.geometry.iloc[0])
    if inhuc != 2:
        raise RuntimeError(f"{source.__class__}: shape not found in hinted HUC '{hint}'")

    result = _findHUC(source, shape_s, in_crs, hint)
    return result


def reduceRivers(rivers : List[River],
                 ignore_small_rivers : int = 0,
                 keep_n_rivers : int = -1,
                 prune_by_area : float = 0.0,
                 area_property : str = names.DRAINAGE_AREA,
                 remove_diversions : bool = False,
                 remove_braided_divergences : bool = False,
                 tol : Optional[float] = 0.1) -> List[River]:
    """Reduce the extent of the river network through a variety of methods.

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
    if keep_n_rivers > 0:
        logging.info(f"Removing all but the biggest {keep_n_rivers} rivers")
        rivers = sorted(rivers, key=lambda a: len(a), reverse=True)
        rivers = rivers[0:keep_n_rivers]
        
    if ignore_small_rivers > 0:
        rivers = watershed_workflow.river_tree.filterSmallRivers(rivers, ignore_small_rivers)

    if prune_by_area > 0.0:
        logging.info(f"Removing rivers with area < {prune_by_area}")
        rivers = [r for r in rivers if r.properties[area_property] > prune_by_area]

    if remove_diversions and remove_braided_divergences:
        rivers = watershed_workflow.river_tree.removeDivergences(rivers)
    elif remove_diversions:
        rivers = watershed_workflow.river_tree.removeDiversions(rivers)
    elif remove_braided_divergences:
        rivers = watershed_workflow.river_tree.removeBraids(rivers)

    if prune_by_area is not None:
        rivers = watershed_workflow.river_tree.pruneRiversByArea(rivers, prune_by_area, area_property)

    if ignore_small_rivers > 0:
        rivers = watershed_workflow.river_tree.filterSmallRivers(rivers, ignore_small_rivers)

    return rivers


def simplify(hucs : SplitHUCs,
             rivers : List[River],
             reach_segment_target_length : float,
             huc_segment_target_length : Optional[float] = None,
             river_close_distance : float = 100.0,
             river_far_distance : float = 500.0,
             resample_by_reach_property : bool = False,
             min_angle : float = 20,
             junction_min_angle : float = 20,
             plot_diagnostics : bool = False,
             keep_points : bool = False) -> None:
    """Simplifies, in place, the HUC and river shapes to create constrained, discrete segments.

    Parameters
    ----------
    hucs : SplitHUCs
       A split-form HUC object containing all reaches.
    rivers : list(River)
       A list of river objects.
    huc_segment_target_length : float
       Target length of a typical triangle edge away from the river.
    reach_segment_target_length : float
       Target length of a typical triangle edge at the river.

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

    """
    logging.info("")
    logging.info("Simplifying")
    logging.info("-" * 30)

    assert len(rivers) > 0

    logging.info(f"Presimplify to remove colinear, coincident points.")
    presimplify = 1.e-4 * reach_segment_target_length
    watershed_workflow.river_tree.simplify(rivers, presimplify)
    watershed_workflow.split_hucs.simplify(hucs, presimplify)

    logging.info(f"Pruning leaf reaches < {reach_segment_target_length}")
    for river in rivers:
        watershed_workflow.river_tree.pruneByLineStringLength(river, reach_segment_target_length)

    logging.info(f"Merging internal reaches < {reach_segment_target_length}")
    for river in rivers:
        watershed_workflow.river_tree.mergeShortReaches(river, 0.75*reach_segment_target_length)

    watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
    watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")
        
    logging.info("Snapping discrete points to make rivers and HUCs discretely consistent.")
    logging.info(" -- snapping HUC triple junctions to reaches")
    snap_triple_junctions_tol = 3*reach_segment_target_length
    watershed_workflow.hydrography.snapHUCsJunctions(hucs, rivers, snap_triple_junctions_tol)
    watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
    watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")

    logging.info(" -- snapping reach endpoints to HUC boundaries")
    for river in rivers:
        watershed_workflow.hydrography.snapReachEndpoints(hucs, river, reach_segment_target_length)
        assert river.isContinuous()
    watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
    watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")

    logging.info(" -- cutting reaches at HUC boundaries")
    watershed_workflow.hydrography.cutAndSnapCrossings(hucs, rivers, reach_segment_target_length)
    for river in rivers:
        assert river.isContinuous()
    watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
    watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")
    
    logging.info("")
    logging.info("Simplification Diagnostics")
    logging.info("-" * 30)
    watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
    watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")

    # resample
    logging.info("")
    logging.info("Resampling HUC and river")
    logging.info("-" * 30)

    if huc_segment_target_length is not None:
        dfunc = (river_close_distance, reach_segment_target_length, 
                 river_far_distance, huc_segment_target_length)
        river_mls = shapely.ops.unary_union([river.to_mls() for river in rivers])
        logging.info(f" -- resampling HUCs based on distance function {dfunc}")
        watershed_workflow.resampling.resampleSplitHUCs(hucs, dfunc, river_mls, keep_points=keep_points)
    else:
        logging.info(f" -- resampling HUCs based on uniform target {reach_segment_target_length}")
        watershed_workflow.resampling.resampleSplitHUCs(hucs, reach_segment_target_length, keep_points=keep_points)

    if resample_by_reach_property:
        logging.info(f" -- resampling reaches based on TARGET_SEGMENT_LENGTH property")
        watershed_workflow.resampling.resampleRivers(rivers, keep_points=keep_points)
    else:
        logging.info(f" -- resampling reaches based on uniform target {reach_segment_target_length}")
        watershed_workflow.resampling.resampleRivers(rivers, reach_segment_target_length, keep_points=keep_points)

    logging.info("")
    logging.info("Resampling Diagnostics")
    logging.info("-" * 30)
    if plot_diagnostics:
        fig, ax = plt.subplots(1,1)
        watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach", ax=ax, color='b')
        watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ", ax=ax, color='orange')
    else:
        watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
        watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")

    # fix bad angles
    logging.info("")
    logging.info("Clean up sharp angles, both internally and at junctions.")
    logging.info("-" * 30)
    count = watershed_workflow.angles.smoothSharpAngles(hucs, rivers, min_angle, junction_min_angle)
    logging.info(f"Cleaned up {count} sharp angles.")
    watershed_workflow.utils.logMinMaxMedianSegment((r.linestring for river in rivers for r in river), "reach")
    watershed_workflow.utils.logMinMaxMedianSegment(hucs.linestrings, "HUC  ")
    

def triangulate(hucs : SplitHUCs,
                rivers : Optional[River] = None,
                internal_boundaries : Optional[List[shapely.geometry.BaseGeometry | River]] = None,
                hole_points : Optional[List[shapely.geometry.Point]] = None,
                diagnostics : bool = False,
                verbosity : int = 1,
                tol : float = 1.0,
                refine_max_area : Optional[float] = None,
                refine_distance : Optional[float] = None,
                refine_max_edge_length : Optional[float] = None,
                refine_min_angle : Optional[float] = None,
                enforce_delaunay : bool = False,
                as_mesh : bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulates HUCs and rivers.

    Note, refinement of a given triangle is done if any of the provided
    criteria is met.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs()
    rivers : list[watershed_workflow.river_tree.River], optional
        List of rivers, used to refine the triangulation in conjunction with refine_distance.
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

    Returns
    -------
    vertices : np.array((n_points, 2), 'd')
        Array of triangle vertices.
    triangles : np.array((n_tris, 3), 'i')
        For each triangle, a list of 3 indices into the vertex array that make
        up that triangle.
    areas : _only if diagnostics=True_, np.array((n_tris), 'd')
        Array of triangle areas.
    distances : _only if diagnostics=True_, np.array((n_tris), 'd')
        Array of triangle distances from the river network.

    """
    verbose = verbosity > 2

    logging.info("")
    logging.info("Triangulation")
    logging.info("-" * 30)

    refine_funcs = []
    if refine_max_area != None:
        refine_funcs.append(watershed_workflow.triangulation.refineByMaxArea(refine_max_area))
    if refine_distance != None:
        refine_funcs.append(
            watershed_workflow.triangulation.refineByRiverDistance(*refine_distance, rivers))
    if refine_max_edge_length != None:
        refine_funcs.append(
            watershed_workflow.triangulation.refineByMaxEdgeLength(refine_max_edge_length))

    def my_refine_func(*args):
        return any(rf(*args) for rf in refine_funcs)

    vertices, triangles = watershed_workflow.triangulation.triangulate(
        hucs,
        internal_boundaries=internal_boundaries,
        hole_points=hole_points,
        tol=tol,
        verbose=verbose,
        refinement_func=my_refine_func,
        min_angle=refine_min_angle,
        enforce_delaunay=enforce_delaunay,
        allow_boundary_steiner=False)

    
    if diagnostics:
        logging.info("Plotting triangulation diagnostics")
        river_multiline = shapely.ops.unary_union([river.to_mls() for river in rivers])
        distances = []
        areas = []
        needs_refine = []
        for tri in triangles:
            verts = vertices[tri]
            bary = np.sum(np.array(verts), axis=0) / 3
            bary_p = shapely.geometry.Point(bary[0], bary[1])
            distances.append(bary_p.distance(river_multiline))
            areas.append(watershed_workflow.utils.computeTriangleArea(*verts))
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

        if as_mesh:
            m2 = watershed_workflow.mesh.Mesh2D(vertices, triangle, crs=hucs.crs)
            return m2, areas, distances
        else:
            return vertices, triangles, areas, distances
    
    if as_mesh:
        return watershed_workflow.mesh.Mesh2D(vertices, triangle, crs=hucs.crs)
    else:
        return vertices, triangles


def tessalateRiverAligned(hucs : SplitHUCs,
                          rivers : List[River],
                          river_width : Any,
                          internal_boundaries : Optional[List[River | shapely.geometry.base.BaseGeometry]] = None,
                          as_mesh : bool = True,
                          debug : bool = False,
                          **kwargs) -> Tuple[np.ndarray, List[List[int]]] | \
                                       Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray]:
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
    elems : list[list[int]]
        For each element, an ordered list of indices into the vertices
        array that make up that cell.
    areas : _only if diagnostics=True_, np.array((n_cell_vertices), 'd')
        Array of areas.

    """
    logging.info("")
    logging.info("Stream-aligned Meshing")
    logging.info("-" * 30)

    # generate the quads
    logging.info('Creating stream-aligned mesh...')
    computeWidth = watershed_workflow.river_mesh.createWidthFunction(river_width)

    if debug:
        fig, ax = plt.subplots(1,1)
    else:
        ax = None
    
    river_coords, river_elems, river_corridors, hole_points = watershed_workflow.river_mesh.createRiversMesh(hucs, rivers, computeWidth, ax=ax)
    if debug:
        plt.show()

    # triangulate the rest
    if internal_boundaries is None:
        internal_boundaries = river_corridors
    else:
        internal_boundaries = river_corridors + internal_boundaries
        
    tri_res  = watershed_workflow.triangulate(hucs,
                                              rivers,
                                              internal_boundaries,
                                              hole_points,
                                              as_mesh = False,
                                              **kwargs)
    
    tri_coords = tri_res[0]
    tri_elems = [tri.tolist() for tri in tri_res[1]]
    
    # merge elements into a single output
    elems = tri_elems + river_elems
    # note, all river verts are in the tri_verts, listed first, and in the same order!
    coords = tri_coords

    if as_mesh:
        m2 = watershed_workflow.mesh.Mesh2D(coords, elems, crs=hucs.crs)
        if len(tri_res) > 2:
            return (m2,) + tuple(tri_res[2:])
        else:
            return m2
        
    else:
        if len(tri_res) > 2:
            return (coords, elems) + tuple(tri_res[2:])
        else:
            return coords, elems


def elevate(m2 : Mesh2D,
            dem : xarray.DataSet,
            **kwargs) -> np.ndarray:
    """Elevate a mesh onto the provided dem, in place.

    Parameters
    ----------
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
    mesh_points = m2.coords
    
    # index the i,j of the points, pick the elevations
    elev = watershed_workflow.datasets.interpolateDataset(mesh_points, m2.crs, dem, **kwargs)

    # create the 3D points
    if mesh_points.shape[1] == 3:
        mesh_points[:,2] = elev
    else:
        new_points = np.zeros((len(mesh_points), 3), 'd')
        new_points[:, 0:2] = mesh_points
        new_points[:, 2] = elev
        m2.coords = new_points




# def colorRasterFromShapes(shapes,
#                           shape_color_column,
#                           raster_bounds,
#                           raster_dx,
#                           raster_crs=None,
#                           nodata=None):
#     """Color in a raster by filling in a collection of shapes.

#     Given a canvas specified by bounds and pixel size, color a raster by, for
#     each shape, finding the intersection of that shape with the canvas and
#     coloring it by a provided value.  Paint by numbers.

#     Note, if the shapes overlap, the last shape containing a pixel gives the
#     color of that pixel.

#     Parameters
#     ----------
#     shapes : geopandas.GeoDataFrame
#         Collection of shapes (likely) overlapping the canvas.
#     shape_color_column : string
#         Column of shapes that is the color.
#     raster_bounds : [xmin, ymin, xmax, ymax]
#         Bounding box for the output raster, in the given CRS.
#     raster_dx : float
#         Pixel size (assumed the same in both x and y).
#     raster_crs : crs-type, optional=shapes_crs
#         Coordinate system of the raster.
#     nodata : dtype, optional={-1 (int), nan (float)}
#         Value to place in pixels which intersect no shape.  Note the type of
#         this should be the same as the type of shape_colors.

#     Returns
#     -------
#     xarray 

#     """
#     assert len(shapes) == len(shape_colors)
#     if len(shapes) == 0:
#         raise ValueError("Cannot generate raster for empty set of shapes")

#     logging.info('Coloring shapes onto raster:')

#     if not watershed_workflow.crs.isEqual(shapes_crs, raster_crs):
#         shapes = watershed_workflow.warp.shplys(shapes, shapes_crs, raster_crs)

#     dtype = np.dtype(type(shape_colors[0]))

#     if nodata is None:
#         try:
#             nodata = dtype(np.nan)
#         except ValueError:
#             nodata = dtype(-1)

#     raster_profile, raster = watershed_workflow.utils.create_empty_raster(
#         raster_bounds, raster_crs, raster_dx, nodata)
#     assert len(raster.shape) == 3 and raster.shape[0] == 1
#     raster = raster[0, :, :]
#     logging.info(f'  of shape: {raster.shape}')
#     logging.info(f'  and {len(set(shape_colors))} independent colors')

#     for p, p_id in zip(shapes, shape_colors):
#         if not p.is_empty:
#             p_list = watershed_workflow.utils.flatten([p, ])
#             mask = rasterio.features.geometry_mask(p_list,
#                                                    raster.shape,
#                                                    raster_profile['transform'],
#                                                    invert=True)
#             raster[mask] = p_id
#     return raster_profile, raster


# def colorExistingRasterFromShapes(shapes_df, shape_colors, raster, raster_profile):
#     """Color in a raster by filling in a collection of shapes.

#     Given a canvas, find the intersection of that shape with the canvas and
#     coloring it by a provided value.  Paint by numbers.

#     Note, if the shapes overlap, the last shape containing a pixel gives the
#     color of that pixel.

#     Parameters
#     ----------
#     shapes : list(Polygon)
#         Collection of shapes (likely) overlapping the canvas.
#     shapes_crs : crs-type
#         Coordinate system of the shapes.
#     shapes_colors : iterable[]
#         Color to label the interior of each polygon with.
#     raster : np.ndarray
#         The canvas to color on.
#     raster_profile : dict
#         Rasterio style profile including at least CRS, nodata, and
#         transform.

#     """
#     assert len(shapes) == len(shape_colors)
#     if len(shapes) == 0:
#         raise ValueError("Cannot generate raster for empty set of shapes")

#     logging.info('Coloring shapes onto raster:')
#     logging.info(f'  and {len(set(shape_colors))} independent colors')

#     if not watershed_workflow.crs.isEqual(shapes_crs, raster_profile['crs']):
#         shapes = watershed_workflow.warp.shplys(shapes, shapes_crs, raster_profile['crs'])

#     for p, p_id in zip(shapes, shape_colors):
#         if not p.is_empty:
#             p_list = watershed_workflow.utils.flatten([p, ])
#             mask = rasterio.features.geometry_mask(p_list,
#                                                    raster.shape,
#                                                    raster_profile['transform'],
#                                                    invert=True)
#             raster[mask] = p_id


def makeMap(m):
    folium.LayerControl().add_to(m)
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MeasureControl().add_to(m)
    return m
