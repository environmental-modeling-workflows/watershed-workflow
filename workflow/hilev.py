"""High-level routines, espeically those that interact with data,
i.e. call download_*() or load_*() functions from conf.

Most scripts use these functions instead of directly using lower-level
capability.

Note, everything from here gets directly imported into the top level
package.
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltc
import logging

import fiona
import rasterio
import rasterio.transform
import rasterio.features
#import rasterio.mask
import shapely
import meshpy.triangle

from workflow_tpls import vtk_io  # from ATS/tools/meshing_ats

import workflow.conf
import workflow.triangulation
import workflow.warp
import workflow.plot
import workflow.tree
import workflow.split_hucs
import workflow.hydrography
import workflow.sources.utils 
from workflow.sources.utils import huc_str
import workflow.sources.manager_shape


#
# functions for getting objects
# -----------------------------------------------------------------------------

def get_huc(source, huc, crs=None, centering=None):
    """Get a HUC shape object from a given code.

    Arguments:
        source: source object providing `get_hucs()`
        huc (str): hydrologic unit code
        crs (:obj:`crs`, optional): Output coordinate system. 
            Defaults to `workflow.conf.default_scrs()`.
        centering (optional): If False (default) does nothing.
            If True or 'geometric', centers based on the geometric center.
            If 'mass', centers based on the center of mass.
            If Point object or tuple of two floats, centers on this coordinate.

    Returns:
        :obj:`shapely`: shapely polygon for the hydrologic unit.
        :obj:`shapely.Point`: The centering point.  `(0,0)` if `centering==False`,
            the centering point otherwise.
    """
    huc = huc_str(huc)
    hu_shapes, centroid = get_hucs(source, huc, len(huc), crs, centering)
    assert(len(hu_shapes) == 1)
    return hu_shapes[0], centroid

def get_hucs(source, huc, level, crs=None, centering=None):
    """Get a list of shape objects for all HUCs at level contained in huc.

    Arguments:
        source: source object providing `get_hucs()`
        huc (str): hydrologic unit code
        level (int): HUC level of the requested sub-basins
        crs (:obj:`crs`, optional): Output coordinate system. 
            Defaults to `workflow.conf.default_scrs()`.
        centering (optional): If False (default) does nothing.
            If True or 'geometric', centers based on the geometric center.
            If 'mass', centers based on the center of mass.
            If Point object or tuple of two floats, centers on this coordinate.

    Returns:
        list(:obj:`shapely`): the shapely polygons
        :obj:`shapely.Point`: The centering point.  `(0,0)` if `centering==False`,
            the centering point otherwise.
    """
    # get the hu from source
    huc = huc_str(huc)
    if level is None:
        level = len(huc)

    logging.info("")
    logging.info("Preprocessing HUC")
    logging.info("-"*30)
    logging.info("Loading level {} HUCs in {}.".format(level, huc))
    
    profile, hus = source.get_hucs(huc, level)
    logging.info('  found {} HUCs.'.format(len(hus)))
    for hu in hus:
        logging.info('  -- {}'.format(hu['properties']['HUC{:d}'.format(level)]))
    
    # convert to destination crs
    if crs is None:
        crs = workflow.conf.default_crs()
    if crs != profile['crs']:
        for hu in hus:
            workflow.warp.warp_shape(hu, profile['crs'], crs)

    # round
    workflow.utils.round(hus, workflow.conf.rcParams['digits'])

    # convert to shapely
    hu_shapes = [workflow.utils.shply(hu['geometry']) for hu in hus]

    # center
    if centering:
        hu_shapes, centroid = workflow.utils.center(hu_shapes, centering)
    else:
        centroid = shapely.geometry.Point(0,0)

    return hu_shapes, centroid


def get_split_form_hucs(source, huc, level=None, crs=None, centering=False):
    """Get a SplitHUCs object for all HUCs at level contained in huc.

    A :obj:`SplitHUCs` object is an object which stores a collection
    of polygons which share boundaries in a format that makes changing
    those shared boundaries possible without having to update all
    shapes that share the boundary.

    Arguments:
        source: source object providing `get_hucs()`
        huc (str): hydrologic unit code
        level (int, optional): HUC level of the requested sub-basins.
            Defaults to the level of huc.
        crs (:obj:`crs`, optional): Output coordinate system. 
            Defaults to `workflow.conf.default_scrs()`.
        centering (optional): If False (default) does nothing.
            If True or 'geometric', centers based on the geometric center.
            If 'mass', centers based on the center of mass.
            If Point object or tuple of two floats, centers on this coordinate.

    Returns:
      :obj:`SplitHUCs`: the HUCs in tiled form
      centroid        | shapely point for the centroid, based on the
                      |  value of centering.

    """
    hu_shapes, centroid = get_hucs(source, huc, level, crs, centering)
    # hu_ind = np.argmax([h.centroid.xy[1][0] for h in hu_shapes])
    # logging.info('bad huc = {}'.format(hu_ind))
    return workflow.split_hucs.SplitHUCs(hu_shapes), centroid


def get_shapes(source, index=-1, crs=None, centering=None):
    """Read a shapefile.

    Arguments:
      filename  | File to parse, should end in .shp
      index     | Index of the requested shape in filename, or -1 to get all.
      crs       | provides the output coordinate system, 
                | defaults to whatever the file is in.
      centering | if False or None (default) does nothing.
                |  if True or 'geometric', centers based on the 
                |  geometric center.
                |  if 'mass', centers based on the center of mass.
                |  if Point object or tuple of two floats, centers 
                |  on this coordinate.

    Returns (shapes, centroid)
      shapes    | list of shapely polygons for the requested shapes
      centroid  | shapely point for the centroid, based on the
                |  value of centering.
    """
    logging.info("")
    logging.info("Preprocessing Shapes")
    logging.info("-"*30)

    # load shapefile
    if type(source) is str:
        logging.info('loading file: "{}"'.format(source))
        source = workflow.sources.manager_shape.FileManagerShape(source)
        
    if index is None or index == -1:
        filter = None
    else:
        filter = lambda i,a: i == index
    profile, shps = source.get_shapes(filter=filter)

    # convert to destination crs
    if crs is None:
        crs = profile['crs']
    elif crs != profile['crs']:
        for shp in shps:
            workflow.warp.warp_shape(shp, profile['crs'], crs)

    # round
    workflow.utils.round(shps, workflow.conf.rcParams['digits'])

    # convert to shapely
    shplys = [workflow.utils.shply(shp['geometry']) for shp in shps]

    # center
    if centering:
        shplys, centroid = workflow.utils.center(shplys, centering)
    else:
        centroid = shapely.geometry.Point(0,0)

    return shplys, crs, centroid


def get_shapes_in_bounds(source, bounds, crs, centering=None):
    """Read a shapefile.

    Arguments:
      source    | Source object
      bounds    | Collect shapes which intersect these bounds.
      crs       | provides the coordinate system of the bounds and the 
                |  output coordinate system of the shapes.
      centering | if False or None (default) does nothing.
                |  if True or 'geometric', centers based on the 
                |  geometric center.
                |  if 'mass', centers based on the center of mass.
                |  if Point object or tuple of two floats, centers 
                |  on this coordinate.

    Returns (shapes, centroid)
      shapes    | list of shapely polygons for the requested shapes
      properties| list of properties associated with the shapes
      centroid  | shapely point for the centroid, based on the
                |  value of centering.
    """
    logging.info("")
    logging.info("Preprocessing Shapes")
    logging.info("-"*30)

    profile, shps, properties = source.get_shapes_in_bounds(bounds, crs)

    # convert to destination crs
    if crs != profile['crs']:
        shps = [workflow.warp.warp_shapely(shp, profile['crs'], crs) for shp in shps]

    # round
    # workflow.utils.round(shps, workflow.conf.rcParams['digits'])

    # center
    if centering:
        shps, centroid = workflow.utils.center(shps, centering)
    else:
        centroid = shapely.geometry.Point(0,0)

    return shps, crs, properties, centroid


def get_split_form_shapes(source, index, crs=None, centering=False):
    """Read a shapefile.

    Arguments:
      source    | File to parse, should end in .shp, or source for get_shapes()
      index     | Index of the requested shape in filename, or -1 to get all.
      crs       | provides the output coordinate system, 
                | defaults to workflow.conf.default_scrs()
      centering | if False or None (default) does nothing.
                |  if True or 'geometric', centers based on the 
                |  geometric center.
                |  if 'mass', centers based on the center of mass.
                |  if Point object or tuple of two floats, centers 
                |  on this coordinate.

    Returns: (split_form_shapes, centroid)
      split_form_shapes | the shapes in tiled form for geometric manipulation
      centroid          | shapely point for the centroid, based on the
                        |  value of centering.

    """
    shapes, crs, centroid = get_shapes(source, index, crs, centering)
    return workflow.split_hucs.SplitHUCs(shapes), centroid


def get_rivers_by_bounds(source, bounds, bounds_crs, huc_hint, centering=None, long=None, merge=True):
    """Collects shapefiles for hydrography data within a given HUC.

    Arguments:
      source    | A source object providing get_hydro()
      bounds    | [xmin, ymin, xmax, ymax] within which to gather rivers
      bounds_crs| Coordinate system of bounds (and coordinate system
                |  in which rivers will be returned)
      huc_hint  | A hint to help the source find the file containing
                |  bounds.  For NHD, this is a HUC4 or smaller.  Eventually
                |  this might be optional.
      centering | if False or None (default) does nothing.
                |  if True or 'geometric', centers based on the 
                |  geometric center.
                |  if 'mass', centers based on the center of mass.
                |  if Point object or tuple of two floats, centers 
                |  on this coordinate.
      long      | float, if a river is longer than this value it 
                | gets filtered.  Some NHD data has issues...

    Returns: (rivers, centroid)
      rivers    | A list of shapely LineString objects representing all 
                | reaches within the shape.
      centroid  | shapely point for the centroid, based on the
                |  value of centering.

    """
    logging.info("")
    logging.info("Preprocessing Hydrography")
    logging.info("-"*30)
    logging.info("loading streams in bounds {}".format(bounds))
    rprofile, rivers = source.get_hydro(bounds, bounds_crs, huc_hint)

    # convert to destination crs
    if (bounds_crs != rprofile['crs']):
        for river in rivers:
            workflow.warp.warp_shape(river, rprofile['crs'], bounds_crs)

    # round
    workflow.utils.round(rivers, workflow.conf.rcParams['digits'])

    # convert to shapely
    rivers_s = [workflow.utils.shply(r) for r in rivers]
    if merge:
        rivers_s = shapely.ops.linemerge(shapely.geometry.MultiLineString(rivers_s)).simplify(1.e-5)

    # not too long
    if long is not None:
        rivers_s = [l for l in rivers_s if l.length() < long]

    # center
    if centering:
        rivers_s, centroid = workflow.utils.center(rivers_s, centering)
    else:
        centroid = shapely.geometry.Point(0,0)
        
    return rivers_s, centroid


def get_raster_on_shape(source, shape, crs):
    """Collects a raster DEM that covers the requested shape.

    Arguments:
        source  | The source object providing get_raster()
        shape   | Shape to clip to.
        crs     | crs of the input shape.

    Returns (dem_profile, dem):
        dem_profile     | A rasterio profile file descriptor object.
        dem             | The raster, in the DEM's crs.
    """
    logging.info("")
    logging.info("Preprocessing Raster")
    logging.info("-"*30)
    logging.info("collecting raster")
    return source.get_raster(shape, crs)


def get_masked_raster_on_shape(source, shape, crs, nodata=-1):
    """Collects a raster DEM that is masked to the requested shape.

    Arguments:
        source  | The source object providing get_raster()
        shape   | Shape to clip to.
        crs     | crs of the input shape.
        nodata  | The value to place in areas not coverd by shape

    Returns (dem_profile, dem):
        dem_profile     | A rasterio profile file descriptor object.
        dem             | A raster, in lat/lon, of elevations.
    """
    logging.info("")
    logging.info("Preprocessing Raster")
    logging.info("-"*30)
    logging.info("collecting raster")

    # ensure shply
    if type(shape) is dict:
        shape = workflow.utils.shply(shape['geometry'])
    
    # get the raster
    profile, raster = source.get_raster(shape, crs)

    # warp the raster to the shape crs
    profile, raster = workflow.warp.warp_raster(profile, raster, crs)

    # mask the raster
    mask = rasterio.features.geometry_mask([shape,], raster.shape, profile['transform'], invert=True)
    masked_raster = np.where(mask, raster, nodata)

    transform = profile['transform']
    x0 = transform * (0,0)
    x1 = transform * (profile['width'], profile['height'])
    logging.info(" raster bounds = {}".format((x0[0], x0[1], x1[0], x1[1])))
    return profile, masked_raster

#
# functions for relating objects
# -----------------------------------------------------------------------------

def find_huc(source, shp, crs, hint, shrink_factor=1.e-5):
    """Finds the smallest HUC containing shp.

    Arguments:
      source    | Source object for HUCs
      shp       | A fiona or shapely polygon
      crs       | The crs of shp
      hint      | A hint for where to look, must be
                |  at least as small as the organizational
                |  file -- 2 for source WBD, 4 for NHDPlus.
                |  This eliminates the need to donwload files
                |  until we find a match.
      shrink_factor
                | A fraction of the radius of shp to shrink
                |  prior for checking containment within HUCs.
                |  This fixes cases where shp is on a HUC boundary
                |  with potentially some numerical error.

    Returns: a code for the smallest containing HUC.
    """

    def _in_huc(shply, huc_shply):
        """Checks whether shp is in HUC"""
        if huc_shply.contains(shply):
            return 2
        elif huc_shply.intersects(shply):
            return 1
        else:
            return 0

    def _find_huc(source, shply, crs, hint):
        """Searches in hint to find shp."""
        print('searching: %s'%hint)
        hint_level = len(hint)
        search_level = hint_level + 2
        if search_level > source.lowest_level:
            return hint

        profile, subhus = source.get_hucs(hint, search_level)
        
        for subhu in subhus:
            workflow.warp.warp_shape(subhu, profile['crs'], crs)
            subhu_shply = workflow.utils.shply(subhu['geometry'])        
            inhuc = _in_huc(shply, subhu_shply)

            if inhuc == 2:
                # fully contained in try_huc, recurse
                hname = subhu['properties']['HUC{:d}'.format(search_level)]
                print('  subhuc: %s contains'%hname)
                return _find_huc(source, shply, crs, hname)
            elif inhuc == 1:
                hname = subhu['properties']['HUC{:d}'.format(search_level)]
                print('  subhuc: %s partially contains'%hname)
                # partially contained in try_huc, return this
                return hint
            else:
                hname = subhu['properties']['HUC{:d}'.format(search_level)]
                print('  subhuc: %s does not contain'%hname)
        assert(False)

    if type(shp) is shapely.geometry.Polygon:
        shply = shp
    else:
        shply = workflow.utils.shply(shp['geometry'])

    # must shrink the poly a bit in case it is close to or on a boundary
    radius = np.sqrt(shply.area/np.pi)
    shply_s = shply.buffer(-shrink_factor*radius)

    hint = workflow.sources.utils.huc_str(hint)

    profile, hint_hu = source.get_huc(hint)
    workflow.warp.warp_shape(hint_hu, profile['crs'], crs)
    
    inhuc = _in_huc(shply_s, workflow.utils.shply(hint_hu['geometry']))
    if inhuc is not 2:
        raise RuntimeError("{}: shape not found in hinted HUC '{}'".format(source.name, hint))

    result = _find_huc(source, shply_s, crs, hint)
    return result



def simplify_and_prune(hucs, rivers, simplify=10, prune_reach_size=0, cut_intersections=False):
    """Cleans up the HUC and river shapes.

    Ensures intersections are proper, snapped, simplified, etc.  Note,
    HUCs and rivers must be in the same crs.

    Arguments:
      hucs      | The split-form HUC object from get_split_form_hucs()
      rivers    | The rivers object from get_rivers()
      args      | A simplify args struct.

    Simplify Args struct must include the following
      simplify  | Simplify rivers nd HUCs by this value (see 
                | shapely's simplify).  Units are length units of the CRS
      prune_reach_size  
                | Remove all rivers with fewer than this many reaches.
      cut_intersections
                | Cut HUC segments at the river input/output, potentially
                |  resulting in simpler geometries.  Work in progress.    

    Returns: the updated rivers object.
    NOTE: Modifieds the HUCs object in-place.
    """
    tol = simplify
    
    logging.info("")
    logging.info("Simplifying and pruning")
    logging.info("-"*30)
    logging.info("Filtering rivers outside of the HUC space")
    rivers = workflow.hydrography.filter_rivers_to_shape(hucs.exterior(), rivers, tol)
    if len(rivers) is 0:
        return rivers

    logging.info("Generate the river tree")
    rivers = workflow.hydrography.make_global_tree(rivers)

    logging.info("Removing rivers with fewer than {} reaches.".format(prune_reach_size))
    for i in reversed(range(len(rivers))):
        ltree = len(rivers[i])
        if ltree < prune_reach_size:
            rivers.pop(i)
            logging.info("  ...removing river with %d reaches"%ltree)
        else:
            logging.info("  ...keeping river with %d reaches"%ltree)
    if len(rivers) is 0:
        return rivers
            
    logging.info("simplifying rivers")
    workflow.hydrography.cleanup(rivers, tol, tol, tol)

    logging.info("simplifying HUCs")
    workflow.split_hucs.simplify(hucs, tol)

    # snap
    logging.info("snapping rivers and HUCs")
    rivers = workflow.hydrography.snap(hucs, rivers, tol, 3*tol, cut_intersections)
    
    logging.info("")
    logging.info("Simplification Diagnostics")
    logging.info("-"*30)
    if len(rivers) is not 0:
        mins = []
        for river in rivers:
            for line in river.dfs():
                coords = np.array(line.coords[:])
                dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
                mins.append(np.min(dz))
        logging.info("  river min seg length: %g"%min(mins))
        logging.info("  river median seg length: %g"%np.median(np.array(mins)))

    mins = []
    for line in hucs.segments:
        coords = np.array(line.coords[:])
        dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
        mins.append(np.min(dz))
    logging.info("  HUC min seg length: %g"%min(mins))
    logging.info("  HUC median seg length: %g"%np.median(np.array(mins)))
    return rivers
    
def triangulate(hucs, rivers, args, diagnostics=True):
    """Triangulates HUCs and rivers.

    Arguments:
      hucs      | The split-form HUC object from get_split_form_hucs()
      rivers    | The rivers object from get_rivers()
    
    Optional:


    Returns:
    

    """
    verbose = args.verbosity > 2
    
    logging.info("")
    logging.info("Meshing")
    logging.info("-"*30)

    refine_funcs = []
    if args.refine_max_area is not None:
        refine_funcs.append(workflow.triangulation.refine_from_max_area(args.refine_max_area))
    if args.refine_distance is not None:
        refine_funcs.append(workflow.triangulation.refine_from_river_distance(*args.refine_distance, rivers))
    if args.refine_max_edge_length is not None:
        refine_funcs.append(workflow.triangulation.refine_from_max_edge_length(args.refine_max_edge_length))
    def my_refine_func(*args):
        return any(rf(*args) for rf in refine_funcs)        

    mesh_points, mesh_tris = workflow.triangulation.triangulate(hucs, rivers,
                                                              verbose=verbose,
                                                              refinement_func=my_refine_func,
                                                              min_angle=args.refine_min_angle,
                                                              enforce_delaunay=args.enforce_delaunay)

    if diagnostics:
        logging.info("Plotting triangulation diagnostics")
        river_multiline = workflow.tree.forest_to_list(rivers)
        distances = []
        areas = []
        needs_refine = []
        for tri in mesh_tris:
            vertices = mesh_points[tri]
            bary = np.sum(np.array(vertices), axis=0)/3
            bary_p = shapely.geometry.Point(bary[0], bary[1])
            distances.append(bary_p.distance(river_multiline))
            areas.append(workflow.utils.triangle_area(vertices))
            needs_refine.append(my_refine_func(vertices, areas[-1]))

        if args.verbosity > 0:
            plt.figure()
            plt.subplot(121)
            plt.hist(distances)
            plt.xlabel("distance from river of triangle centroids [m]")
            plt.ylabel("count [-]")
            plt.subplot(122)
            plt.scatter(distances, areas,c=needs_refine,marker='x')
            plt.xlabel("distance [m]")
            plt.ylabel("triangle area [m^2]")

            # plt.figure()
            # plt.subplot(111)
            # workflow.plot.hucs(hucs)
            # workflow.plot.rivers(rivers)
            # workflow.plot.triangulation(mesh_points, mesh_tris, areas)
            # plt.title("triangle area [m^2]")
    return mesh_points, mesh_tris

def elevate(mesh_points, mesh_crs, dem, dem_profile):
    """Elevate mesh_points onto the dem."""
    logging.info("")
    logging.info("Elevating Triangulation to DEM")
    logging.info("-"*30)

    # index the i,j of the points, pick the elevations
    elev = values_from_raster(mesh_points, mesh_crs, dem, dem_profile)

    # create the 3D points
    mesh_points_3 = np.zeros((len(mesh_points),3),'d')
    mesh_points_3[:,0:2] = mesh_points
    mesh_points_3[:,2] = elev
    return mesh_points_3

def values_from_raster(points, points_crs, raster, raster_profile):
    """Take the value of the nearest pixel to each point in points."""
    points_raster_crs = np.array(workflow.warp.warp_xy(points[:,0], points[:,1], points_crs, raster_profile['crs'])).transpose()
    values = raster[rasterio.transform.rowcol(raster_profile['transform'], points_raster_crs[:,0], points_raster_crs[:,1])]
    return values

def raster_on_shapes(shapes, shapes_crs, raster, raster_profile):
    """Mask the raster so that it only appears on a given set of shapes."""
    

def color_raster_from_shapes(target_bounds, target_dx, shapes, shape_colors, shapes_crs, nodata=-1):
    assert(len(shapes) == len(shape_colors))
    assert(len(shapes) > 0)
    
    dtype = np.dtype(type(shape_colors[0]))
    
    target_x0 = np.round(target_bounds[0] - target_dx/2)
    target_y1 = np.round(target_bounds[3] + target_dx/2)
    width = int(np.ceil((target_bounds[2] + target_dx/2 - target_x0)/target_dx))
    height = int(np.ceil((target_y1 - target_bounds[1] - target_dx/2)/target_dx))

    img_bounds = [target_x0, target_y1 - target_dx*height, target_x0 + target_dx*width, target_y1]

    logging.info('Coloring shapes onto raster:')
    logging.info('  target_bounds = {}'.format(target_bounds))
    logging.info('  img_bounds = {}'.format(img_bounds))
    logging.info('  pixel_size = {}'.format(target_dx))
    logging.info('  width = {}, height = {}'.format(width, height))
    logging.info('  and {} independent colors of dtype {}'.format(len(set(shape_colors)), dtype))

    transform = rasterio.transform.from_origin(target_x0, target_y1, target_dx, target_dx)
    
    raster_profile = {'height':height,
                      'width':width,
                      'count':1,
                      'dtype':dtype,
                      'crs':shapes_crs,
                      'transform':transform,
                      'nodata':nodata}
    
    z = nodata * np.ones((width, height), dtype)
    for p, p_id in zip(shapes, shape_colors):
        mask = rasterio.features.geometry_mask([p,], z.shape, transform, invert=True)
        z[mask] = p_id
    return z, raster_profile, img_bounds



    
