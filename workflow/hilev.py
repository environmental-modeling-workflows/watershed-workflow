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
import shapely
import meshpy.triangle

import workflow.conf
import workflow.triangulation
import workflow.warp
import workflow.plot
import workflow.tree
import workflow.split_hucs
import workflow.hydrography
import workflow.clip
import workflow.rowcol
import workflow.sources.utils 
from workflow.sources.utils import huc_str


import vtk_io # from ATS/tools/meshing_ats

def get_hu(source, huc, crs=None, centering=None):
    """Download and read a HUC file.

    Arguments:
      source    | source object providing get_huc()
      huc       | hydrologic unit code
      crs       | provides the output coordinate system, 
                | defaults to workflow.conf.default_crs()
      centering | if False or None (default) does nothing.
                |  if True or 'geometric', centers based on the 
                |  geometric center.
                |  if 'mass', centers based on the center of mass.
                |  if Point object or tuple of two floats, centers 
                |  on this coordinate.

    Returns: (hu, centroid) 
      hu        | shapely polygon for the hydrologic unit
      centroid  | shapely point for the centroid, based on the
                |  value of centering.
    """
    huc = huc_str(huc)
    hu_shapes, centroid = get_hus(source, huc, len(huc), crs, centering)
    assert(len(hu_shapes) == 1)
    return h_shapes[0], centroid

def get_hus(source, huc, level, crs=None, centering=None):
    """Download and read a HUC file.

    Arguments:
      source    | source object providing get_huc()
      huc       | hydrologic unit code
      level     | level of the requested sub-units
      crs       | provides the output coordinate system, 
                | defaults to workflow.conf.default_crs()
      centering | if False or None (default) does nothing.
                |  if True or 'geometric', centers based on the 
                |  geometric center.
                |  if 'mass', centers based on the center of mass.
                |  if Point object or tuple of two floats, centers 
                |  on this coordinate.

    Returns: (hus, centroid)
      hus       | list of shapely polygons for the hydrologic units
                |  at level within huc
      centroid  | shapely point for the centroid, based on the
                |  value of centering
    """
    # get the hu from source
    huc = huc_str(huc)
    if level is None:
        level = len(huc)

    logging.info("Loading level {} HUCs in {}.".format(level, huc))
    
    profile, hus = source.get_hucs(huc, level)

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


def get_split_form_hucs(source, myhuc, level=None, crs=None, centering=False):
    """Loads HUCs from a source.

    Arguments:
      source   | The source object, see workflow.sources
      myhuc    | a string for the code of the requested HUC.
      level    | additionally provide subhucs of this level
      centering| if False or None (default) does nothing.
               |  if True or 'geometric', centers based on the 
               |  geometric center.
               |  if 'mass', centers based on the center of mass.
               |  if Point object or tuple of two floats, centers 
               |  on this coordinate.
      crs      | Output coordinate system.  Defaults to
               |  workflow.conf.default_crs()

    Returns: (split_form_hucs, centroid)
      split_form_hucs | the hucs in tiled form for geometric manipulation
      centroid        | shapely point for the centroid, based on the
                      |  value of centering.

    """
    hu_shapes, centroid = get_hus(source, myhuc, level, crs, centering)
    return workflow.split_hucs.SplitHUCs(hu_shapes), centroid
        
def get_rivers_by_bounds(source, bounds, bounds_crs, huc_hint, centering=None, long=None):
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
    # load stream network
    logging.info("loading streams in bounds {}".format(bounds))
    rprofile, rivers = source.get_hydro(bounds, bounds_crs, huc_hint)

    # convert to destination crs
    if (bounds_crs != rprofile['crs']):
        for river in rivers:
            workflow.warp.warp_shape(river, rprofile['crs'], bounds_crs)

    # round
    workflow.utils.round(rivers, workflow.conf.rcParams['digits'])

    # convert to shapely
    logging.info("merging reaches")
    rivers_s = shapely.geometry.MultiLineString([workflow.utils.shply(r['geometry']) for r in rivers])
    rivers_s = shapely.ops.linemerge(rivers_s).simplify(1.e-5)

    # not too long
    if long is not None:
        rivers_s = [l for l in rivers_s if l.length() < long]

    # center
    if centering:
        rivers_s, centroid = workflow.utils.center(rivers_s, centering)
    else:
        centroid = shapely.geometry.Point(0,0)
        
    return shapely.geometry.MultiLineString(rivers_s), centroid

def get_raster_on_huc(shape, source_dem):
    """Collects a raster DEM that covers the requested HUC.

    Arguments:
        huc     | The fiona shapefile of the HUC.  Output from 
                | get_hucs().

    Returns (dem_profile, dem):
        dem_profile     | A rasterio profile file descriptor object.
        dem             | A raster, in lat/lon, of elevations.
    """
    logging.info("")
    logging.info("Preprocessing DEM")
    logging.info("==========================")
    # load shapefiles for the HUC of interest
    logging.info("loading HUC %s"%myhuc)
    profile, huc = sources['HUC'].load_huc(myhuc)
    assert(profile['crs']['init'] == 'epsg:4269') # latlong

    dem = sources['DEM'].load_dem(huc)
    dem_profile, dem = workflow.clip.clip_dem(dem, huc)
    dem = dem[0,:,:] # only the first band
    return dem_profile, dem






def find_huc(source, shp, crs, hint, shrink=1.e-5):
    """Finds the smallest HUC containing shp, starting with a potential
    hint, i.e. '06' for Tennessee River Valley.
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
    shply_s = shply.buffer(-shrink*radius)

    hint = workflow.sources.utils.huc_str(hint)

    profile, hint_hu = source.get_huc(hint)
    workflow.warp.warp_shape(hint_hu, profile['crs'], crs)
    
    inhuc = _in_huc(shply_s, workflow.utils.shply(hint_hu['geometry']))
    if inhuc is not 2:
        raise RuntimeError("{}: shape not found in hinted HUC '{}'".format(source.name, hint))

    result = _find_huc(source, shply_s, crs, hint)
    return result


def get_dem_on_shape(source, shape, crs):
    """Collects a raster DEM that covers the requested shape.

    Arguments:
        profile | The fiona profile from the shape.
                | Used to check the CRS
        shape   | Shape to clip to.
        sources | Source dictionary.

    Returns (dem_profile, dem):
        dem_profile     | A rasterio profile file descriptor object.
        dem             | A raster, in lat/lon, of elevations.

    NOTE: this WILL warp shape to the DEM's CRS!
    """
    logging.info("")
    logging.info("Preprocessing DEM")
    logging.info("==========================")
    logging.info("downloading DEM")
    return source.get_dem(shape, crs)


def get_shapes(filename, index, center=True, make_hucs=True):
    """Collects shapefiles.

    Arguments:
        filename| File to parse, should end in .shp
        index   | Index of the requested shape in filename, or -1 to get all.
        center  | If true, subtract off the centroid.

    Returns (profile, sheds, boundary, centroid)
        profile | the fiona profile/projection/etc for the shapefile
                | Note this includes original projection.
        sheds   | a workflow.split_hucs.SplitHUCs object for all watershed shapes requested, 
                | in the default coordinate system.
        boundary| The boundary of the union of watersheds, in lat-lon
        centroid| The centroid of the watersheds requested, for use in uncentering.
    """
    logging.info("")
    logging.info("Preprocessing Shapes")
    logging.info("=====================")

    # load shapefile
    logging.info("loading file: %s"%filename)
    with fiona.open(filename, 'r') as fid:
        profile = fid.profile
        if index < 0:
            shps = [s for s in fid]
        else:
            shps = [fid[index],]

    # convert the original coordinate system to lat-lon to get a lat-lon boundary
    if profile['crs']['init'] != 'epsg:4269':
        for shp in shps:
            workflow.warp.warp_shape(shp, profile['crs'], workflow.conf.latlon_crs())
        profile['crs']['init'] = 'epsg:4269'
            
    # convert original coordinate system to shapely
    huc_shapes = [workflow.utils.shply(s['geometry']) for s in shps]
    boundary = shapely.ops.cascaded_union(huc_shapes)
            
    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for shp in shps:
        workflow.warp.warp_shape(shp, profile['crs'], workflow.conf.default_crs())

    # convert to shapely
    huc_shapes = [workflow.utils.shply(s['geometry']) for s in shps]

    # center the HUCs
    if center:
        huc_shapes, centroid = workflow.utils.center(huc_shapes, center)
        logging.info("centering %d shapes to (%g,%g)"%(len(huc_shapes), centroid.xy[0][0], centroid.xy[1][0]))
    else:
        centroid = shapely.geometry.Point(0,0)

    # split
    logging.info("Split form subwatersheds")
    if make_hucs:
        hucs = workflow.split_hucs.SplitHUCs(huc_shapes)
    else:
        hucs = huc_shapes
    logging.info("...done")
    return profile, hucs, boundary, centroid
    

def simplify_and_prune(hucs, rivers, args):
    """Cleans up the HUC and river shapes, making sure intersections are
    proper, snapped, simplified, etc.

    """
    tol = args.simplify
    
    logging.info("")
    logging.info("Simplifying and pruning")
    logging.info("========================")
    logging.info("filtering rivers outside of the HUC space")
    rivers = workflow.hydrography.filter_rivers_to_huc(hucs, rivers, tol)
    if len(rivers) is 0:
        return rivers

    logging.info("removing rivers with only a few reaches")
    for i in reversed(range(len(rivers))):
        ltree = len(rivers[i])
        if ltree < args.prune_reach_size:
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
    rivers = workflow.hydrography.snap(hucs, rivers, tol, 3*tol, args.cut_intersections)
    
    logging.info("...done")

    logging.info("Resulting info")
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
    verbose = args.verbosity > 2
    
    logging.info("")
    logging.info("Meshing")
    logging.info("===============")

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
                                                              enforce_delaunay=args.delaunay)

    if diagnostics:
        logging.info("triangulation diagnostics")
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

def elevate(mesh_points, dem, dem_profile):
    # -- must map back to lat/lon to take from dem
    logging.info("elevating")
    triangles_3d = []
    mesh_points_ll = np.array(workflow.warp.warp_xy(mesh_points[:,0], mesh_points[:,1], workflow.conf.default_crs(), workflow.conf.latlon_crs())).transpose()
    elev = dem[workflow.rowcol.rowcol(dem_profile['transform'], mesh_points_ll[:,0], mesh_points_ll[:,1])]
    mesh_points_3 = np.zeros((len(mesh_points),3),'d')
    mesh_points_3[:,0:2] = mesh_points
    mesh_points_3[:,2] = elev
    return mesh_points_3
    
def save(filename, points3, tris, metadata):
    """Save as a VTK mesh. 

    This could be Exodus, but meshing_ats is in python2 (and uses exodus which is in python2)
    """
    logging.info("saving mesh: %s"%filename)
    cells = {'triangle':tris}
    vtk_io.write(filename, points3, cells)
    with open(filename+'.readme','w') as fid:
        fid.write(metadata)

    
