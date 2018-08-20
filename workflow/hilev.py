"""High-level routines, espeically those that interact with data,
i.e. call download_*() or load_*() functions from conf.

Most scripts use these functions instead of directly using lower-level
capability.

"""

import sys,os
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.collections as pltc
import logging

import fiona
import shapely
import meshpy.triangle

import workflow.conf
import workflow.triangulate
import workflow.warp
import workflow.plot
import workflow.tree
import workflow.hucs
import workflow.hydrography
import workflow.clip
import workflow.rowcol

import vtk_io # from ATS/tools/meshing_ats

def _in_huc(shp, hucstr, source):
    """Checks whether shp is in HUC"""
    logging.debug("Checking: shp in '%s'?"%hucstr)

    try:
        fname = source.download(hucstr)
        profile, huc = source.load_huc(hucstr)
    except RuntimeError as err:
        logging.debug("No such HUC %s found? %s"%(hucstr,str(err)))
        raise err

    if profile['crs']['init'] != 'epsg:4269':
        # latlong
        raise RuntimeError("HUC file for '%s' not in Lat-Lon?"%hucstr)

    huc_shp = shapely.geometry.shape(huc['geometry'])
    logging.debug(" shp bounds = %r"%list(shp.bounds))
    logging.debug(" huc bounds = %r"%list(huc_shp.bounds))
    if huc_shp.contains(shp):
        logging.debug('  yes!')
        return 2
    elif huc_shp.intersects(shp):
        logging.debug('  sorta!')
        return 1
    else:
        logging.debug('  no!')
        return 0

def _find_huc(shp, hint, source):
    for i in range(0,100):
        try_huc = hint+'%02i'%i
        try:
            inhuc = _in_huc(shp, try_huc, source)
        except RuntimeError:
            if try_huc.endswith('00'):
                # some huc levels have 00, some don't?
                continue
            else:
                return -1

        if inhuc == 2:
            # fully contained in try_huc, recurse if not HUC12
            if len(try_huc) == 12:
                return try_huc
            else:
                return _find_huc(shp, try_huc, source)
        elif inhuc == 1:
            # partially contained in try_huc, return this
            return hint
    return -1


def find_huc(shp_profile, shply, source, hint=None):
    """Finds the smallest HUC containing shp, starting with a potential
    hint, i.e. '06' for Tennessee River Valley.

    Expects shp in lat-lon, i.e. epsg:4269
    """
    if shp_profile['crs']['init'] != 'epsg:4269':
        # latlong
        raise RuntimeError("shapefile not in Lat-Lon")

    shply = shply.buffer(-.001)

    if hint is None:
        hint = ''
    if len(hint) is 12:
        inhuc = _in_huc(shply, hint, source)
        if inhuc is not 2:
            raise RuntimeError("Shape not found in hinted HUC '%s'"%hint)
        return hint
        
    result = _find_huc(shply, hint, source)
    if type(result) is not str:
        raise RuntimeError("Shape not found in hinted HUC '%s'"%hint)
    return result

def get_hucs(myhuc, source, level=12, center=True):
    """Collects shapefiles for HUCs given a HUC code in string form.

    Arguments:
        myhuc   | a length N string for the number of the requested HUC.
                | Note this must be an even number of digits, i.e. 01, not 1.
        center  | If true, subtract off the HUC centroid.

    Returns (huc,huc12s,centroid):
        huc     | the fiona shape representation of the requested HUC
        huc12s  | a workflow.hucs.HUCs object for shapely shapes of all HUC 
                | 12s in myhuc
        centroid| The centroid of the HUC requested, for use in uncentering.
    """
    ## === Preprocess HUCs ===
    logging.info("")
    logging.info("Preprocessing HUCs")
    logging.info("=====================")

    # collect HUC shapefile
    logging.info("collecting HUC %s"%myhuc[0:2])
    source.download(myhuc)

    # load shapefiles for all HUC 12s
    logging.info("loading all %is"%level)
    profile, huc12s = source.load_hucs_in(myhuc, level)

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for huc12 in huc12s:
        workflow.warp.warp_shape(huc12, profile['crs'], workflow.conf.default_crs())

    # convert to shapely
    huc_shapes = [shapely.geometry.shape(s['geometry']) for s in huc12s]

    # if multi-poly, make sure we can convert to single-poly
    single_huc_shapes = []
    for huc_shp in huc_shapes:
        if type(huc_shp) is not shapely.geometry.Polygon:
            assert(len(huc_shp) is 1)
            huc_shp = huc_shp[0]
            assert(type(huc_shp) is shapely.geometry.Polygon)
        single_huc_shapes.append(huc_shp)
    huc_shapes = single_huc_shapes

    # center the HUCs
    if center:
        huc_shapes, centroid = workflow.utils.center(huc_shapes)
    else:
        centroid = shapely.geometry.Point(0,0)

    # split
    logging.info("Split form HUCs")
    hucs = workflow.hucs.HUCs(huc_shapes)
    logging.info("...done")
    return hucs, centroid

def get_rivers(myhuc, source):
    """Collects shapefiles for hydrography data within a given HUC.

    Arguments:
        myhuc   | a length N string for the number of the requested HUC.
                | Note this must be an even number of digits, i.e. 01, not 1.

    Returns:
        rivers  | A list of shapely LineString objects representing all 
                | reaches within the HUC.
    """
    ## === Preprocess hydrography ===
    logging.info("")
    logging.info("Preprocessing hydrography")
    logging.info("==========================")

    # collect hydrography
    logging.info("collecting Hydrography %s"%myhuc)
    source.download(myhuc)

    # load the HUC and get a bounding box
    profile, huc = source.load_huc(myhuc)
    bounds = shapely.geometry.shape(huc['geometry']).bounds
    
    # load stream network
    logging.info("loading streams")
    rprofile, rivers = source.load_hydro(myhuc, bounds)

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for river in rivers:
        workflow.warp.warp_shape(river, rprofile['crs'], workflow.conf.default_crs())

    # convert to shapely
    logging.info("merging reaches")
    rivers_s = shapely.geometry.MultiLineString([shapely.geometry.shape(r['geometry']) for r in rivers])
    rivers_s2 = shapely.ops.linemerge(rivers_s).simplify(1.e-5)
    return rivers_s2

def get_dem(myhuc, sources):
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
    logging.info("downloading DEM")

    # load shapefiles for the HUC of interest
    logging.info("loading HUC %s"%myhuc)
    profile, huc = sources['HUC'].load_huc(myhuc)
    assert(profile['crs']['init'] == 'epsg:4269') # latlong

    dem_profile, dem = workflow.clip.clip_dem(huc, sources['DEM'])
    dem = dem[0,:,:] # only the first band
    return dem_profile, dem


def get_shapes(filename, index, center=True, make_hucs=True):
    """Collects shapefiles.

    Arguments:
        filename| File to parse, should end in .shp
        index   | Index of the requested shape in filename, or -1 to get all.
        center  | If true, subtract off the centroid.

    Returns (profile, sheds, boundary, centroid)
        profile | the fiona profile/projection/etc for the shapefile
                | Note this includes original projection.
        sheds   | a workflow.hucs.HUCs object for all watershed shapes requested, 
                | in the default coordinate system.
        boundary| The boundary of the union of watersheds, in the original
                |  coordinate system.
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
    huc_shapes = [shapely.geometry.shape(s['geometry']) for s in shps]
    boundary = shapely.ops.cascaded_union(huc_shapes)
            
    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for shp in shps:
        workflow.warp.warp_shape(shp, profile['crs'], workflow.conf.default_crs())

    # convert to shapely
    huc_shapes = [shapely.geometry.shape(s['geometry']) for s in shps]

    # center the HUCs
    if center:
        huc_shapes, centroid = workflow.utils.center(huc_shapes)
    else:
        centroid = shapely.geometry.Point(0,0)

    # split
    logging.info("Split form subwatersheds")
    if make_hucs:
        hucs = workflow.hucs.HUCs(huc_shapes)
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
            logging.info("  removing river with %d reaches"%ltree)
        else:
            logging.info("  keeping river with %d reaches"%ltree)
    if len(rivers) is 0:
        return rivers
            
    logging.info("simplifying rivers")
    workflow.hydrography.cleanup(rivers, tol, tol, tol)

    logging.info("simplify HUCs")
    workflow.hucs.simplify(hucs, tol)

    # snap
    logging.info("snapping rivers and HUCs")
    rivers = workflow.hydrography.snap(hucs, rivers, tol, 10*tol, args.cut_intersections)
    
    logging.info("filtering cut reaches outside the HUC space")
    rivers = workflow.hydrography.filter_rivers_to_huc(hucs, rivers, -0.1*tol)
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
    if args.refine_max_area is not None:
        refine_func = workflow.triangulate.refine_from_max_area(args.refine_max_area)
    elif args.refine_distance is not None:
        refine_func = workflow.triangulate.refine_from_river_distance(*args.refine_distance, rivers)
    else:
        def refine_func(*args, **kwargs):
            return False

    mesh_points, mesh_tris = workflow.triangulate.triangulate(hucs, rivers, verbose=verbose,
                                                              refinement_func=refine_func)

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
            needs_refine.append(refine_func(vertices, areas[-1]))

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
    return mesh_points, mesh_tris

def elevate(mesh_points, dem, dem_profile):
    # -- must map back to lat/lon to take from dem
    logging.info("elevating")
    triangles_3d = []
    mesh_points_ll = np.array(workflow.warp.warp_xy(mesh_points[:,0], mesh_points[:,1], workflow.conf.default_crs(), workflow.conf.latlon_crs())).transpose()
    elev = dem[workflow.rowcol.rowcol(dem_profile['affine'], mesh_points_ll[:,0], mesh_points_ll[:,1])]
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

    
