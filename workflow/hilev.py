"""High-level routines, espeically those that interact with data,
i.e. call download_*() or load_*() functions from conf.

Most scripts use these functions instead of directly using lower-level
capability.

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
import workflow.triangulate
import workflow.warp
import workflow.plot
import workflow.tree
import workflow.split_hucs
import workflow.hydrography
import workflow.clip
import workflow.rowcol

import vtk_io # from ATS/tools/meshing_ats

def get_hucs(source, myhuc, level=None, crs=None):
    """Loads HUCs from a source.

    Arguments:
        source  | The source object, see workflow.sources
        myhuc   | a length N string for the number of the requested HUC.
                |  Note this must be an even number of digits, i.e. 01, not 1.
        crs     | Destination coordinate reference system, e.g. 'epsg:5070',
                |  defaults to workflow.conf.default_crs()

    Returns: (profile, huc, hucs_at_level)
        profile       | The fiona profile for the shape.
        huc           | The fiona shape representation of the requested HUC
        hucs_at_level | A list of all 
    """
    ## === Preprocess HUCs ===
    logging.info("")
    logging.info("Preprocessing HUCs")
    logging.info("=====================")

    # load shapefiles for all HUC of the given level
    logging.info("loading all %is"%level)
    profile, huc12s = source.load_hucs_in(myhuc, level)
    for huc in huc12s:
        logging.info("  found: %s"%huc['properties']['HUC12'])        

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    if crs is None:
        crs = workflow.conf.default_crs()
    for huc12 in huc12s:
        workflow.warp.warp_shape(huc12, profile['crs'], crs)

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
    hucs = workflow.split_hucs.SplitHUCs(huc_shapes)
    logging.info("...done")
    return hucs, centroid

def get_rivers(myhuc, source, filter_long=None):
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

    # load the HUC and get a bounding box
    profile, huc = source.load_huc(myhuc)
    bounds = shapely.geometry.shape(huc['geometry']).bounds
    
    # load stream network
    logging.info("loading streams")
    rprofile, rivers = source.load_hydro(myhuc, bounds)

    # some strange long segments show up in some strange cases?
    if filter_long is not None:
        rivers = [r for r in rivers if r['properties']['LengthKM'] < filter_long]

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for river in rivers:
        workflow.warp.warp_shape(river, rprofile['crs'], workflow.conf.default_crs())

    # convert to shapely
    logging.info("merging reaches")
    rivers_s = shapely.geometry.MultiLineString([shapely.geometry.shape(r['geometry']) for r in rivers])
    rivers_s2 = shapely.ops.linemerge(rivers_s).simplify(1.e-5)
    return rivers_s2

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























def get_dem_on_shape(profile, shape, sources):
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

    dem_profile, dem = workflow.clip.clip_dem(shape, sources['DEM'])
    if dem_profile['crs']['init'] != profile['crs']['init']:
        workflow.warp.warp_shape(shape, profile['crs'], dem_profile['crs'])
        profile['crs'] = dem_profile['crs']

        

        
    
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
            logging.info("  removing river with %d reaches"%ltree)
        else:
            logging.info("  keeping river with %d reaches"%ltree)
    if len(rivers) is 0:
        return rivers
            
    logging.info("simplifying rivers")
    workflow.hydrography.cleanup(rivers, tol, tol, tol)

    logging.info("simplify HUCs")
    workflow.split_hucs.simplify(hucs, tol)

    # snap
    logging.info("snapping rivers and HUCs")
    rivers = workflow.hydrography.snap(hucs, rivers, tol, 3*tol, args.cut_intersections)
    
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

    refine_funcs = []
    if args.refine_max_area is not None:
        refine_funcs.append(workflow.triangulate.refine_from_max_area(args.refine_max_area))
    if args.refine_distance is not None:
        refine_funcs.append(workflow.triangulate.refine_from_river_distance(*args.refine_distance, rivers))
    if args.refine_max_edge_length is not None:
        refine_funcs.append(workflow.triangulate.refine_from_max_edge_length(args.refine_max_edge_length))
    def my_refine_func(*args):
        return any(rf(*args) for rf in refine_funcs)        

    mesh_points, mesh_tris = workflow.triangulate.triangulate(hucs, rivers,
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

    
