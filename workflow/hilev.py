import sys,os
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.collections as pltc
import logging

import shapely
import meshpy.triangle

import workflow.conf
import workflow.download
import workflow.triangulate
import workflow.warp
import workflow.plot
import workflow.tree
import workflow.hucs
import workflow.hydrography
import workflow.clip
import workflow.rowcol

import vtk_io # from ATS/tools/meshing_ats

def get_hucs(myhuc, center=True):
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
    logging.info("downloading HUC %s"%myhuc[0:2])
    workflow.download.download_huc(myhuc[0:2])

    # load shapefiles for the HUC of interest
    logging.info("loading HUC %s"%myhuc)
    profile, huc = workflow.conf.load_huc(myhuc)

    # load shapefiles for all HUC 12s
    logging.info("loading all 12s")
    profile, huc12s = workflow.conf.load_hucs_in(myhuc, 12)

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for huc12 in huc12s:
        workflow.warp.warp_shape(huc12, profile['crs'], workflow.conf.default_crs())

    # convert to shapely
    huc_shapes = [shapely.geometry.shape(s['geometry']) for s in huc12s]

    # center the HUCs
    if center:
        huc_shapes, centroid = workflow.utils.center(huc_shapes)
    else:
        centroid = shapely.geometry.Point(0,0)

    # split
    logging.info("Split form HUCs")
    hucs = workflow.hucs.HUCs(huc_shapes)
    logging.info("...done")
    return huc, hucs, centroid

def get_rivers(myhuc):
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
    logging.info("downloading Hydrography %s"%myhuc)
    workflow.download.download_hydro(myhuc)

    # load stream network
    logging.info("loading streams")
    rprofile, rivers = workflow.conf.load_hydro(myhuc)

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for river in rivers:
        workflow.warp.warp_shape(river, rprofile['crs'], workflow.conf.default_crs())

    # convert to shapely
    logging.info("merging reaches")
    rivers_s = shapely.geometry.MultiLineString([shapely.geometry.shape(r['geometry']) for r in rivers])
    rivers_s2 = shapely.ops.linemerge(rivers_s).simplify(1.e-5)
    return rivers_s2

def get_dem(huc):
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
    dem_profile, dem = workflow.clip.clip_dem(huc)
    dem = dem[0,:,:] # only the first band
    return dem_profile, dem

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

    logging.info("removing rivers with only a few reaches")
    for i in reversed(range(len(rivers))):
        ltree = len(rivers[i])
        if ltree < 10:
            rivers.pop(i)
            logging.info("  removing river with %d reaches"%ltree)
        else:
            logging.info("  keeping river with %d reaches"%ltree)

    logging.info("simplifying rivers")
    workflow.hydrography.cleanup(rivers, tol, tol, tol)
    logging.info("simplify HUCs")
    workflow.hucs.simplify(hucs, tol)

    # snap
    logging.info("snapping rivers and HUCs")
    workflow.hydrography.snap(hucs, rivers, tol, 10*tol)

    logging.info("filtering cut reaches outside the HUC space")
    rivers = workflow.hydrography.filter_rivers_to_huc(hucs, rivers, -0.1*tol)
    logging.info("...done")

    logging.info("Resulting info")
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
    
def triangulate(hucs, rivers, args):
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
    
def save(filename, points3, tris):
    """Save as a VTK mesh. 

    This could be Exodus, but meshing_ats is in python2 (and uses exodus which is in python2)
    """
    logging.info("saving mesh")
    cells = {'triangle':tris}
    vtk_io.write(filename, points3, cells)

    
