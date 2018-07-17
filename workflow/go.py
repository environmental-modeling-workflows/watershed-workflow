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
    ## === Preprocess HUCs ===
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
    profile, huc12s = workflow.conf.load_hucs_in(my_huc, 12)

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
    ## === Preprocess hydrography ===
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
    logging.info("Digital Elevation Model")
    logging.info("==========================")
    logging.info("downloading DEM")
    dem_profile, dem = workflow.clip.clip_dem(huc)
    dem = dem[0,:,:] # only the first band
    return dem_profile, dem

def simplify_and_prune(hucs, rivers, tol):
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
    return rivers
    
def info(hucs, rivers):
    logging.info("Resulting info")
    logging.info("========================")
    mins = []
    for river in rivers:
        for line in river.dfs():
            coords = np.array(line.coords[:])
            dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
            mins.append(np.min(dz))
    logging.info("river min seg length: %g"%min(mins))
    logging.info("river median seg length: %g"%np.median(np.array(mins)))

    mins = []
    for line in hucs.segments:
        coords = np.array(line.coords[:])
        dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
        mins.append(np.min(dz))
    logging.info("HUC min seg length: %g"%min(mins))
    logging.info("HUC median seg length: %g"%np.median(np.array(mins)))  


    
def triangulate(hucs, rivers, verbose=False):
    logging.info("Triangulating")
    logging.info("===============")
    refine_func = workflow.triangulate.refine_from_river_distance(0, 1000, 1000, 10000, rivers)
    #refine_func = workflow.triangulate.refine_from_max_area(3000.)
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


if __name__ == '__main__':
    # deal with logging to stdout
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    simplify_distance = 10 # meters
    my_huc = sys.argv[-1]
    if my_huc.endswith(".py"): # no arg
        my_huc = '060102080101'

    # collect data
    containing_huc, hucs, centroid = get_hucs(my_huc, True)
    rivers = get_rivers(my_huc)
    dem_profile, dem = get_dem(containing_huc)

    # make 2D mesh
    rivers = [shapely.affinity.translate(r, -centroid.coords[0][0], -centroid.coords[0][1]) for r in rivers]
    rivers = simplify_and_prune(hucs, rivers, simplify_distance)
    info(hucs, rivers)
    mesh_points2, mesh_tris = triangulate(hucs,rivers)

    # elevate to 3D
    mesh_points2_uncentered = mesh_points2 + np.expand_dims(np.array(centroid.coords[0]),0)
    mesh_points3_uncentered = elevate(mesh_points2_uncentered, dem, dem_profile)
    mesh_points3 = np.empty(mesh_points3_uncentered.shape,'d')
    mesh_points3[:,0:2] = mesh_points2
    mesh_points3[:,2] = mesh_points3_uncentered[:,2]

    # plot the result
    plt.figure()
    workflow.plot.triangulation(mesh_points3, mesh_tris, linewidth=0.5)
    workflow.plot.hucs(hucs, 'k')
    workflow.plot.rivers(rivers)
    plt.gca().set_aspect('equal', 'datalim')    
    plt.show()


    # save mesh
    outdir = "data/meshes"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fileout = os.path.join(outdir, 'huc_%s.vtk'%my_huc)
    save(fileout, mesh_points3_uncentered, mesh_tris)
    
