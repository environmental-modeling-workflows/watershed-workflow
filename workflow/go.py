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

#import vtk_io # from ATS/tools/meshing_ats

def get_hucs(myhuc):
    ## === Preprocess HUCs ===
    logging.info("Preprocessing HUCs")
    logging.info("=====================")

    # collect HUC shapefile
    logging.info("downloading HUC %s"%myhuc[0:2])
    workflow.download.download_huc(myhuc[0:2])

    # load shapefiles for the HUC8 of interest
    logging.info("loading HUC %s"%myhuc)
    profile, huc = workflow.conf.load_huc(myhuc)

    # load shapefiles for all HUC 12s in the Obed HUC 8.
    logging.info("loading all 12s")
    profile, huc12s = workflow.conf.load_hucs_in(my_huc, 12)

    # change coordinates to meters (in place)
    logging.info("change coordinates to m")
    for huc12 in huc12s:
        workflow.warp.warp_shape(huc12, profile['crs'], workflow.conf.default_crs())

    # convert to shapely
    huc_shapes = [shapely.geometry.shape(s['geometry']) for s in huc12s]

    # split
    logging.info("Split form HUCs")
    hucs = workflow.hucs.HUCs(huc_shapes)
    logging.info("...done")
    return hucs

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
    refine_func = workflow.triangulate.refine_from_river_distance(30, 3, 500, 100, rivers)
    #refine_func = workflow.triangulate.refine_from_max_area(1.e6)
    mesh_points, mesh_tris = workflow.triangulate.triangulate(hucs, rivers, verbose=verbose,
                                                              refinement_func=refine_func)
    return mesh_points, mesh_tris

    

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
    
    # outdir = "data/meshes/%s/12"%my_huc
    # if not os.path.isdir(outdir):
    #     os.makedirs(outdir)

    hucs = get_hucs(my_huc)
    rivers = get_rivers(my_huc)
    rivers = simplify_and_prune(hucs, rivers, simplify_distance)
    info(hucs, rivers)

    points, tris = triangulate(hucs,rivers,)

    plt.figure()
    workflow.plot.triangulation(points, tris, linewidth=0.5)
    workflow.plot.hucs(hucs, 'k')
    workflow.plot.rivers(rivers)
    plt.show()
    
    
