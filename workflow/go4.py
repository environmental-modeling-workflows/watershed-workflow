import sys,os
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import logging

import shapely
import meshpy.triangle

import workflow.conf
import workflow.smooth
import workflow.download
import workflow.triangulate
import workflow.clip
import workflow.warp
import workflow.rowcol
import workflow.colors
import workflow.plot
import workflow.tree
import workflow.splitpoly

import vtk_io # from ATS/tools/meshing_ats


simplify_distance = 10 # meters

myhuc = '06010208'
outdir = "data/meshes/%s/12"%myhuc
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# deal with logging to stdout
root = logging.getLogger()
root.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
    
## === Preprocess HUCs ===
logging.info("Preprocessing HUCs")
logging.info("=====================")

# collect HUC shapefile
logging.info("downloading HUC %s"%myhuc[0:2])
workflow.download.download_huc(myhuc[0:2])

# load shapefiles for the HUC8 of interest
logging.info("loading HUC %s"%myhuc)
profile, huc8 = workflow.conf.load_huc(myhuc)

# load shapefiles for all HUC 12s in the Obed HUC 8.
logging.info("loading all 12s")
profile, huc12s = workflow.conf.load_hucs_in('06010208', 12)

# change coordinates to meters (in place)
logging.info("change coordinates to m")
for huc12 in huc12s:
    workflow.warp.warp_shape(huc12, profile['crs'], workflow.conf.default_crs())

# convert to shapely
hucs_s = [shapely.geometry.shape(s['geometry']) for s in huc12s]

## === Preprocess hydrography ===
logging.info("Preprocessing hydrography")
logging.info("==========================")

# collect hydrography
logging.info("downloading Hydrography %s"%myhuc)
workflow.download.download_hydro(myhuc)

# load stream network
logging.info("loading streams")
rprofile, rivers = workflow.conf.load_hydro('06010208')

# change coordinates to meters (in place)
logging.info("change coordinates to m")
for river in rivers:
    workflow.warp.warp_shape(river, rprofile['crs'], workflow.conf.default_crs())

# convert to shapely
rivers_s = shapely.geometry.MultiLineString([shapely.geometry.shape(r['geometry']) for r in rivers])

## === process unsimplified data to get all intersections, topology ===
logging.info("Converting topologies")
logging.info("=======================")

# convert hucs to split form
logging.info("Split form HUCs")
hucs_s = workflow.hucs.HUCs(hucs_s)

# merge reaches where there are no intersections
logging.info("merging reaches")
rivers_s = shapely.ops.linemerge(rivers_s)

# cut and bin rivers into by-huc bins
logging.info("cut-and-bin reaches")
bins = workflow.hydrography.cut_and_bin(hucs_s, rivers_s)
del rivers_s

# convert each bin into forests of river network trees
logging.info("rivers to forests of river network trees")
forests = [workflow.tree.make_trees(bin) for bin in bins]

# prune short leaf reaches, simplify
logging.info("prune and simplify rivers")
for forest in forests:
    for tree in forest:
        workflow.hydrography.prune(tree, simplify_distance)
        workflow.hydrography.simplify(tree, simplify_distance)

# simplify the hucs
logging.info("simplify hucs")
workflow.hucs.simplify(hucs_s, simplify_distance)


# check min distances
logging.info("Min distance:")
min_seg = 1.e10
min_len = 1.e10
for seg in hucs_s.segments:
    min_len = min(min_len, seg.length)
    coords = np.array(seg.coords)
    assert(coords.shape[1] == 2)
    l2 = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    assert(len(l2) == len(coords)-1)
    assert(l2.min() > 1.e-3)
    min_seg = min(min_seg, l2.min())
logging.info("  min coord dist in HUC boundaries = %g"%min_len)
logging.info("  min seg length in HUC boundaries = %g"%min_seg)

min_seg = 1.e10
for forest in forests:
    for tree in forest:
        for seg in tree.dfs():
            coords = np.array(seg.coords)
            assert(coords.shape[1] == 2)
            l2 = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            assert(len(l2) == len(coords)-1)
            assert(l2.min() > 1.e-3)
            min_seg = min(min_seg, l2.min())
seg_lens = [seg.length for forest in forests for tree in forest for seg in tree.dfs()]
logging.info("  min coord dist in rivers = %g"%min_seg)
logging.info("  min seg length in rivers = %g"%min(seg_lens))
logging.info("  mean seg length in rivers = %g"%np.mean(seg_lens))
logging.info("  median seg length in rivers = %g"%np.median(seg_lens))

if True:
    for poly in hucs_s.polygons():
        workflow.plot.huc(poly, style='-x', color='r')
    for forest in forests:
        rivers = workflow.tree.forest_to_list(forest)
        workflow.plot.river(rivers, style='-+', color='b')

    exits = np.array([next(tree.preFetch()).segment.coords[-1] for forest in forests for tree in forest])
    plt.scatter(exits[:,0], exits[:,1], color='g')

    for seg in hucs_s.segments:
        if abs(min_len - seg.length) < 0.001:
            workflow.plot.river([seg,], style="-o", color='k')
    plt.show()
    

    plt.show()



# # === triangulate ===
# logging.info("triangulate")
# def max_area_valid(distance):
#     """A function to make sure max area scales with distance from river network

#     Units in [m]
#     """
#     far = 1000
#     near = 500
#     far_size = 500
#     near_size = 100
#     if distance > far:
#         size = far_size
#     elif distance < near:
#         size = near_size
#     else:
#         size = near_size + (distance - near) / (far - near) * (far_size - near_size)
#     return size**2 / 2
    

# def needs_refinement(vertices, area):
#     bary = np.sum(np.array(vertices), axis=0)/3
#     bary_p = shapely.geometry.Point(bary[0], bary[1])
#     distance = bary_p.distance(rivers_part[0])
#     return bool(area > max_area_valid(distance))

# mesh_points, mesh_tris = workflow.triangulate.triangulate_with_rivers(hucs_simp[0], rivers_part[0], verbose=True, refinement_func=needs_refinement)


# # plot
# plt.figure()
# workflow.plot.tri(mesh_points, mesh_tris)
# #plt.figure()
# cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
# for i, (s,r) in enumerate(zip(hucs_simp, rivers_part)):
#     c = cm(i)
#     c_huc = workflow.colors.darken(c)
#     workflow.plot.huc(s, color=c_huc, style='-x')
#     workflow.plot.river(r, color=c)

# plt.axes().set_aspect('equal', 'datalim')
# plt.show()


# # download and tile a DEM for this entire HUC
# logging.info("tiling with DEMs")
# dem_profile, dem = workflow.clip.clip_dem(huc8)
# dem = dem[0,:,:] # only the first band

# # collect DEM values from the points (mostly done, add here) --etc
# # -- must map back to lat/lon to take from dem
# logging.info("grabbing elevation")
# triangles_3d = []
# mesh_points_ll = np.array(workflow.warp.warp_xy(mesh_points[:,0], mesh_points[:,1], workflow.conf.default_crs(), workflow.conf.latlon_crs())).transpose()
# elev = dem[workflow.rowcol.rowcol(dem_profile['affine'], mesh_points_ll[:,0], mesh_points_ll[:,1])]
# mesh_points_3 = np.zeros((len(mesh_points),3),'d')
# mesh_points_3[:,0:2] = mesh_points
# mesh_points_3[:,2] = elev


# # # plot
# # fig =plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_trisurf(mesh_points[:,0], mesh_points[:,1], elev, triangles=mesh_tris)
# # plt.show()
    
    
# # save as a mesh
# # this could be cleaner, but meshing_ats is in python2 (and uses exodus which is in python2)
# logging.info("saving mesh")
# filename = os.path.join(outdir, 'huc_%s.vtk'%huc12s[0]['properties']['HUC12'])
# cells = {'triangle':mesh_tris}
# vtk_io.write(filename, mesh_points_3, cells)
