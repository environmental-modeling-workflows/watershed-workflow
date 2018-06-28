import os
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

myhuc = '06010208'
outdir = "data/meshes/%s/12"%myhuc
if not os.path.isdir(outdir):
    os.makedirs(outdir)

## === Preprocess HUCs ===
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

# cleanup rivers
rivers_s = workflow.hydrography.cleanup(hucs_s, rivers_s, simp_tol=0.1, prune_tol=10)

if True:
    workflow.plot.huc(hucs_s[0], style='-x')
    h0rivers = workflow.tree.forest_to_list(rivers_s[0])
    workflow.plot.river(h0rivers, style='-+')
    plt.show()


## === Combine and split HUCs and Hydrography into HUC partitions ===
# split polygons into spine and boundary
logging.info("splitting HUCs into spine and exterior boundary")
uniques, intersections = workflow.smooth.intersect_and_split(hucs_s)
assert(all(type(u) is shapely.geometry.LineString or
           type(u) is shapely.geometry.MultiLineString or
           type(u) is type(None) for u in uniques))
assert(all(type(s) is shapely.geometry.LineString or
           type(s) is shapely.geometry.MultiLineString or
           type(s) is type(None) for i in intersections for s in i))

# generate a list of all spine/boundary segments
logging.info("cut the spine with rivers as needed")
all_segs = []
for s in uniques:
    if type(s) is shapely.geometry.LineString:
        all_segs.append(s)
    elif type(s) is shapely.geometry.MultiLineString:
        for seg in s:
            assert(type(seg) is shapely.geometry.LineString)
            all_segs.append(seg)
for i in intersections:
    for s in i:
        if type(s) is shapely.geometry.LineString:
            all_segs.append(s)
        elif type(s) is shapely.geometry.MultiLineString:
            for seg in s:
                assert(type(seg) is shapely.geometry.LineString)
                all_segs.append(seg)

# intersect rivers with spine and boundary, adding points at any
# intersection to both river and spine/boundary segment
assert(all(type(r) is shapely.geometry.LineString for r in rivers_s))
assert(all(type(s) is shapely.geometry.LineString for s in all_segs))
workflow.hydrography.split_spine(all_segs, rivers_s, 10) # units m
assert(all(type(r) is shapely.geometry.LineString for r in rivers_s))
assert(all(type(s) is shapely.geometry.LineString for s in all_segs))

# add rivers to all segs -- this is truely now all line objects in the
# combined HUC/river shapelist
logging.info("simplify")
all_segs = rivers_s + all_segs
all_segs = shapely.geometry.MultiLineString(all_segs)

# simplify to coarsen
all_segs_simp = list(all_segs.simplify(10.))  # units = m
assert(len(all_segs_simp) == len(all_segs))
assert(all(type(s) is shapely.geometry.LineString for s in all_segs_simp))

# check min distances
min_seg = 1.e10
for seg in all_segs_simp:
    coords = np.array(seg.coords)
    assert(coords.shape[1] == 2)
    l2 = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    assert(len(l2) == len(coords)-1)
    assert(l2.min() > 1.e-3)
    min_seg = min(min_seg, l2.min())
logging.info("  Min distance = %g"%min_seg)

# restructure back to the original format of uniques/intersections/rivers
# -- pop the rivers
rivers_simp = all_segs_simp[0:len(rivers_s)]
all_segs_simp = all_segs_simp[len(rivers_s):]

# -- next the uniques
uniques_simp = [None,]*len(uniques)
pos = 0
for i,u in enumerate(uniques):
    if type(u) is shapely.geometry.LineString:
        uniques_simp[i] = all_segs_simp[pos]
        pos += 1
    elif type(u) is shapely.geometry.MultiLineString:
        num_segs = len(u)
        uniques_simp[i] = shapely.geometry.MultiLineString(all_segs_simp[pos:pos+num_segs])
        pos += num_segs

# -- finally the intersections
intersections_simp = [[None for i in range(len(intersections))] for j in range(len(intersections))]
for i,inter in enumerate(intersections):
    for j,u in enumerate(inter):
        if type(u) is shapely.geometry.LineString:
            intersections_simp[i][j] = all_segs_simp[pos]
            pos += 1
        elif type(u) is shapely.geometry.MultiLineString:
            num_segs = len(u)
            intersections_simp[i][j] = shapely.geometry.MultiLineString(all_segs_simp[pos:pos+num_segs])
            pos += num_segs

# -- check the final tally -- we better have gotten them all            
assert(pos == len(all_segs_simp))

# recombine the simplified objects
logging.info("recombine into HUCs")
hucs_simp = workflow.smooth.recombine(uniques_simp, intersections_simp)

# sort rivers by containing poly
logging.info("partition rivers")
rivers_part = workflow.hydrography.sort_precut(hucs_simp, rivers_simp)

min_len = min(line.length for river_part in rivers_part for line in river_part)
logging.info("min length segment = %g"%min_len)
        

# plot to check before triangulating
# cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
# for i, (s,r) in enumerate(zip(hucs_simp, rivers_part)):
#     c = cm(i)
#     c_huc = workflow.colors.darken(c)
#     workflow.plot.huc(s, color=c_huc, style='-x')
#     workflow.plot.river(r, color=c)
# plt.show()

workflow.plot.huc(hucs_simp[0], style='-x')
workflow.plot.river(rivers_part[0], style='-+')
plt.show()

# === triangulate ===
logging.info("triangulate")
def max_area_valid(distance):
    """A function to make sure max area scales with distance from river network

    Units in [m]
    """
    far = 1000
    near = 500
    far_size = 500
    near_size = 100
    if distance > far:
        size = far_size
    elif distance < near:
        size = near_size
    else:
        size = near_size + (distance - near) / (far - near) * (far_size - near_size)
    return size**2 / 2
    

def needs_refinement(vertices, area):
    bary = np.sum(np.array(vertices), axis=0)/3
    bary_p = shapely.geometry.Point(bary[0], bary[1])
    distance = bary_p.distance(rivers_part[0])
    return bool(area > max_area_valid(distance))

mesh_points, mesh_tris = workflow.triangulate.triangulate_with_rivers(hucs_simp[0], rivers_part[0], verbose=True, refinement_func=needs_refinement)


# plot
plt.figure()
workflow.plot.tri(mesh_points, mesh_tris)
#plt.figure()
cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
for i, (s,r) in enumerate(zip(hucs_simp, rivers_part)):
    c = cm(i)
    c_huc = workflow.colors.darken(c)
    workflow.plot.huc(s, color=c_huc, style='-x')
    workflow.plot.river(r, color=c)

plt.axes().set_aspect('equal', 'datalim')
plt.show()


# download and tile a DEM for this entire HUC
logging.info("tiling with DEMs")
dem_profile, dem = workflow.clip.clip_dem(huc8)
dem = dem[0,:,:] # only the first band

# collect DEM values from the points (mostly done, add here) --etc
# -- must map back to lat/lon to take from dem
logging.info("grabbing elevation")
triangles_3d = []
mesh_points_ll = np.array(workflow.warp.warp_xy(mesh_points[:,0], mesh_points[:,1], workflow.conf.default_crs(), workflow.conf.latlon_crs())).transpose()
elev = dem[workflow.rowcol.rowcol(dem_profile['affine'], mesh_points_ll[:,0], mesh_points_ll[:,1])]
mesh_points_3 = np.zeros((len(mesh_points),3),'d')
mesh_points_3[:,0:2] = mesh_points
mesh_points_3[:,2] = elev


# # plot
# fig =plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(mesh_points[:,0], mesh_points[:,1], elev, triangles=mesh_tris)
# plt.show()
    
    
# save as a mesh
# this could be cleaner, but meshing_ats is in python2 (and uses exodus which is in python2)
logging.info("saving mesh")
filename = os.path.join(outdir, 'huc_%s.vtk'%huc12s[0]['properties']['HUC12'])
cells = {'triangle':mesh_tris}
vtk_io.write(filename, mesh_points_3, cells)
