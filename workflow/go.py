import os
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

import shapely
import meshpy.triangle

import workflow.conf
import workflow.smooth
import workflow.download
import workflow.triangulate
import workflow.clip
import workflow.warp
import workflow.rowcol

import vtk_io # from ATS/tools/meshing_ats

myhuc = '06010208'
outdir = "data/meshes/%s/12"%myhuc
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# collect HUC shapefile
print("downloading HUC %s"%myhuc[0:2])
workflow.download.download_huc(myhuc[0:2])

# load shapefiles for the HUC8 of interest
print("loading HUC %s"%myhuc)
profile, huc8 = workflow.conf.load_huc(myhuc)

# download and tile a DEM for this entire HUC
print("tiling with DEMs")
dem_profile, dem = workflow.clip.clip_dem(huc8)
dem = dem[0,:,:] # only the first band

# load shapefiles for all HUC 12s in the Obed HUC 8.
print("loading all 12s")
profile, huc12s = workflow.conf.load_hucs_in('06010208', 12)

# change coordinates to meters (in place)
print("change coordinates to m")
for huc12 in huc12s:
    workflow.warp.warp_shape(huc12, profile['crs'], workflow.conf.default_crs())

# convert to shapely
shps = [shapely.geometry.shape(s['geometry']) for s in huc12s]

# intersect, finding shared boundaries
print("intersecting to find boundary spine")
uniques, intersections = workflow.smooth.intersect_and_split(shps)

# smooth/simplify/resample to a given spacing (in meters)
print("smoothing")
uniques_sm, intersections_sm = workflow.smooth.smooth(uniques,intersections,100.)  # units = m

# recombine
print("recombine")
shps_sm = workflow.smooth.recombine(uniques_sm, intersections_sm)

# triangulate (to a refinement with max_area, units a bit unclear.
# I believe these should be degrees^2, then m^2 once in UTM, but the magnitude seems
# wrong for that.  Takes some fiddling.)
#
# Then plot the triangles.
# TODO -- add refinement function based on distance function and hydrography data
print("triangulating")
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
triangles = []
for shp in shps_sm:
    mesh_points, mesh_tris = workflow.triangulate.triangulate(shp, max_area=1e5) # units = m^2?
    #ax.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    triangles.append((mesh_points, mesh_tris))

# collect DEM values from the points (mostly done, add here) --etc
# -- must map back to lat/lon to take from dem
print("grabbing elevation")
ax = fig.add_subplot(1, 2, 2, projection='3d')

triangles_3d = []

for mesh_points, mesh_tris in triangles:
    mesh_points_ll = np.array(workflow.warp.warp_xy(mesh_points[:,0], mesh_points[:,1], workflow.conf.default_crs(), workflow.conf.latlon_crs())).transpose()
    elev = dem[workflow.rowcol.rowcol(dem_profile['affine'], mesh_points_ll[:,0], mesh_points_ll[:,1])]
    #ax.plot_trisurf(mesh_points[:,0], mesh_points[:,1], elev, triangles=mesh_tris)

    triangles_3d.append((np.array([mesh_points[:,0], mesh_points[:,1], elev]).transpose(), mesh_tris))
#plt.show()

# bring in other data streams? (not done) --etc

# save as a mesh
# this could be cleaner, but meshing_ats is in python2 (and uses exodus which is in python2)
for huc, mesh in zip(huc12s, triangles_3d):
    filename = os.path.join(outdir, 'huc_%s.vtk'%huc['properties']['HUC12'])
    cells = {'triangle':mesh[1]}
    vtk_io.write(filename, mesh[0], cells)
