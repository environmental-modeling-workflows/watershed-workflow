import workflow.conf
import workflow.smooth
import workflow.download
import workflow.triangulate
import shapely

import matplotlib.pyplot as plt

# collect HUC shapefile
workflow.download.collect_huc('06')

# collect DEM (done, add here) --etc
    
# load shapefiles for all HUC 12s in the Obed HUC 8.
profile, hucs = workflow.conf.load_hucs_in('06010208', 12)

# change coordinates to UTM
# (done add here) --etc

# convert to shapely
hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]

# intersect, finding shared boundaries
uniques, intersections = workflow.smooth.intersect_and_split(hucs_s)

# smooth/simplify/resample to a given spacing (in meters)
uniques_sm, intersections_sm = workflow.smooth.smooth(uniques,intersections,100./111000.) # roughly converts 100m to degrees, fixme once in UTM --etc

# recombine
hucs_sm = workflow.smooth.recombine(uniques_sm, intersections_sm)

# triangulate (to a refinement with max_area, units a bit unclear.
# I believe these should be degrees^2, then m^2 once in UTM, but the magnitude seems
# wrong for that.  Takes some fiddling.)
#
# Then plot the triangles.
# TODO -- add refinement function based on distance function and hydrography data
plt.figure()
for huc in hucs_sm:
    mesh_points, mesh_tris = workflow.triangulate.triangulate(huc, max_area=0.0001)
    plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
plt.show()

# collect DEM values from the points (mostly done, add here) --etc

# bring in other data streams? (not done) --etc

# extrude (done, using meshing_ats) --etc
