import os,sys
import numpy as np
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib import pyplot as plt

# Watershed Workflow
import workflow
import workflow.utils
import workflow.colors
import workflow.condition
import workflow.mesh
import workflow.split_hucs
import workflow.river_tree
import workflow.triangulation_with_streams

## **** Mimicking the river network and watershed boundaries that WW will pull from respective sources
watershed_boundary_points = [(0,0),(100,0.001),(300,0.000),(400,0.001),(400.001,100),(400,200),(400.001,300),(400,400),(300,450.001),(200,500),(100,450.001),(0,400),(0.001,300),(0,200),(0.001,100)]
stream_network_points= [[(200,200),(200.001,150),(200,100),(200+0.001,50),(200,0)],
              [(300,200),(250,200.001),(200,200)],
              [(100,200),(150,200.001),(200,200)],
              [(100,300),(100.001,250),(100,200)],
              [(300,300),(300.001,250),(300,200)],
              [(50,300),(75,300.001),(100,300)],
              [(350,300),(300,300.001)],
              [(200,300),(250,300.001),(300,300)],
              [(200,400),(200.001,350),(200,300)],
              [(350,400),(350.001,350),(350,300)],
               ] 
               # note that we have avoided collinearity by inserting small pertubations to preserve
               # these points else "simplify" might delete them and "masking" might ignore them
               # once we have more controlled refining, this is not needed

## what we would get from "workflow.get_split_form_hucs" and "workflow.get_reaches" 
stream_network= shapely.geometry.MultiLineString(stream_network_points)
watershed_boundary=shapely.geometry.Polygon(watershed_boundary_points)
sbox = workflow.split_hucs.SplitHUCs([watershed_boundary,])
## ****  --  --  --  --  

### **** some functions 

def points_on_in_rc(points, polygon_):
    """function to identify which nodes of the mesh falls on or in the river corridor"""
    # in mixed-element mesh we have this information, this function is for a more general case
    points_ = [Point(point[0], point[1])  for point in points] # underscore represents shapely object
    res1=[polygon_.touches(point_) for point_ in points_]
    res2=[polygon_.contains(point_) for point_ in points_]
    res=[]
    for j in range(len(res1)):
        res.append(res1[j] or res2[j])
    return res

### **** Meshing begin

rivers = workflow.simplify_and_prune(sbox,stream_network, simplify=0) ## simply function needs to be updated so as not to remove points from straight lines
river = rivers[0] # river as tree

# plotting huc and river tree
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(sbox.exterior().exterior.xy[0], sbox.exterior().exterior.xy[1], 'k-x')
for reach in rivers[0].dfs():
    ax.plot(reach.xy[0], reach.xy[1], 'r-x')
plt.title('HUCs and River Network')
plt.show()

## **** River corridor masking
delta=4 # river width
corr1, corr2, corr3 = workflow.river_tree.create_river_corridor(river, delta)
corr = corr3

# plotting river corridor and river tree together
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.plot(sbox.exterior().exterior.xy[0], sbox.exterior().exterior.xy[1], 'k-x')
ax.plot(corr.exterior.xy[0], corr.exterior.xy[1], 'b-x')
for reach in rivers[0].dfs():
   ax.plot(reach.xy[0], reach.xy[1], 'r-x')
plt.title('HUCs, Network and River Corridor')
plt.show()

## **** Triangulating the domain with river corridor as a hole
kwargs = {'max_volume': 700, 'allow_boundary_steiner': False}
points, elems= workflow.triangulation_with_streams.triangulate(sbox, corr ,mixed=True ,tol=1, **kwargs)

# computing area triangles for statistics
areas = np.array([workflow.utils.triangle_area(points[e]) for e in elems])
print('maximum area of the traingle', round(max(areas),6) ,', minimum area of the triangle', round(min(areas),6), ', number of triangles', len(areas))

# plotting triangulated domain with river-corridor as a hole
plt.rcParams['figure.figsize'] = [10, 10]
plt.tripcolor(points[:,0], points[:,1], elems, areas, edgecolor='r')
plt.title('Triangulated Watershed with River-corridor as a Hole')
plt.show()

## **** Inserting mesh elements into hole

# creating quad elements
corr_coords=corr.exterior.coords[:-1]
fig = plt.figure(figsize=(10,10)) # this plotting is currently mandated in to_quad function
ax = fig.add_subplot(111)
ax.plot(sbox.exterior().exterior.xy[0], sbox.exterior().exterior.xy[1], 'k-x')
quads= workflow.river_tree.to_quads(river, delta, sbox, corr_coords,ax)
for elem in quads:
    looped_conn = elem[:]
    looped_conn.append(elem[0])
    cc = np.array([corr_coords[n] for n in looped_conn])
    ax.plot(cc[:,0], cc[:,1], 'm-^')
plt.title('Quads Created in the River Corridor')
plt.show()

# adding quad elements into existing element list
elems_list=list(elems)
for elem in quads:
    elems_list.append(elem)

## **** Elevating the mesh to create a surface terrain mesh

# providing elevations to each point of the mesh (real case this would come from DEMs)
points3=np.zeros((len(points),3))
points3[:,:2]=points
points3[:,2]=9+abs(points[:,0]-200)/800+points[:,1]/500 # gradient towards the outlet 

# identifying the mesh points in the river corridor
inds=points_on_in_rc(points, corr)
points3[inds,2]=points3[inds,2]-2 # depressing the surface by 2 units in the stream

# creating surface mesh in 3D using watershed workflow tools
m2 = workflow.mesh.Mesh2D(points3.copy(), elems_list)
#workflow.condition.fill_pits(m2)

# plotting surface mesh with elevations
crs=None
start=min(m2.centroids()[:,2])
step=(max(m2.centroids()[:,2])-(min(m2.centroids()[:,2])))/20
stop=max(m2.centroids()[:,2])+step
legend_values=np.arange(start,stop,step)
indices, cmap, norm, ticks, labels = workflow.colors.generate_indexed_colormap(legend_values, cmap='jet')
fig, ax = workflow.plot.get_ax(crs)
fig.set_size_inches(12, 10)

plt.rcParams['figure.figsize'] = [10, 10]
mp = workflow.plot.mesh(m2, crs, ax=ax, 
                        linewidth=2 ,color=m2.centroids()[:,2], 
                        cmap=cmap, norm = norm)

workflow.colors.colorbar_index(ncolors=len(legend_values), cmap=cmap, labels = labels) 
plt.title('Surface Mesh with Elevations')
plt.show()

## **** Creating a 3D mesh

# this is oversimplified case where we wille xtrude this mesh in 3D with 10 layers of prescribed thicknesses

total_thickness = 10
dzs=[0.5,0.5,0.5,0.5,0.75,0.75,1.25,1.25,2,2]
assert(sum(dzs)==total_thickness)

# layer extrusion
# -- data structures needed for extrusion
layer_types = []
layer_data = []
layer_ncells = []
layer_mat_ids = []

depth = 0
for dz in dzs:
    depth += 0.5 * dz
    layer_types.append('constant')
    layer_data.append(dz)
    layer_ncells.append(1)
    layer_mat_ids.append(1000)
  
    depth += 0.5 * dz

# print the summary
workflow.mesh.Mesh3D.summarize_extrusion(layer_types, layer_data, 
                                            layer_ncells, layer_mat_ids)
# extrude
m3 = workflow.mesh.Mesh3D.extruded_Mesh2D(m2,layer_types, layer_data, 
                                            layer_ncells, layer_mat_ids)

# saving mesh as exodus file
m3.write_exodus('mixed_element_mesh.exo')











