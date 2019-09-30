Example: mesh a delineated watershed
====================================

Here we mesh the Coweeta Hydrologic Laboratory as an example of how to
pull data in from default locations and generate a fully functional ATS
mesh.

This includes the full shebang: - NHD Plus for river network - NRCS
soils data for soil types - NLCD for land cover/transpiration/rooting
depths - NED for elevation

.. code:: ipython3

    %matplotlib inline
    #%matplotlib OSX

.. code:: ipython3

    import os,sys
    import numpy as np
    from matplotlib import pyplot as plt
    import shapely
    import logging
    import math
    
    import workflow
    import workflow.source_list
    import workflow.ui
    import workflow.conf
    import workflow.split_hucs
    import workflow.extrude
    import workflow.colors
    import workflow.condition
    
    workflow.ui.setup_logging(1,None)

Sources and setup
-----------------

Next we set up the source watershed and coordinate system and all data
sources for our mesh. We will use the CRS that is included in the
shapefile.

A wide range of sources are available; here we use the defaults except
for using NHD Plus for watershed boundaries and hydrography, just to
demonstrate how to change the defaults.

.. code:: ipython3

    coweeta_shapefile = '../data/hydrologic_units/others/coweeta/coweeta_basin.shp'
    hint = '0601'  # hint: HUC 4 containing this shape.  
        # This is necessary to avoid downloading all HUCs to search for this shape
    
    logging.info("")
    logging.info("Meshing shape: {}".format(coweeta_shapefile))
    logging.info("="*30)


::

    2019-09-24 09:01:00,630 - root - INFO: 
    2019-09-24 09:01:00,633 - root - INFO: Meshing shape: ../data/hydrologic_units/others/coweeta/coweeta_basin.shp
    2019-09-24 09:01:00,634 - root - INFO: ==============================


.. code:: ipython3

    sources = workflow.source_list.get_default_sources()
    sources['HUC'] = workflow.source_list.huc_sources['NHD Plus']
    sources['hydrography'] = workflow.source_list.hydrography_sources['NHD Plus']
    workflow.source_list.log_sources(sources)


::

    2019-09-24 09:01:00,642 - root - INFO: Using sources:
    2019-09-24 09:01:00,644 - root - INFO: --------------
    2019-09-24 09:01:00,645 - root - INFO: HUC: National Hydrography Dataset Plus High Resolution (NHDPlus HR)
    2019-09-24 09:01:00,647 - root - INFO: hydrography: National Hydrography Dataset Plus High Resolution (NHDPlus HR)
    2019-09-24 09:01:00,648 - root - INFO: DEM: National Elevation Dataset (NED)
    2019-09-24 09:01:00,649 - root - INFO: soil type: National Resources Conservation Service Soil Survey (NRCS Soils)
    2019-09-24 09:01:00,650 - root - INFO: land cover: National Land Cover Database (NLCD) Layer: NLCD_2016_Land_Cover_L48
    2019-09-24 09:01:00,651 - root - INFO: soil thickness: None


.. code:: ipython3

    # get the shape and crs of the shape
    crs, coweeta_shapes = workflow.get_shapes(coweeta_shapefile)
    
    # the split-form is a useful data structure representation used by some of the 
    # simplification and manipulation routines.  it is more useful for multiple, 
    # nested shapes, but we'll use it here anyway.
    coweeta_shape_split = workflow.split_hucs.SplitHUCs(coweeta_shapes)



::

    2019-09-24 09:01:00,658 - root - INFO: 
    2019-09-24 09:01:00,660 - root - INFO: Preprocessing Shapes
    2019-09-24 09:01:00,661 - root - INFO: ------------------------------
    2019-09-24 09:01:00,663 - root - INFO: loading file: "../data/hydrologic_units/others/coweeta/coweeta_basin.shp"
    /Users/uec/codes/anaconda/3/envs/ats_meshing_20190719/lib/python3.7/site-packages/fiona/collection.py:336: FionaDeprecationWarning: Collection slicing is deprecated and will be disabled in a future version.
      return self.session.__getitem__(item)


Generate the surface mesh
-------------------------

First we’ll generate the flattened, 2D triangulation, which builds on
hydrography data. Then we download a digital elevation map from the
National Elevation Dataset, and extrude that 2D triangulation to a 3D
surface mesh based on nearest-neighbor pixels of the DEM.

.. code:: ipython3

    # download/collect the river network within that shape's bounds
    _, reaches = workflow.get_reaches(sources['hydrography'], hint, coweeta_shape_split.exterior().bounds, crs)
    
    # simplify and prune rivers not IN the shape, constructing a tree-like data structure for the river network
    rivers = workflow.simplify_and_prune(coweeta_shape_split, reaches, cut_intersections=True)


::

    2019-09-24 09:01:00,696 - root - INFO: 
    2019-09-24 09:01:00,697 - root - INFO: Preprocessing Hydrography
    2019-09-24 09:01:00,698 - root - INFO: ------------------------------
    2019-09-24 09:01:00,699 - root - INFO: loading streams in HUC 0601
    2019-09-24 09:01:00,700 - root - INFO: and/or bounds (273971.0911428, 3878839.6361173, 279140.9150949, 3883953.7853134)
    2019-09-24 09:01:00,702 - root - INFO: Using Hydrography file "/Users/uec/research/water/data/meshing/data/hydrography/NHDPlus_H_0601_GDB/NHDPlus_H_0601.gdb"
    2019-09-24 09:01:04,015 - root - INFO: 
    2019-09-24 09:01:04,015 - root - INFO: Simplifying and pruning
    2019-09-24 09:01:04,016 - root - INFO: ------------------------------
    2019-09-24 09:01:04,017 - root - INFO: Filtering rivers outside of the HUC space
    2019-09-24 09:01:04,019 - root - INFO:   ...filtering
    2019-09-24 09:01:04,038 - root - INFO: Generate the river tree
    2019-09-24 09:01:04,041 - root - INFO: Removing rivers with fewer than 0 reaches.
    2019-09-24 09:01:04,041 - root - INFO:   ...keeping river with 21 reaches
    2019-09-24 09:01:04,042 - root - INFO: simplifying rivers
    2019-09-24 09:01:04,053 - root - INFO: simplifying HUCs
    2019-09-24 09:01:04,056 - root - INFO: snapping rivers and HUCs
    2019-09-24 09:01:04,058 - root - INFO:   snapping polygon segment boundaries to river endpoints
    2019-09-24 09:01:04,062 - root - INFO:   snapping river endpoints to the polygon
    2019-09-24 09:01:04,080 - root - INFO:   cutting at crossings
    2019-09-24 09:01:04,107 - root - INFO:   filtering rivers to HUC
    2019-09-24 09:01:04,109 - root - INFO:   ...filtering
    2019-09-24 09:01:04,120 - root - INFO: 
    2019-09-24 09:01:04,120 - root - INFO: Simplification Diagnostics
    2019-09-24 09:01:04,121 - root - INFO: ------------------------------
    2019-09-24 09:01:04,126 - root - INFO:   river min seg length: 38.7145
    2019-09-24 09:01:04,127 - root - INFO:   river median seg length: 65.1981
    2019-09-24 09:01:04,129 - root - INFO:   HUC min seg length: 11.4828
    2019-09-24 09:01:04,130 - root - INFO:   HUC median seg length: 20.9713


.. code:: ipython3

    # form a triangulation on the shape + river network
    
    # triangulation refinement:
    # Refine triangles if their area (in m^2) is greater than A(d), where d is the distance from the triangle
    # centroid to the nearest stream.
    # A(d) is a piecewise linear function -- A = A0 if d <= d0, A = A1 if d >= d1, and linearly interpolates
    # between the two endpoints.
    d0 = 100; d1 = 500
    A0 = 500; A1 = 2500
    #A0 = 100; A1 = 500
    
    # Refine triangles if they get to acute
    min_angle = 32 # degrees
    
    # make 2D mesh
    mesh_points2, mesh_tris = workflow.triangulate(coweeta_shape_split, rivers, 
                                                   refine_distance=[d0,A0,d1,A1],
                                                   refine_min_angle=min_angle,
                                                   diagnostics=True)


::

    2019-09-24 09:01:04,138 - root - INFO: 
    2019-09-24 09:01:04,139 - root - INFO: Meshing
    2019-09-24 09:01:04,140 - root - INFO: ------------------------------
    2019-09-24 09:01:04,141 - root - INFO: Triangulating...
    2019-09-24 09:01:04,144 - root - INFO:    265 points and 265 facets
    2019-09-24 09:01:04,145 - root - INFO:  checking graph consistency
    2019-09-24 09:01:04,149 - root - INFO:  building graph data structures
    2019-09-24 09:01:04,152 - root - INFO:  triangle.build...
    2019-09-24 09:01:22,704 - root - INFO:   ...built: 16097 mesh points and 31875 triangles
    2019-09-24 09:01:22,705 - root - INFO: Plotting triangulation diagnostics



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_9_1.png


.. code:: ipython3

    # get a raster for the elevation map
    dem_profile, dem = workflow.get_raster_on_shape(sources['DEM'], coweeta_shape_split.exterior(), crs)
    
    # elevate the triangle nodes to the dem
    mesh_points3 = workflow.elevate(mesh_points2, crs, dem, dem_profile)


::

    2019-09-24 09:01:26,805 - root - INFO: 
    2019-09-24 09:01:26,809 - root - INFO: Preprocessing Raster
    2019-09-24 09:01:26,810 - root - INFO: ------------------------------
    2019-09-24 09:01:26,812 - root - INFO: collecting raster
    2019-09-24 09:01:26,815 - root - INFO: Collecting DEMs to tile bounds: [-83.48845037186398, 35.01734099944024, -83.41165773504356, 35.08381933600244]
    2019-09-24 09:01:26,818 - root - INFO:   Need:
    2019-09-24 09:01:26,819 - root - INFO:     /Users/uec/research/water/data/meshing/data/dem/USGS_NED_1as_n36_w084.img
    2019-09-24 09:01:26,843 - root - INFO: 
    2019-09-24 09:01:26,844 - root - INFO: Elevating Triangulation to DEM
    2019-09-24 09:01:26,845 - root - INFO: ------------------------------


.. code:: ipython3

    # plot the resulting surface mesh
    fig = plt.figure(figsize=(10,6))
    ax = workflow.plot.get_ax('3d', fig, window=[0.0,0.2,1,0.8])
    cax = fig.add_axes([0.23,0.18,0.58,0.03])
    
    mp = ax.plot_trisurf(mesh_points3[:,0], mesh_points3[:,1], mesh_points3[:,2], triangles=mesh_tris, 
                    cmap='viridis', edgecolor=(0,0,0,.2), linewidth=0.5)
    cb = fig.colorbar(mp, orientation="horizontal", cax=cax)
    
    rivers_2 = [np.array(r.xy) for riv in rivers for r in riv]
    rivers_e = [workflow.values_from_raster(r.transpose(), crs, dem, dem_profile) for r in rivers_2]
    rivers_l3 = [np.array([i[0], i[1], j]).transpose() for i,j in zip(rivers_2, rivers_e)]
    for r in rivers_l3:
        ax.plot(r[:,0]+1, r[:,1], r[:,2]+10, color='red', linewidth=3)
    
    t = cax.set_title('elevation [m]')
    ax.view_init(55,0)
    
    #fig.savefig('dem_b.png', dpi=600)
    plt.show()



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_11_0.png


.. code:: ipython3

    # plot the resulting surface mesh
    fig = plt.figure(figsize=(16,16))
    ax = workflow.plot.get_ax(crs, fig)
    
    mp = workflow.plot.triangulation(mesh_points3, mesh_tris, crs, ax=ax, linewidth=0.5, 
                                     color='elevation', edgecolor='gray')
    fig.colorbar(mp, orientation="horizontal", pad=0.1)
    workflow.plot.hucs(coweeta_shape_split, crs, ax=ax, color='k', linewidth=1)
    workflow.plot.rivers(rivers, crs, ax=ax, color='red', linewidth=1)
    ax.set_aspect('equal', 'datalim')
    ax.set_title('2D mesh and digital elevation map')


::

    got projection: _EPSGProjection(26917)
    got projection: _EPSGProjection(26917)




::

    Text(0.5, 1.0, '2D mesh and digital elevation map')




.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_12_2.png


.. code:: ipython3

    # construct the 2D mesh
    m2 = workflow.extrude.Mesh2D(mesh_points3.copy(), list(mesh_tris))

.. code:: ipython3

    # hydrologically condition the mesh
    workflow.condition.condition(m2)
    
    # plot the change between the two meshes
    diff = np.copy(mesh_points3)
    diff[:,2] = m2.points[:,2] - mesh_points3[:,2] 
    print("max diff = ", np.abs(diff[:,2]).max())
    fig, ax = workflow.plot.get_ax(crs, figsize=(20,20))
    workflow.plot.triangulation(diff, m2.conn, crs, color='elevation', edgecolors='k', ax=ax)
    ax.set_title('conditioned dz')
    plt.show()


::

    max diff =  3.8717899038992982



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_14_1.png


Surface properties
------------------

Meshes interact with data to provide forcing, parameters, and more in
the actual simulation. Specifically, we need vegetation type on the
surface to provide information about transpiration and subsurface
structure to provide information about water retention curves, etc.

We’ll start by downloading and collecting land cover from the NLCD
dataset, and generate sets for each land cover type that cover the
surface. Likely these will be some combination of grass, deciduous
forest, coniferous forest, and mixed.

.. code:: ipython3

    # download the NLCD raster
    lc_profile, lc_raster = workflow.get_raster_on_shape(sources['land cover'], coweeta_shape_split.exterior(), crs)
    
    # resample the raster to the triangles
    lc = workflow.values_from_raster(m2.centroids(), crs, lc_raster, lc_profile)
    
    # what land cover types did we get?
    logging.info('Found land cover dtypes: {}'.format(lc.dtype))
    logging.info('Found land cover types: {}'.format(set(lc)))



::

    2019-09-24 09:01:34,525 - root - INFO: 
    2019-09-24 09:01:34,525 - root - INFO: Preprocessing Raster
    2019-09-24 09:01:34,526 - root - INFO: ------------------------------
    2019-09-24 09:01:34,528 - root - INFO: collecting raster
    2019-09-24 09:01:34,534 - root - INFO: CRS: PROJCS["Albers_Conical_Equal_Area",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,-0,-0,-0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["meters",1]]


::

      filename: /Users/uec/research/water/data/meshing/data/land_cover/NLCD_2016_Land_Cover_L48/NLCD_2016_Land_Cover_L48.img


::

    2019-09-24 09:01:35,176 - root - INFO: Found land cover dtypes: uint8
    2019-09-24 09:01:35,182 - root - INFO: Found land cover types: {41, 42, 43, 81, 52, 21, 22, 23}


.. code:: ipython3

    import collections
    import matplotlib.colors
    nlcd_color_map_values = collections.OrderedDict({
        21:    (0.86666666667,  0.78823529412,  0.78823529412),
        22:    (0.84705882353,  0.57647058824,  0.50980392157),
        23:    (0.92941176471,  0.00000000000,  0.00000000000),
        41:    (0.40784313726,  0.66666666667,  0.38823529412),
        42:    (0.10980392157,  0.38823529412,  0.18823529412),
        43:    (0.70980392157,  0.78823529412,  0.55686274510),
        52:    (0.80000000000,  0.72941176471,  0.48627450980),
        81:    (0.85882352941,  0.84705882353,  0.23921568628),
    })
    
    _nlcd_labels = collections.OrderedDict({
        21: 'Developed, Open',
        22: 'Developed, Light',
        23: 'Developed, Med.',
        41: 'Deciduous Forest',
        42: 'Evergreen Forest',
        43: 'Mixed Forest',
        52: 'Shrub/Scrub',
        81: 'Pasture/Hay',
    })
        
    nlcd_cmap = matplotlib.colors.ListedColormap(list(nlcd_color_map_values.values()))
    
    _nlcd_indices = np.array(list(_nlcd_labels.keys()),'d')
    nlcd_norm = matplotlib.colors.BoundaryNorm(list(_nlcd_labels.keys())+[93,], len(_nlcd_labels))
    nlcd_ticks = list(_nlcd_labels.keys()) + [93,]
    nlcd_labels = list(_nlcd_labels.values()) + ['',]
    
    
    
    # plot the resulting surface mesh
    fig = plt.figure(figsize=(10,6))
    ax = workflow.plot.get_ax('3d', fig, window=[0.0,0.2,1,0.8])
    cax = fig.add_axes([0.23,0.18,0.58,0.03])
    
    mp = ax.plot_trisurf(mesh_points3[:,0], mesh_points3[:,1], mesh_points3[:,2], triangles=mesh_tris, 
                    color=lc, cmap=nlcd_cmap, norm=nlcd_norm,
                    edgecolor=(0,0,0,.2), linewidth=0.5)
    
    
    
    #mp = workflow.plot.triangulation(mesh_points3, mesh_tris, crs, ax=ax, linewidth=0.5, color='elevation', edgecolor='gray')
    cb = fig.colorbar(mp, orientation='horizontal', cax=cax)
    
    cb.set_ticks(nlcd_ticks)
    cb.ax.set_xticklabels(nlcd_labels, rotation=45)
    
    #rivers_2 = [np.array(r.xy) for riv in rivers for r in riv]
    #rivers_e = [workflow.values_from_raster(r.transpose(), crs, dem, dem_profile) for r in rivers_2]
    #rivers_l3 = [np.array([i[0], i[1], j]).transpose() for i,j in zip(rivers_2, rivers_e)]
    #for r in rivers_l3:
    #    ax.plot(r[:,0]+1, r[:,1], r[:,2]+1, color='red', linewidth=3)
    
    #ax.set_aspect('equal', 'datalim')
    t = cb.ax.set_title('land cover type')
    ax.view_init(55,0)
    
    #fig.savefig('land_cover_b.png', dpi=600)
    plt.show()



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_17_0.png


.. code:: ipython3

    # plot the NLCD data
    fig = plt.figure(figsize=(16,16))
    ax = workflow.plot.get_ax(crs, fig)
    
    mp = workflow.plot.triangulation(mesh_points3, mesh_tris, crs, ax=ax, linewidth=0.5, color=lc,
                                     cmap=workflow.colors.nlcd_cmap, norm=workflow.colors.nlcd_norm)
    cb = fig.colorbar(mp)
    cb.set_ticks(workflow.colors.nlcd_ticks)
    cb.set_ticklabels(workflow.colors.nlcd_labels)
    plt.show()



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_18_0.png


Subsurface properties
---------------------

Get soil structure from SSURGO

.. code:: ipython3

    # download the NRCS soils data as shapes and convert it to raster
    import workflow.sources.manager_nrcs
    import matplotlib.cm
    
    # -- download the shapes
    target_bounds = coweeta_shape_split.exterior().bounds
    logging.info('target bounds: {}'.format(target_bounds))
    _, soil_survey = workflow.get_shapes(sources['soil type'], target_bounds, crs)
    logging.info('shape union bounds: {}'.format(shapely.ops.cascaded_union(soil_survey).bounds))
    soil_ids = np.array([shp.properties['id'] for shp in soil_survey], np.int32)
    
    # -- color a raster by the polygons (this makes identifying a triangle's value much more efficient)
    soil_color_raster, soil_color_profile, img_bounds = workflow.color_raster_from_shapes(target_bounds, 10,
                                                                    soil_survey, soil_ids, crs)
    
    # resample the raster to the triangles
    soil_color = workflow.values_from_raster(m2.centroids(), crs, soil_color_raster, soil_color_profile)


::

    2019-09-24 09:01:38,891 - root - INFO: target bounds: (273971.0911428, 3878839.6361173, 279140.9150949, 3883953.7853134)
    2019-09-24 09:01:38,892 - root - INFO: 
    2019-09-24 09:01:38,893 - root - INFO: Preprocessing Shapes
    2019-09-24 09:01:38,894 - root - INFO: ------------------------------
    2019-09-24 09:01:38,897 - root - INFO:   Using filename: /Users/uec/research/water/data/meshing/data/soil_survey/soil_survey_shape_-83.4790_35.0269_-83.4208_35.0743.gml
    2019-09-24 09:01:38,965 - root - INFO:   Found 460 shapes.
    2019-09-24 09:01:38,966 - root - INFO:   and crs: {'init': 'epsg:4326'}
    2019-09-24 09:01:39,740 - root - INFO: shape union bounds: (272780.3245135, 3877673.1650815, 281292.5015333, 3887703.7251369)
    2019-09-24 09:01:39,742 - root - INFO: Coloring shapes onto raster:
    2019-09-24 09:01:39,742 - root - INFO:   target_bounds = (273971.0911428, 3878839.6361173, 279140.9150949, 3883953.7853134)
    2019-09-24 09:01:39,743 - root - INFO:   img_bounds = [273966.0, 3878839.0, 279146.0, 3883959.0]
    2019-09-24 09:01:39,744 - root - INFO:   pixel_size = 10
    2019-09-24 09:01:39,745 - root - INFO:   width = 518, height = 512
    2019-09-24 09:01:39,746 - root - INFO:   and 43 independent colors of dtype int32


.. code:: ipython3

    # create a cmap for SSURGO
    import collections
    import matplotlib.colors
    ssurgo_ids = list(sorted(set(soil_color)))
    browns = matplotlib.cm.get_cmap('copper')
    ssurgo_colors = [browns( (i+1)/len(ssurgo_ids) ) for i in range(len(ssurgo_ids))]
    ssurgo_cmap = matplotlib.colors.ListedColormap(ssurgo_colors)
    
    ssurgo_ticks = ssurgo_ids + [ssurgo_ids[-1]+1,]
    ssurgo_norm = matplotlib.colors.BoundaryNorm(ssurgo_ticks, len(ssurgo_ids))
    ssurgo_labels = ssurgo_ids + ['',]
    
    # plot the soil on surface mesh
    fig = plt.figure(figsize=(10,6))
    ax = workflow.plot.get_ax('3d', fig, window=[0.0,0.2,1,0.8])
    cax = fig.add_axes([0.23,0.18,0.58,0.03])
    
    mp = ax.plot_trisurf(mesh_points3[:,0], mesh_points3[:,1], mesh_points3[:,2], triangles=mesh_tris, 
                    color=soil_color, cmap=ssurgo_cmap, norm=ssurgo_norm,
                    edgecolor=(0,0,0,.2), linewidth=0.5)
    cb = fig.colorbar(mp, orientation='horizontal', cax=cax)
    cb.set_ticks(list())
    #cb.set_ticks(ssurgo_ticks)
    #cb.ax.set_xticklabels(ssurgo_labels, rotation=45)
    cb.ax.set_title('soil type')
    ax.view_init(55,0)
    
    #fig.savefig('soil_type_b.png', dpi=600)
    plt.show()



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_21_0.png


.. code:: ipython3

    # plot the soil data
    fig = plt.figure(figsize=(16,16))
    ax = workflow.plot.get_ax(crs, fig)
    
    mp = workflow.plot.triangulation(mesh_points3, mesh_tris, crs, ax=ax, linewidth=0.5, 
                                     color=soil_color, cmap='gist_rainbow')
    #cb = fig.colorbar(mp)
    #cb.set_ticks(workflow.colors.nlcd_ticks)
    #cb.set_ticklabels(workflow.colors.nlcd_labels)
    plt.show()



.. image:: static/examples/mesh_coweeta_files/mesh_coweeta_22_0.png


Mesh extrusion
--------------

Given the surface mesh and material IDs on both the surface and
subsurface, we can extrude the surface mesh in the vertical to make a 3D
mesh.

.. code:: ipython3

    # layer extrusion
    # -- data structures needed for extrusion
    layer_types = []
    layer_data = []
    layer_ncells = []
    layer_mat_ids = []
    z = 0.0
    
    # -- soil layer --
    #  top 6 m
    #  5 cm initial top cell
    #  10 cells
    #  expanding dz, growing with depth
    ncells = 9
    dz = 0.05
    layer_dz = 4
    
    def telescope_factor(ncells, dz, layer_dz):
        """Calculates a telescoping factor"""
        if ncells * dz > layer_dz:
            raise ValueError(("Cannot telescope {} cells of thickness at least {} "+
                              "and reach a layer of thickness {}").format(ncells, dz, layer_dz))
    
        import scipy.optimize
        def seq(r):
            calc_layer_dz = dz * (1 - r**ncells)/(1-r)
            #print('tried: {} got: {}'.format(r, calc_layer_dz))
            return layer_dz - calc_layer_dz
        res = scipy.optimize.root_scalar(seq, x0=1.0001, x1=2)
        return res.root
    
    tele = telescope_factor(ncells, dz, layer_dz)
    logging.info("Got telescoping factor: {}".format(tele))
    for i in range(ncells):
        layer_types.append('constant')
        layer_data.append(dz)
        layer_ncells.append(1)
        layer_mat_ids.append(soil_color)
        z += dz
        dz *= tele
        
    # one more 2m layer makes 6m
    dz = 2.0
    layer_types.append('constant')
    layer_data.append(dz)
    layer_ncells.append(1)
    layer_mat_ids.append(soil_color)
    z += dz
    
    # -- geologic layer --
    # keep going for 2m cells until we hit the bottom of
    # the domain
    layer_types.append("constant")
    layer_data.append(40 - z) # depth of bottom of domain is 40 m
    layer_ncells.append(int(round(layer_data[-1] / dz)))
    layer_mat_ids.append(999*np.ones_like(soil_color))
    
    # print the summary
    workflow.extrude.Mesh3D.summarize_extrusion(layer_types, layer_data, layer_ncells, layer_mat_ids)


::

    2019-09-24 09:01:44,939 - root - INFO: Got telescoping factor: 1.515910144611108
    2019-09-24 09:01:44,941 - root - INFO: Cell summary:
    2019-09-24 09:01:44,942 - root - INFO: ------------------------------------------------------------
    2019-09-24 09:01:44,943 - root - INFO: l_id	| c_id	|mat_id	| dz		| z_top
    2019-09-24 09:01:44,944 - root - INFO: ------------------------------------------------------------
    2019-09-24 09:01:44,944 - root - INFO:  00 	| 00 	| 545854 	|   0.050000 	|   0.000000
    2019-09-24 09:01:44,945 - root - INFO:  01 	| 01 	| 545854 	|   0.075796 	|   0.050000
    2019-09-24 09:01:44,946 - root - INFO:  02 	| 02 	| 545854 	|   0.114899 	|   0.125796
    2019-09-24 09:01:44,947 - root - INFO:  03 	| 03 	| 545854 	|   0.174177 	|   0.240695
    2019-09-24 09:01:44,950 - root - INFO:  04 	| 04 	| 545854 	|   0.264036 	|   0.414872
    2019-09-24 09:01:44,952 - root - INFO:  05 	| 05 	| 545854 	|   0.400255 	|   0.678908
    2019-09-24 09:01:44,955 - root - INFO:  06 	| 06 	| 545854 	|   0.606751 	|   1.079163
    2019-09-24 09:01:44,956 - root - INFO:  07 	| 07 	| 545854 	|   0.919781 	|   1.685915
    2019-09-24 09:01:44,957 - root - INFO:  08 	| 08 	| 545854 	|   1.394305 	|   2.605695
    2019-09-24 09:01:44,958 - root - INFO:  09 	| 09 	| 545854 	|   2.000000 	|   4.000000
    2019-09-24 09:01:44,959 - root - INFO:  10 	| 10 	|  999 	|   2.000000 	|   6.000000
    2019-09-24 09:01:44,960 - root - INFO:  10 	| 11 	|  999 	|   2.000000 	|   8.000000
    2019-09-24 09:01:44,962 - root - INFO:  10 	| 12 	|  999 	|   2.000000 	|  10.000000
    2019-09-24 09:01:44,963 - root - INFO:  10 	| 13 	|  999 	|   2.000000 	|  12.000000
    2019-09-24 09:01:44,964 - root - INFO:  10 	| 14 	|  999 	|   2.000000 	|  14.000000
    2019-09-24 09:01:44,965 - root - INFO:  10 	| 15 	|  999 	|   2.000000 	|  16.000000
    2019-09-24 09:01:44,966 - root - INFO:  10 	| 16 	|  999 	|   2.000000 	|  18.000000
    2019-09-24 09:01:44,968 - root - INFO:  10 	| 17 	|  999 	|   2.000000 	|  20.000000
    2019-09-24 09:01:44,969 - root - INFO:  10 	| 18 	|  999 	|   2.000000 	|  22.000000
    2019-09-24 09:01:44,971 - root - INFO:  10 	| 19 	|  999 	|   2.000000 	|  24.000000
    2019-09-24 09:01:44,972 - root - INFO:  10 	| 20 	|  999 	|   2.000000 	|  26.000000
    2019-09-24 09:01:44,974 - root - INFO:  10 	| 21 	|  999 	|   2.000000 	|  28.000000
    2019-09-24 09:01:44,975 - root - INFO:  10 	| 22 	|  999 	|   2.000000 	|  30.000000
    2019-09-24 09:01:44,975 - root - INFO:  10 	| 23 	|  999 	|   2.000000 	|  32.000000
    2019-09-24 09:01:44,977 - root - INFO:  10 	| 24 	|  999 	|   2.000000 	|  34.000000
    2019-09-24 09:01:44,978 - root - INFO:  10 	| 25 	|  999 	|   2.000000 	|  36.000000
    2019-09-24 09:01:44,979 - root - INFO:  10 	| 26 	|  999 	|   2.000000 	|  38.000000


.. code:: ipython3

    # extrude
    m3 = workflow.extrude.Mesh3D.extruded_Mesh2D(m2, layer_types, layer_data, layer_ncells, layer_mat_ids)

.. code:: ipython3

    # add back on land cover side sets
    surf_ss = m3.side_sets[1]
    
    unique_lc = set(lc)
    print(unique_lc)
    for lc_id in unique_lc:
        where = np.where(lc == lc_id)[0]
        ss = workflow.extrude.SideSet(workflow.colors._nlcd_labels[lc_id], int(lc_id), 
                                      [surf_ss.elem_list[w] for w in where],
                                      [surf_ss.side_list[w] for w in where])        
        m3.side_sets.append(ss)


::

    {41, 42, 43, 81, 52, 21, 22, 23}


.. code:: ipython3

    # save to disk
    m3.write_exodus('coweeta_basin3.exo')


::

    
    You are using exodus.py v 1.13 (seacas-beta), a python wrapper of some of the exodus library.
    
    Copyright (c) 2013, 2014, 2015, 2016, 2017, 2018, 2019 National Technology &
    Engineering Solutions of Sandia, LLC (NTESS).  Under the terms of
    Contract DE-NA0003525 with NTESS, the U.S. Government retains certain
    rights in this software.
    
    Opening exodus file: coweeta_basin3.exo
    Closing exodus file: coweeta_basin3.exo

