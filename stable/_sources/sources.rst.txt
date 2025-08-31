Data Sources
~~~~~~~~~~~~
Watershed Workflow stores a library of sources, which provide
functionality to access data as if it was local.  Given appropriate
bounds (spatial and/or temporal), the sources typically use REST-APIs
or other web-based services to locate, download, unzip, and file
datasets, which are then stored indefinitely for future use.  These
datasets are stored in a local data store whose location is specified
in the :ref:`Package Configuration` file.

The following sections lay out the sources subpackage, which is simply
a way of getting and working with default sources, and the broad
classes of sources frequently used in workflows.

.. autosummary::
   :nosignatures:

      watershed_workflow.sources.manager_shapefile.ManagerShapefile
      watershed_workflow.sources.manager_raster.ManagerRaster
      watershed_workflow.sources.manager_wbd.ManagerWBD
      watershed_workflow.sources.manager_nhd.ManagerNHD
      watershed_workflow.sources.manager_3dep.Manager3DEP
      watershed_workflow.sources.manager_nrcs.ManagerNRCS
      watershed_workflow.sources.manager_glhymps.ManagerGLHYMPS
      watershed_workflow.sources.manager_soilgrids_2017.ManagerSoilGrids2017
      watershed_workflow.sources.manager_pelletier_dtb.ManagerPelletierDTB
      watershed_workflow.sources.manager_nlcd.ManagerNLCD
      watershed_workflow.sources.manager_daymet.ManagerDaymet
      watershed_workflow.sources.manager_aorc.ManagerAORC
      watershed_workflow.sources.manager_modis_appeears.ManagerMODISAppEEARS


Source List
+++++++++++

Most users will access sources through the dictionaries of types of
sources created here.  In particular, `getDefaultSources()` will be
the standard starting point.

.. automodule:: watershed_workflow.sources
        :members:


Watershed boundaries and hydrography
++++++++++++++++++++++++++++++++++++

Watershed boundary datasets and hydrography datasets together form the
geographic structure of a watershed.  Watershed boundary datasets are
typically formed through analysis of elevation datasets, collecting
within the same watershed all parts of the land surface which drain to
a common river outlet.  Watersheds are hierarchical, ranging in scale
from small primary watersheds which drain into first order streams to
full river basins which drain into an ocean.  In the United States,
the USGS formally calculates hydrologic units and identifies them
using Hydrologic Unit Codes, or HUCs, which respect this hierarchy.
HUC 2 regions (e.g. the Upper Colorado River or the Tennessee River
Basin) are the largest in areal extent, while HUC 12s, or
sub-watersheds, are the smallest, representing on the order of 100
square kilometers.  Watershed Workflow uses HUCs as an organizing unit
for working with data, primarily because most datasets in the US are
organized by the HUC, but also because they form physically useful
domains for simulation.

Hydrography datasets provide surveys of river networks, which form the
drainage network of watersheds and are where most of the fast-time
scale dynamics occur.  Some hydrologic models (for instance river
routing models, dam operations management models, and many flood
models) directly use the river network as their simulation domain,
while others (for instance the class of integrated, distributed models
described here) can use the river network to refine meshes near the
rivers and therefore improve resolution where fast dynamics are
occuring.  Watershed boundary and Hydrography datasets are typically
available as GIS shapefiles, where each watershed boundary or reach is
represented as a shape.

Currently two ways of getting watershed boundaries are supported --
USGS HUC delineations and user-provided shape files.  Watershed
boundaries read from shapefiles can use the shapefile manager.

.. autoclass:: watershed_workflow.sources.manager_shapefile.ManagerShapefile
      :members:
               
.. autoclass:: watershed_workflow.sources.manager_wbd.ManagerWBD
      :members:

Getting the reaches used to construct rivers is done as either
shapefiles as above, or through NHD datasets, which include NHD Medium
Resolution, NHD Medium Resolution v2.1 (preferred) and NHD High Res.


.. autoclass:: watershed_workflow.sources.manager_nhd.ManagerNHD
      :members:

         
Digital Elevation Models
++++++++++++++++++++++++

For any distributed, integrated hydrologic model, elevation datasets
are critical.  These set the local spatial gradients that drive flow
in Richards and overload flow equations, and are necessary to form a
mesh, whether structured or unstructured, for simulation.  Elevation
datasets are typically stored as raster images.

Then workflows can query the raster for interpolated elevations at
points on a mesh, river, or other locations of interest.  Internally,
affine coordinate system transformations are hidden; the coordinate
system of the requested points are mapped to that of the raster and
interpolated.  By default, piecewise bilinear interpolation is used to
ensure that extremely high-resolution queries do not look
stairstepped; this improves mesh quality in meshes near the resolution
of the underlying elevation dataset.

.. autoclass:: watershed_workflow.sources.manager_raster.ManagerRaster
        :members:

.. autoclass:: watershed_workflow.sources.manager_3dep.Manager3DEP
        :members:

           
Land Cover
++++++++++

Land cover datasets set everything from impervious surfaces to plant
function and therefore evaportranspiration, and are used in some
integrated hydrologic models for a wide range of processes.  Land
cover is used to define a collection of indices on which mesh sets are
generated and then used to generate and affect processes and process
parameters.  Additionally, leaf area index (LAI) is used frequently in
determining potential evapotranspiration.

.. autoclass:: watershed_workflow.sources.manager_nlcd.ManagerNLCD
        :members:

.. autoclass:: watershed_workflow.sources.manager_modis_appeears.ManagerMODISAppEEARS
        :members:


Soil structure and properties
+++++++++++++++++++++++++++++

Soil structure and hydrologic properties (i.e. porosity, permeability,
water retention curves) are often derived from texture
parameterizations.  Similarly, depth to bedrock and other subsurface
data can be essential in these types of simulations.  Often these are
mapped into the simulation mesh.

.. autoclass:: watershed_workflow.sources.manager_nrcs.ManagerNRCS
        :members:

.. autoclass:: watershed_workflow.sources.manager_glhymps.ManagerGLHYMPS
        :members:

.. autoclass:: watershed_workflow.sources.manager_pelletier_dtb.ManagerPelletierDTB
        :members:

.. autoclass:: watershed_workflow.sources.manager_soilgrids_2017.ManagerSoilGrids2017
        :members:
           

Meteorology
+++++++++++

Meteorological data is used for forcing hydrologic models.  Note that
we keep DayMet here, but it is currently deprecated and unusable due
to the NASA DAAC THREDDS API being down indefinitely.  Use AORC
instead.

.. autoclass:: watershed_workflow.sources.manager_aorc.ManagerAORC
        :members:

.. autoclass:: watershed_workflow.sources.manager_daymet.ManagerDaymet
        :members:
                  
                  

