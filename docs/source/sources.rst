Data Sources
~~~~~~~~~~~~
Watershed Workflow stores a library of sources, which provide
functionality to access data as if it was local.  Given appropriate
bounds (spatial and/or temporal), the sources typically use REST-APIs
or other web-based services to locate, download, unzip, and file
datasets, which are then stored indefinitely for future use.  These
datasets are stored in a local data store whose location is specified
in the :ref:`Package configuration` file.

The following sections lay out the source list, which is simply a way
of getting and working with default sources, and the broad classes of
sources frequently used in workflows.

.. autosummary::
   :nosignatures:

      watershed_workflow.sources.manager_nhd._FileManagerNHD
      watershed_workflow.sources.manager_ned.FileManagerNED
      watershed_workflow.sources.manager_nlcd.FileManagerNLCD
      watershed_workflow.sources.manager_modis_appeears.FileManagerMODISAppEEARS
      watershed_workflow.sources.manager_nrcs.FileManagerNRCS
      watershed_workflow.sources.manager_glhymps.FileManagerGLHYMPS
      watershed_workflow.sources.manager_soilgrids_2017.FileManagerSoilGrids2017
      watershed_workflow.sources.manager_daymet.FileManagerDaymet
      watershed_workflow.sources.manager_raster.FileManagerRaster
      watershed_workflow.sources.manager_shape.FileManagerShape


Source List
+++++++++++

.. automodule:: watershed_workflow.source_list
        :members:

Implementing a new data source for an existing type of data should
follow the API for existing implementations.  This makes it easy to
use it with the existing high level API.  See the
:ref:`Sources API` for how managers are used within the API.


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
boundaries read from shapefiles can use the :ref:`Generic shapefiles`
manager.

.. autoclass:: watershed_workflow.sources.manager_nhd._FileManagerNHD
      :members: get_huc, get_hucs, get_hydro
               
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

.. autoclass:: watershed_workflow.sources.manager_ned.FileManagerNED
        :members: get_raster

Land Cover
++++++++++

Land cover datasets set everything from impervious surfaces to plant
function and therefore evaportranspiration, and are used in some
integrated hydrologic models for a wide range of processes.  Land
cover is used to define a collection of indices on which mesh sets are
generated and then used to generate and affect processes and process
parameters.  Additionally, leaf area index (LAI) is used frequently in
determining potential evapotranspiration.

.. autoclass:: watershed_workflow.sources.manager_nlcd.FileManagerNLCD
        :members: get_raster               

.. autoclass:: watershed_workflow.sources.manager_modis_appeears.FileManagerMODISAppEEARS
        :members: get_data


Soil structure and properties
+++++++++++++++++++++++++++++

Soil structure and hydrologic properties (i.e. porosity, permeability,
water retention curves) are often derived from texture
parameterizations.  Similarly, depth to bedrock and other subsurface
data can be essential in these types of simulations.  Often these are
mapped into the simulation mesh.

.. autoclass:: watershed_workflow.sources.manager_nrcs.FileManagerNRCS
        :members: get_shapes, get_shapes_and_properties

.. autoclass:: watershed_workflow.sources.manager_glhymps.FileManagerGLHYMPS
        :members: get_shapes, get_shapes_and_properties

.. autoclass:: watershed_workflow.sources.manager_soilgrids_2017.FileManagerSoilGrids2017
        :members: get_raster
           

Meteorology
+++++++++++

Meteorological data is used for forcing hydrologic models.

.. autoclass:: watershed_workflow.sources.manager_daymet.FileManagerDaymet
        :members: get_data

                  
Generic Files
+++++++++++++

We also provide readers for user-provided rasters and shapefiles for
generic use.

.. autoclass:: watershed_workflow.sources.manager_raster.FileManagerRaster
        :members: get_raster               
   
.. autoclass:: watershed_workflow.sources.manager_shape.FileManagerShape
        :members: get_shape, get_shapes

