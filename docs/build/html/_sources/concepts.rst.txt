Workflow library Concepts
=========================

Package configuration
~~~~~~~~~~~~~~~~~~~~~

Watershed Workflow is configured through a limited set of parameters
specified in a file `".watershed_workflowrc`", located in the current
working directory or the user's home directory.  An example including
all defaults is shown in the top level directory as
`"watershed_workflowrc`".

Working with data sources
~~~~~~~~~~~~~~~~~~~~~~~~~

Watershed Workflow stores a library of managers, which provide
functionality to access data as if it was local.  Given appropriate
bounds (spatial and/or temporal), the managers typically use REST-APIs
or other web-based services to locate, download, unzip, and file
datasets, which are then stored indefinitely for future use.  These
datasets are stored in a local data store whose location is specified
in the `Package configuration`_ file.

Implementing a new data source for an existing type of data should
follow the API for existing implementations.  This makes it easy to
use it with the existing high level API.  See the
:ref:`Sources API` for how managers are used within the API.

shapes vs shapely
~~~~~~~~~~~~~~~~~

Watershed Workflow works with two different representations of shapes:
fiona's GeoJSON-like, python dictionary-based representation and
shapely's shape classes.  Each has advantages: the former is simple,
native python, and allows for data attributes to be associated with
the shape, while the latter allows for simpler geometric operations.
Furthermore, while fiona shapes can often be manipulated in-place
(their internal coordinates are most frequently lists, and therefore
mutable), shapely shapes cannot.

So while we initially keep shapes as fiona objects as long as
possible, and then attach their properties to the shape object when it
is created, as soon as that shapely shape is modified it loses its
properties.  Currently we have no solution to this, and shape
properties must be managed by the user external to the shape object.

In general, Watershed Workflow does not introduce its own shape
objects, and most of its data structures store shapely objects
internally.

Coordinate Reference Systems (CRS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coordinate Reference Systems are used to locate geographic positions.
These define a specific map projection, transforming 3D positions on
the Earth's surface to 2D coordinates.  Different projections can be
used to optimize for different things, but typically hydrologic
simulations work on equal area projections.  These projects maintain,
at least regionally, proportional areas for polygons, which is
critical for ensuring accurate water balances.

CRSs are specified by a dataset, and differ across datasets;
standardizing and managing these across the workflow is a necessary
technical detail.  That said, rarely does the user care what
coordinate system is being used, as long as it is appropriate for the
watershed in question.  Watershed Workflow aims to make using datasets
in different CRSs as streamlined as possible.  Typically, a workflow
will pick a CRS based upon either a default for the region or by
simply using the CRS of the shapefile that specifies the watershed
boundary.  This CRS is the passed into each function that acquires
more data, and that data's coordinates are changed to the CRS
requested.

Often it can be a good idea to work with a CRS that is used by a
raster dataset, for instance meterological data.  Interpolating from a
raster to a set of points (e.g. mesh cell centroids) is done by first
transforming those points into the CRS of the raster and then
interpolating.  While reprojecting rasters is possible (and supported
by rasterio), it involves some error and is tricky.  Working in a
raster's native CRS allows interpolation without reprojection, which
is especially useful for rasters that must be repeatedly interpolated
(i.e. meterological data or other time-dependent datasets).

See :ref:`CRS` for detailed documentation of working with CRSs.
