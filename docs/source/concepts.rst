Key Concepts
============

.. _concepts_crs:

Coordinate Reference Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Coordinate Reference System (CRS) defines how geographic positions
are mapped to 2D coordinates.  Hydrologic workflows typically use
equal-area projections (e.g. Albers, EPSG:5070) so that polygon areas
and water balances are computed correctly.  Datasets come in many
different CRSs; Watershed Workflow converts all data into a single
working CRS that the user chooses at the start of a workflow —
typically the CRS of the meteorological data to be used for forcing,
or one of the module-level constants such as
``watershed_workflow.crs.default_crs``.

Users should create and convert CRS objects using the ``from_*`` and ``to_*``
functions in :mod:`watershed_workflow.crs` rather than constructing CRS objects
directly.  This keeps user code independent of the underlying representation,
which is currently ``pyproj.CRS``.

See :ref:`CRS` for the full API.


.. _concepts_sources:

Source Managers
~~~~~~~~~~~~~~~

Source managers in :mod:`watershed_workflow.sources` handle the full lifecycle
of geospatial and temporal datasets: discovering, downloading, caching, and
loading them.  Given a spatial region (and optionally a time range), a manager
fetches data from a REST API or web service, writes it to a local cache
directory, and returns it as a standard Python object (GeoDataFrame, xarray
Dataset, etc.).  Subsequent requests for the same region are served from the
cache without network access.

This cache is stored locally in a "data library" that is shared across
all workflows on a given machine.  In this way nested watersheds can
share the same underlying data downloads, reducing disk space and time
spent waiting for data.

See :ref:`package configuration` for how to set your data library, and
:ref:`data-sources` for the full list of sources.


.. _concepts_hydro:

Hydrology Objects
~~~~~~~~~~~~~~~~~

Two data structures represent the geographic skeleton of a watershed:

:class:`~watershed_workflow.hydro.river.River` is a tree whose
nodes are individual stream reaches.  Children flow into their parent, so the
root is the outlet.  Each node stores reach geometry as a Shapely LineString
along with any attributes from the source dataset (e.g. NHDPlus VAAs).  The
tree structure supports traversal, pruning by stream order, simplification, and
snapping to watershed boundaries.  Note that the tree structure means Watershed
Workflow is restricted to *dendritic* (non-braided) river networks.

:class:`~watershed_workflow.hydro.watershed.Watershed` is a
collection of polygons that partition the simulation domain into sub-catchments
— for example, HUC-12 units within a HUC-8, or differential contributing areas
between stream gauges.  The watershed boundary is represented as a set of
shared linestrings so that topological relationships (shared edges, junction
points) are maintained and can be manipulated cleanly.

These two objects are constructed by the source managers and then prepared
for meshing through a **simplification and snapping** step.  Raw hydrography
datasets contain more vertices than a mesh needs and may have small
topological gaps between river endpoints and watershed boundaries.
:func:`watershed_workflow.simplify` resamples reach and boundary linestrings
to a target segment length, removes near-duplicate points, smooths sharp
angles, and then snaps river endpoints onto watershed boundaries and inserts
junction points so that the river network and watershed boundary are
topologically consistent — a requirement for well-formed mesh generation.
Simplification tolerances directly control mesh density, so choosing them
carefully is important.

See :ref:`hydrology` for the full API.


.. _concepts_mesh:

Meshes and Discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the key roles of Watershed Workflow is to create a mesh that is grounded
in site-specific hydrologic data — elevation, subsurface structure, and surface
land cover.  Each of the concepts below represents an important decision on the
way to creating a mesh suitable for hydrologic modelling.

Mesh Structure
^^^^^^^^^^^^^^

A **mesh** is the spatial discretization of the simulation domain.
:class:`~watershed_workflow.mesh.mesh.Mesh2D` stores vertices, polygonal
cells, and their connectivity.  :class:`~watershed_workflow.mesh.mesh.Mesh3D`
extends a 2D mesh vertically into a column structure suitable for subsurface
models.

Watershed Workflow generates *stream-aligned* meshes [Rathore2024]_: long
quadrilateral elements track the bank-full river corridor, higher-order
polygonal elements (pentagons or more) handle stream junctions, and
unstructured triangles fill the remainder of the watershed.  This design
ensures that the river channel is resolved by mesh edges rather than
interpolated across cells, which is critical for accurate surface–subsurface
exchange.

What makes a good mesh for hydrology:

* **Stream-aligned** — quad elements follow the river corridor; the channel
  boundary is a mesh edge, not a cut through a cell.
* **Graded** — cells are fine near rivers (where dynamics are fast) and coarser
  away from them.
* **Well-conditioned** — triangles should be as close to equilateral as
  possible; very obtuse or very acute triangles degrade numerical accuracy and
  solver performance.
* **Pit-free** — the surface elevation field must drain correctly; local minima
  (pits) that would trap water unphysically are filled or conditioned before
  use.

Meshes carry **data** as cell- or node-centered fields (e.g. elevation, soil
properties, land cover index).  **Regions** are named subsets of cells or
faces used to apply boundary conditions, assign material properties, or
instrument the model to form simulated observations for evaluation against
field measurements.  :class:`~watershed_workflow.mesh.mesh.LabeledSet` and
:class:`~watershed_workflow.mesh.mesh.SideSet` are the discrete mesh
representation of a region — a labeled set of cell/node indices or face
indices respectively.

.. [Rathore2024] Rathore, S., Coon, E., et al. "Stream-aligned meshes for
   integrated hydrologic simulation." *Computers & Geosciences* (2024).

Hydrologic Conditioning
^^^^^^^^^^^^^^^^^^^^^^^

A mesh elevation field derived directly from a DEM almost always contains
**pits** — cells lower than all their neighbours — that would trap water and
prevent drainage.  Some pits are real (lakes, reservoirs) and should be
preserved; others are numerical artifacts that must be removed.
:func:`~watershed_workflow.mesh.condition.conditionMesh` detects pits and
raises artifact cells just enough to ensure downslope connectivity, while
leaving intentional depressions intact.

The river corridor requires additional conditioning beyond pit-filling.  DEM
resolution, positional errors in the NHDPlus flowlines, and features such as
culverts or low-head dams can create artificial obstructions in the river
channel.  River mesh conditioning adjusts the elevation profile of the
stream-aligned quad elements to be monotonically decreasing from headwater to
outlet, ensuring that the modelled channel drains correctly regardless of DEM
artifacts.  Conditioning the river corridor and the surrounding triangulated
domain are treated as separate steps because the quad elements in the corridor
must never be filled as pits.

Land Cover
^^^^^^^^^^

Land cover — the vegetation, impervious surfaces, and open water that cover the
land surface — controls many of the key fluxes in a hydrologic model.
Evapotranspiration rates, interception, root depth, and leaf area index (LAI)
all vary by vegetation type.  Surface roughness (Manning's coefficient) affects
overland flow velocity and partitioning between fast and slow runoff pathways.
Impervious fraction determines how much precipitation infiltrates versus runs
off directly.

In Watershed Workflow, land cover is typically sourced from NLCD (30 m,
categorical) and used to define mesh regions corresponding to vegetation or
land-use classes.  Time-varying LAI, often derived from MODIS, can be
crosswalked from MODIS land cover classes to NLCD classes and attached to the
mesh as a time series forcing.  See :mod:`watershed_workflow.properties.land_cover`
and :mod:`watershed_workflow.sources` for the relevant managers and
crosswalking utilities.

Subsurface Layering
^^^^^^^^^^^^^^^^^^^

Integrated hydrologic models resolve both the land surface and the subsurface.
:class:`~watershed_workflow.mesh.mesh.Mesh3D` is constructed by extruding a
2D surface mesh vertically into a stack of layers.  Layer thicknesses are
typically prescribed as a geometric progression — thin near the surface where
gradients are steep, coarser at depth — and are optimised to span from a
minimum cell thickness (e.g. 5 cm) to several metres over a target total
depth.

Material properties in the subsurface vary with depth.  A common layering
strategy assigns soil properties (from NRCS or SoilGrids) in the shallow
column, geologic properties (from GLHYMPS) below, with depth-to-bedrock data
(Pelletier DTB) determining the transition.  Each vertical layer is assigned a
material identifier that the simulator uses to look up constitutive relations.
Regions defined on the 2D surface mesh are extruded alongside the geometry, so
that boundary conditions and observations defined on the surface carry through
to the 3D mesh.

See :ref:`mesh` for the full API.


.. _concepts_data:

Data Utilities
~~~~~~~~~~~~~~

:mod:`watershed_workflow.utils` provides functions for working with
geospatial and temporal data after it has been downloaded.  Common operations
include interpolating raster values at a set of points (e.g. mesh cell
centroids or river coordinates), rasterizing vector datasets onto a grid,
aggregating gridded data over polygons, and resampling time series.  These
functions operate on standard types — NumPy arrays, Pandas DataFrames, xarray
Datasets, and GeoPandas GeoDataFrames — and handle coordinate system
conversions internally.

Coordinate transformations between CRSs are also provided by
:mod:`watershed_workflow.utils`, which wraps ``pyproj`` for points,
arrays, and Shapely geometries.

See :ref:`utilities` for the full API.
