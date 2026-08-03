Examples
==========

The best way to learn Watershed Workflow is to work through our examples.
Each notebook demonstrates a specific capability or data source, ranging from
quick standalone demos to complete end-to-end workflows.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/toy_problem_stream_aligned_mesh.ipynb
   examples/comparison_of_soil_structure.ipynb
   examples/get_AORC_met_data.ipynb
   examples/get_MODIS_LAI.ipynb
   examples/coweeta_stream_aligned_mesh.ipynb
   examples/coweeta_ats.ipynb


Example Descriptions
---------------------

:doc:`toy_problem_stream_aligned_mesh <examples/toy_problem_stream_aligned_mesh>`
  Introduces the mixed-element, stream-aligned meshing workflow using synthetic
  geometry (no network access needed).  Demonstrates uniform vs. distance-based
  resampling, triangulation, elevation assignment, and 3D mesh extrusion.
  A good first stop before tackling real-data workflows.

:doc:`comparison_of_soil_structure <examples/comparison_of_soil_structure>`
  Compares multiple subsurface soil and geologic datasets available for the
  continental US: GLHYMPS (global permeability/porosity formations), Pelletier
  depth-to-bedrock, SSURGO/NRCS surface soils, SoilGrids 2017 and 2.0 rasters,
  POLARIS 30-m hydraulic properties, and the HydroFrame CONUS2 indicator grid.
  Useful for understanding what products are available and how to choose between
  them for a given domain.

:doc:`get_AORC_met_data <examples/get_AORC_met_data>`
  Downloads hourly AORC meteorological forcing data (precipitation, radiation,
  temperature, humidity, wind) from NOAA's AWS Zarr archive.  Shows how to
  warp the data to a projected CRS, convert variables to ATS units, and
  construct a smoothed "typical year" for cyclic-steady-state spinup.

:doc:`get_MODIS_LAI <examples/get_MODIS_LAI>`
  Downloads MODIS LAI and land-cover data via the NASA AppEEARS API.
  Demonstrates how to compute a spatially-averaged, per-land-cover time series
  of leaf area index, remove leap days, and produce both a transient and a
  "typical year" LAI dataset for ATS.

:doc:`coweeta_stream_aligned_mesh <examples/coweeta_stream_aligned_mesh>`
  Builds a complete stream-aligned mesh for the Coweeta watershed (NC) using
  real data from NHD, 3DEP, NLCD, and SSURGO.  Covers watershed delineation,
  river network processing, surface meshing, DEM conditioning, and 3D extrusion.

:doc:`coweeta_ats <examples/coweeta_ats>`
  End-to-end workflow for generating all ATS simulation inputs for the Coweeta
  watershed.  Combines NHD hydrography, 3DEP elevation, NLCD land cover, MODIS
  LAI, GLHYMPS/Pelletier/SSURGO subsurface structure, and AORC/DayMet
  meteorology into a complete set of mesh, forcing, and XML input files.
