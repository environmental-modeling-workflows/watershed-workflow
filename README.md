# Watershed Workflow
[![Docs](https://img.shields.io/badge/docs-link-blue?style=for-the-badge)](https://ecoon.github.io/watershed-workflow/build/html/index.html)
[![Release](https://img.shields.io/github/v/release/ecoon/watershed-workflow?display_name=release&style=for-the-badge)](https://github.com/ecoon/watershed-workflow/releases/tag/watershed-workflow-1.1.0)

[![Build Status](https://img.shields.io/github/workflow/status/ecoon/watershed-workflow/CI%20to%20Docker%20Hub?label=tests&style=for-the-badge)](https://github.com/ecoon/watershed-workflow/actions)
[![Issues](https://img.shields.io/github/issues/ecoon/watershed-workflow?style=for-the-badge)](https://github.com/ecoon/watershed-workflow/issues)
[![Issues](https://img.shields.io/github/issues-pr/ecoon/watershed-workflow?style=for-the-badge)](https://github.com/ecoon/watershed-workflow/pulls)

![sample image](https://ecoon.github.io/watershed-workflow/build/html/_images/watershed_workflow.png "Example output of the Coweeta Hydrologic Lab watersheds across scales.")

[Please prefer to see our documentation.](https://ecoon.github.io/watershed-workflow/build/html/index.html)

Watershed Workflow is a python-based, open source chain of tools for generating meshes and other data inputs for hyper-resolution hydrology, anywhere in the (conterminous + Alaska?) US.  

Hyper-resolution hydrologic models have huge data requirements, thanks to their large extent (full river basins) and very high resolution (often ~10-100 meters).  Furthermore, most process-rich models of integrated, distributed hydrology at this scale require meshes that understand both surface land cover and subsurface structure.  Typical data needs for simulations such as these include:

* Watershed delineation (what is your domain?)
* Hydrography data (river network geometry, hydrographs for model evaluation)
* A digital elevation model (DEM) for surface topography
* Surface land use / land cover
* Subsurface soil types and properties
* Meterological data,

and more.

This package is a python library of tools and a set of jupyter notebooks for interacting with these types of data streams using free and open (both free as in freedom and free as in free beer) python and GIS libraries and data.  Critically, this package provides a way for **automatically and quickly** downloading, interpreting, and processing data needed to **generate a "first" hyper-resolution simulation on any watershed** in the conterminous United States (and most of Alaska/Hawaii/Puerto Rico).

To do this, this package provides tools to automate downloading a wide range of **open data streams,** including data from United States governmental agencies, including USGS, USDA, DOE, and others.  These data streams are then colocated on a mesh which is generated based on a watershed delineation and a river network, and that mesh is written in one of a variety of mesh formats for use in hyper-resolution simulation tools.

Note: Hypothetically, this package works on all of Linux, Mac, and Windows.  It has been tested on the first two, but not the third.

## Installation

[Visit our Installation documentation.](https://ecoon.github.io/watershed-workflow/build/html/install.html)

## For more...

* [See the documentation](https://ecoon.github.io/watershed-workflow)
* [See a gallery of data product images (work in progress)](https://ecoon.github.io/watershed-workflow/build/html/gallery.html)

## Funding, attribution, etc

This work was supported by multiple US Department of Energy projects, largely by Ethan Coon (coonet _at_ ornl _dot_ gov) at the Oak Ridge National Laboratory.  Use of this codebase in the academic literature should cite this repository (paper in preparation).

Collaborators and contributions are very welcome!
