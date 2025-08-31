*******************   
Watershed Workflow
*******************

.. image:: _static/images/watershed_workflow.png

.. include:: ../AUTHORS.rst

Watershed Workflow is a python-based, open source chain of tools for
generating meshes and other data inputs for hyper-resolution
hydrology, anywhere in the US.

Fully distributed ydrologic models have huge data requirements, thanks
to their large extent (full river basins) and often very high
resolution (~1-500 meters).  Furthermore, most process-rich
models of integrated, distributed hydrology at this scale require
meshes that understand both surface land cover and subsurface
structure.  Typical data needs for simulations such as these include:

* Watershed delineation (what is your domain?)
* Hydrography data (river network geometry, hydrographs for model evaluation)
* A digital elevation model (DEM) for surface topography
* Surface land use / land cover
* Subsurface soil types and properties
* Meterological data,

and more.

This package is a python library of tools and a set of jupyter
notebooks for interacting with these types of data streams using free
and open (both free as in freedom and free as in free beer) python and
GIS libraries and data.  Critically, this package aims to provide a
way for **automatically and quickly** downloading, interpreting, and
processing data needed to **generate a "first" simulation on any
watershed** in the United States.  Some, but not all, of the data
products used here are global; the tools are directly applicable to
global datasets as well.

To do this, this package provides tools to automate querying and
downloading a wide range of **open datasets** from various data
portals, including data from United States governmental agencies such
as USGS, USDA, DOE, NASA, and others.  These datasets are then
colocated on a mesh which is generated based on a watershed
delineation and a river network, and that mesh is written in one of a
variety of mesh formats for use in simulation tools.


Workflows via Jupyter notebooks
------------------------------------

Workflows are the composition of partially automated steps to
accomplish a range of tasks.  Manual intervention is most commonly
needed when there are problems or inconsistencies with the datasets
themselves, or corner cases in data that these authors have not yet
found.  Combining automated and manual steps in a single workflow is
reasonably supported by Jupyter notebooks.

Note that the majority of code is NOT in notebooks.  Notebooks have
`all sorts of issues for software development, demonstration, and
reproducibility
<https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/>`_
but they are great for providing a template for **modifiable**
tutorials.


Acknowledgements, citation, etc
-----------------------------------

This work was supported by multiple US Department of Energy projects,
including ORNL LDRD funds, the ExaSheds project, and the IDEAS
project, and has been contributed to by authors at the Oak Ridge
National Laboratory, Pacific Northwest National Laboratory, and Utah
State University.  Use of this codebase in the academic literature
should cite:

* Coon, Ethan T., and Pin Shuai. "Watershed Workflow: A toolset for parameterizing data-intensive, integrated hydrologic models." Environmental Modelling & Software 157 (2022): 105502. : `https://doi.org/10.1016/j.envsoft.2022.105502 <https://doi.org/10.1016/j.envsoft.2022.105502>`_

Collaborators and contributions are very welcome!

