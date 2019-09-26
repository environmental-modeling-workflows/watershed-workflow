.. Watershed Workflow documentation master file, created by
   sphinx-quickstart on Thu Sep 26 10:21:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

# Watershed Workflow

.. image:: _static/gallery/watershed_workflow.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:


# Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



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


## Installation and Setup

All code is in python3, though the dependencies (because of their need for GIS libraries) can be tricky to get right.  It is recommended to use Anaconda3 as a package manager, generating a unique environment for use with this package, as this makes it fairly easy to get all the required packages.

Note that this package is not currently installed in a typical pythononic way (i.e. setuptools), but instead expects you to simply use it in place.  This will change shortly.  In the meantime, to install this package, simply place it in your python path:

.. code-block:: console
                
    export PYTHONPATH=/path/to/watershed-workflow:${PYTHONPATH}

### Dependencies


Standard packages needed include `argparse` and `subprocess`, and for testing, `pytest` and `dist_utils`. 
 Standard math packages include `numpy`, `matplotlib`, and `scipy`.

GIS work is typically done using expensive/closed source and GUI-based tools.  For this work, we instead build on the extremely high-quality, open-source packages for GIS in python: `fiona`, `rasterio`, `shapely` and `cartopy`.

Mesh generation of 2D, "map view" surface meshes uses the open source library Triangle, which can be wrapped for python using `meshpy`.  This in turn depends upon boost python.  Optionally, extrusion of this 2D mesh into a 3D mesh for use in integrated hydrologic models requires a 3D mesh library -- we tend to use ExodusII here (though it would be straightforward to extend this to other packages such as VTK).  ExodusII, part of the `SEACAS <https://github.com/gsjaardema/seacas>`_ suite of tools, provides a python3 set of wrappers, but there is no current package, so this must be installed separately.  See below.

### Recommended process

Download and install `Anaconda3 <https://www.anaconda.com/distribution/>`_.  Then create a new environment that includes the required packages:

.. code-block:: console
                
    conda create -n ats_meshing -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs pytest
    conda activate ats_meshing


Check your python installation:

.. code-block:: console
                
     python -c 'import numpy, matplotlib, scipy, rasterio, fiona, shapely, cartopy, meshpy.triangle; print("SUCCESS")'

### Installing ExodusII (optional)


Clone the package from `source <https://github.com/gsjaardema/seacas>`_

Unfortunately this package does not do regular releases except as part of the Trilinos project, and those releases are often somewhat old.  Then, edit the script at `workflow_tpls/configure-seacas.sh` <../master/workflow_tpls/configure-seacas.sh>`_, defining your compilers (likely clang if Mac and gcc if Linux) and pointing to your SEACAS repo and Anaconda environment installation.

Hopefully you are then able to add your installed SEACAS to your PYTHONPATH and import the python wrappers:

.. code-block:: console
                
    export PYTHONPATH=${SEACAS_DIR}/lib
    python -c 'import exodus; print("SUCCESS")'

### Setting up your environment

Once all of the above work and are installed, setting up the environment from scratch consists of the following:

    conda activate ats_meshing
    export PYTHONPATH=/path/to/watershed-workflow:/path/to/SEACAS/install/lib:${PYTHONPATH}


## A first example

A good way to get started is to open your jupyter notebook and check out the main workflow:

.. code-block:: console
                
    jupyter notebook

And navigate to `examples/mesh_coweeta.ipynb <../master/examples/mesh_coweeta.ipynb>`_


## For more...

* See our `documentation <https://ecoon.github.io/watershed-workflow>`_
* See our `gallery <https://ecoon.github.io/watershed-workflow/gallery>`_

## Funding, attribution, etc

This work was supported by multiple US Department of Energy projects, largely by Ethan Coon (coonet _at_ ornl _dot_ gov) at the Oak Ridge National Laboratory.  Use of this codebase in the academic literature should cite this repository (paper in preparation).

Collaborators and contributions are very welcome!
