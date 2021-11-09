Installation and Setup
=========================

All code in this package is pure python3, though the dependencies
(because of their need for GIS libraries) can be tricky to get all of
the dependencies to coexist.

Standard packages needed include `argparse` and `subprocess`, and for
testing, `pytest` and `dist_utils`.  Standard math packages include
`numpy`, `matplotlib`, and `scipy`.  Soil properties often come in
geodatabase files, which are best read with `pandas` and `geopandas`.

GIS work is typically done using expensive/closed source and GUI-based
tools.  For this work, we instead build on the extremely high-quality,
open-source packages for GIS in python: `fiona`, `rasterio`, `shapely`
and `cartopy`.

**Optional:** Mesh generation of 2D, "map view" surface meshes uses
the open source library Triangle, which can be wrapped for python
using `meshpy`.  This in turn depends upon boost python.  Optionally,
extrusion of this 2D mesh into a 3D mesh for use in integrated
hydrologic models requires a 3D mesh library ~~ we tend to use
ExodusII here (though it would be straightforward to extend this to
other packages such as VTK).  ExodusII, part of the `SEACAS
<https://github.com/gsjaardema/seacas>`_ suite of tools, provides a
python3 set of wrappers, but there is no current package, so this must
be installed separately.  See below.  Exodus, in turn, needs
`netCDF4`, which can be from conda.

**Optional:** Finally, soil properties often come in percent
silt/clay/sand, yet hydrologic properties such as porosity,
permeability, and van Genuchten curves are the most frequently used in
models.  Rosetta3 is a tool providing pedotransfer functions to
convert these properties into the needed model parameters.  Rosetta
packages are provided by the `rosetta-soil` pip package, which _is_
installed via the above setup.py.

**Optional:** Here we use `jupyter` notebooks to provide examples and
illustrate usage of the package.  If you do not intend to use jupyter,
it is not necessary, adds a lot of extra packages, and can safely be
removed from the installation below.  Similarly, if you use `jupyter`,
you should also use `papermill`, which allows workflows to be
pipelined -- you develop a notebook, then use `papermill` to use the
notebook as a script.


Use with Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
~~~~~~~~~~~~~~~~~~~

It is recommended to use Anaconda3 as a package manager, generating a
unique environment for use with this package, as this makes it fairly
easy to get all the required packages.

Download and install `Anaconda3
<https://www.anaconda.com/distribution/>`_.  Then create a new
environment that includes the required packages:

.. code-block:: console
    :caption: Packages for general users.
                
    conda create -n watershed_workflow -c conda-forge -c defaults python=3 ipython jupyterlab ipykernel nb_conda_kernels nb_conda numpy matplotlib scipy pandas meshpy fiona rasterio shapely cartopy descartes pyproj requests sortedcontainers attrs pip libarchive h5py netCDF4 pytest papermill 
    conda activate watershed_workflow

.. code-block:: console
    :caption: Packages for developers and building documentation

    conda create -n watershed_workflow_dev -c conda-forge -c defaults python=3 ipython ipykernel jupyter notebook nb_conda_kernels nb_conda numpy matplotlib scipy pandas geopandas meshpy fiona rasterio shapely cartopy descartes pyproj requests sortedcontainers attrs pip libarchive h5py netCDF4 pytest papermill sphinx numpydoc sphinx_rtd_theme nbsphinx
    conda activate watershed_watershed_dev

.. code-block:: console
   :caption: Packages for CI -- the minimum requirements to just run tests.

    conda create -n watershed_workflow_ci -c conda-forge -c defaults python=3 numpy matplotlib scipy pandas meshpy fiona rasterio shapely cartopy descartes pyproj requests sortedcontainers attrs pip libarchive h5py netCDF4 pytest

Note that, for OSX users, it is recommended you install `python.app`
as well, which ensures a framework python and makes matplotlib
plotting cleaner.  Solving this environment can take a few minutes.  A
current, complete, and tested conda environement for OSX is provided
in `environment.yml`, `environment_dev.yml`, and `environment_CI.yml`
respectively.  Note that the docker container is built on this
`environment_dev.yml`.

Check your python installation:

.. code-block:: console
                
     python -c 'import numpy, matplotlib, scipy, rasterio, fiona, shapely, cartopy, meshpy.triangle; print("SUCCESS")'

     
Installing ExodusII (optional)
--------------------------------

Clone the package from `source <https://github.com/gsjaardema/seacas>`_

Unfortunately this package does not do regular releases except as part
of the Trilinos project, and those releases are often somewhat old.
So we must build from master; the python3 wrappers are fairly new.
Edit the script at:

.. code-block:: console

      workflow_tpls/configure-seacas.sh

defining your compilers (likely clang if Mac and gcc if Linux) and
pointing to your SEACAS repo and Anaconda environment installation.

Hopefully you are then able to add your installed SEACAS to your
PYTHONPATH and import the python wrappers:

.. code-block:: console
                
    export PYTHONPATH=${SEACAS_DIR}/lib
    python -c 'import exodus3; print("SUCCESS")'


Installing this package
--------------------------------------

Once you've got this environment set up, this package and the Rosetta
dependency are installed via:

.. code-block:: console

     cd /path/to/this/repository
     python -m pip install -e .



Configuration
--------------------

Little is needed to set up the package, but likely you want to set a
data directory for storing all downloaded files.  This directory,
along with other configuration information (e.g. account info for
protected data archives) are stored in an rc file.  setup.py installs
a blank configure template in your home directory, so you should look
at and potentially edit this file: `~/.watershed_workflowrc`

Run the test suite
~~~~~~~~~~~~~~~

Given that you have activated your environment and successfully
install the above, the following tests should all pass.  They are not
all fast -- some download files and may be internet-connection-speed
dependent.  You may be happy enough just running the core
functionality tests:

.. code-block:: console

   pytest watershed_workflow/test


but you can also run the entire suite:

.. code-block:: console

    pytest watershed_workflow                

