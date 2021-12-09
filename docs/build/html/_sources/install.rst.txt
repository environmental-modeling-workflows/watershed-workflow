Installation and Setup
=========================

All code in this package is pure python3, though it can be tricky to
get all of the dependencies to coexist because of their need for GIS
libraries.

Once the code is installed, typical usage builds on two directories:
the data library and the working directory.  Watershed Workflow
downloads a lot of datasets, and stores them in a common "data
library" for (re-)use by multiple workflows.  Any given workflow
consists of scripts or notebooks along with the synthesized data
products created by the workflow (meshes, forcing files, etc) -- these
live in the "working directory."  Additionally, for each workflow, a
configuration file is required.  This file is typically stored as
`~/.watershed_workflowrc` but may also live in the working directory.
This includes the path to the data library, along with other common
settings.

Summary of Dependencies and their Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard packages needed include `argparse` and `subprocess`, and for
testing, `pytest` and `dist_utils`.  Standard math packages include
`numpy`, `matplotlib`, and `scipy`.  Soil properties often come in
geodatabase files, which are best read with `pandas`.

GIS work is typically done using expensive/closed source and GUI-based
tools.  For this work, we instead build on extremely high-quality,
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
`netCDF4`.

**Optional:** Soil properties often come in percent silt/clay/sand,
yet hydrologic properties such as porosity, permeability, and van
Genuchten curves are the most frequently used in models.  Rosetta3 is
a tool providing pedotransfer functions to convert these properties
into the needed model parameters.  Rosetta packages are provided by
the `rosetta-soil` pip package.

**Optional:** Here we use `jupyter` lab/notebooks to provide examples
and illustrate usage of the package.  If you do not intend to use
jupyter, it is not necessary, adds a lot of extra packages, and can
safely be removed from the installation below.  If you do use
`jupyter`, you should also use `papermill`, which allows workflows to
be pipelined -- you develop a notebook, then use `papermill` to use
the notebook as a script.


Use with Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because of this complex set of processes, the simplest form of using
Watershed Workflow is through the provided Docker containers.  To do
this, simply install the Docker desktop app, then run the script
`bin/run_ww_lab.py`:

.. code-block:: console

   python bin/run_ww_lab.py /path/to/working/directory

Then, following the instructions on the terminal window, navigate your
browser to the provided URL.

Note that this script needs a `.watershed_workflowrc` file -- it will
search, in order of precedence, for:

 - A path provided on the command line via the `--rc` option,
 - `/path/to/working/directory/watershed_workflowrc`,
 - `/path/to/working/directory/.watershed_workflowrc`, or
 - `${HOME}/.watershed_workflowrc`

This file will then be stored at
`/path/to/working/directory/.docker_watershed_workflowrc` for use within the
container.

An example configure file is found at `watershed_workflowrc` in the
top directory of this repository.
   

Local Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to use Anaconda3 as a package manager, generating a
unique environment for use with this package, as this makes it fairly
easy to get all the required packages.

Download and install `Anaconda3
<https://www.anaconda.com/distribution/>`_.  Then create a new
environment that includes the required packages:

.. code-block:: console
    :caption: Packages for general users.
                
    conda env create -f environments/environment-OSX.yml
    conda activate watershed_workflow

Note OSX can be replaced by Linux where appropriate, and the
environments folder includes environment-\*.yml files for developers
and CI.  Windows users are recommended to use the docker container, or
to run the `environments/create_envs.py` script to generate
appropriate environments.

Developers should also install a few packages for building
documentation, testing, etc:

.. code-block:: console
    :caption: Packages for developers and building documentation

    conda env create -f environments/environment-OSX-dev.yml
    conda activate watershed_watershed_dev

The expectation is that you have installed jupyterlab and/or related
packages in your own base environment or elsewhere, and will simply
use the watershed_workflow environment as a kernel within Jupyter.

    
Check your python installation:

.. code-block:: console
                
     python -c 'import numpy, matplotlib, scipy, rasterio, fiona, shapely, cartopy, meshpy.triangle; print("SUCCESS")'

     
Installing ExodusII (optional)
--------------------------------

Clone the package from `source <https://github.com/gsjaardema/seacas>`_

Unfortunately this package does not do semantic versioned releases
except as part of the Trilinos project, and those releases are often
somewhat old.  Configuration is done through cmake -- an example use
is provided at `docker/configure-seacas.sh`.  Create a configure
script defining your compilers (likely clang if Mac and gcc if Linux)
and pointing to your SEACAS repo and Anaconda environment installation
of the required packages (which are all in your environment created
above).

Hopefully you are then able to add your installed SEACAS to your
PYTHONPATH and import the python wrappers:

.. code-block:: console
                
    export PYTHONPATH=${SEACAS_DIR}/lib
    python -c 'import exodus3; print("SUCCESS")'

Note if you have trouble doing this for a local build, try following
the more detailed formula in Stage 3 of the docker file,
`docker/User-Env.Dockerfile`.


Installing this package
--------------------------------------

Once you've got this environment set up, this package and the Rosetta
dependency are installed via:

.. code-block:: console

     cd /path/to/this/repository
     python -m pip install -e .


As in the docker case, a configuration file must be found.  By
default, installing this package via `setup.py` places a copy of
`watershed_workflowrc` in your home directory -- this can and should
be modified.


Run the test suite
~~~~~~~~~~~~~~~~~~

Given that you have activated your environment and successfully
install the above, the unit tests should all pass.  They are not
all fast -- some download files and may be internet-connection-speed
dependent.  You may be happy enough just running the core
library tests:

.. code-block:: console

   pytest watershed_workflow/test


but you can also run the entire suite:

.. code-block:: console

    pytest watershed_workflow                

