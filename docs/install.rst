Installation and Setup
=========================

All code is in python3, though the dependencies (because of their need
for GIS libraries) can be tricky to get right.  It is recommended to
use Anaconda3 as a package manager, generating a unique environment
for use with this package, as this makes it fairly easy to get all the
required packages.

Note that this package is not currently installed in a typical
pythononic way (i.e. setuptools), but instead expects you to simply
use it in place.  This will change shortly.  In the meantime, to
install this package, simply place it and its third party libraries
(TPLs) in your python path:

.. code-block:: console

    cd /path/to/repository
    export PYTHONPATH=`pwd`:`pwd`/workflow_tpls:${PYTHONPATH}

Dependencies
~~~~~~~~~~~~~~~~~~

Standard packages needed include `argparse` and `subprocess`, and for
testing, `pytest` and `dist_utils`.  Standard math packages include
`numpy`, `matplotlib`, and `scipy`.

GIS work is typically done using expensive/closed source and GUI-based
tools.  For this work, we instead build on the extremely high-quality,
open-source packages for GIS in python: `fiona`, `rasterio`, `shapely`
and `cartopy`.

Mesh generation of 2D, "map view" surface meshes uses the open source
library Triangle, which can be wrapped for python using `meshpy`.
This in turn depends upon boost python.  Optionally, extrusion of this
2D mesh into a 3D mesh for use in integrated hydrologic models
requires a 3D mesh library ~~ we tend to use ExodusII here (though it
would be straightforward to extend this to other packages such as
VTK).  ExodusII, part of the `SEACAS
<https://github.com/gsjaardema/seacas>`_ suite of tools, provides a
python3 set of wrappers, but there is no current package, so this must
be installed separately.  See below.

Recommended process
~~~~~~~~~~~~~~~~~~~

Download and install `Anaconda3
<https://www.anaconda.com/distribution/>`_.  Then create a new
environment that includes the required packages:

.. code-block:: console
    :caption: Packages for general users
                
    conda create -n ats_meshing -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs pytest
    conda activate ats_meshing

.. code-block:: console
    :caption: Packages for developers and documentation

    conda create -n ats_meshing -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs pytest sphinx=1.8.5 numpydoc sphinx_rtd_theme nbsphinx
    conda activate ats_meshing_dev

Note that, for OSX users, it is recommended you install `python.app`
as well, which ensures a framework python and makes matplotlib
plotting cleaner.  A current and complete conda environement for OSX
is provided in `workflow_tpls/environment.yml` and
`workflow_tpls/environment_dev.yml`.
    
Check your python installation:

.. code-block:: console
                
     python -c 'import numpy, matplotlib, scipy, rasterio, fiona, shapely, cartopy, meshpy.triangle; print("SUCCESS")'

     
Installing ExodusII (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the package from `source <https://github.com/gsjaardema/seacas>`_

Unfortunately this package does not do regular releases except as part
of the Trilinos project, and those releases are often somewhat old.
So we must build from master; the python3 wrappers are fairly new.
Edit the script at `workflow_tpls/configure-seacas.sh
<../master/workflow_tpls/configure-seacas.sh>`_, defining your
compilers (likely clang if Mac and gcc if Linux) and pointing to your
SEACAS repo and Anaconda environment installation.

Hopefully you are then able to add your installed SEACAS to your
PYTHONPATH and import the python wrappers:

.. code-block:: console
                
    export PYTHONPATH=${SEACAS_DIR}/lib
    python -c 'import exodus; print("SUCCESS")'

Sometimes this takes some fiddling with python versions -- if you keep
both python2 and python3 interpreters around, and both are available,
sometimes SEACAS's cmake seems to find the wrong ones.  A workaround
is to configure, then manually search for `python` in the
`CMakeCache.txt` file that results, and make sure it points to the
correct python3 binary and version number.  I have seen instances
where the binary is correct by the version number is still listed as
python2.  Manually changing it to the correct version number and then
calling `make install` again seems to fix the problem.

Note that the path to your SEACAS installation must also go in your
PYTHONPATH; `exodus.py` is located in the install directory's `lib`
subdirectory.



