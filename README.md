Workflow for building HUC-based meshes in python.

Installation and Setup
========================
All code is in python3.  Recommended is to download anaconda3, as this
makes it fairly easy to get all the required packages.

Note that this package expects you to place the top-level directory in your PYTHONPATH:

    export PYTHONPATH=`pwd`:${PYTHONPATH}

Required Packages
-------------------

Standard math packages include numpy, matplotlib, and scipy.

GIS work uses packages: fiona, rasterio, and shapely.
The packages for fiona and rasterio from conda-forge seem to work better?

    conda install -c conda-forge fiona
    conda install -c conda-forge rasterio
    conda install shapely

Mesh generation uses Triangle, which can be wrapped for python using
meshpy.  This one is a little more difficult, as there seem to be no
good distributions of it?


Note this depends upon boost python, which seems to come by default
with anaconda?

For now, build from source:

     export ANACONDA_DIR=/path/to/your/anaconda
     export BOOST_ROOT=${ANACONDA_DIR}
     git clone https://github.com/ecoon/meshpy.git
     cd meshpy
     ./configure.py --python-exe=${ANACONDA_DIR}/bin/python --boost-inc-dir=${ANACONDA_DIR}/include  --boost-lib-dir=${ANACONDA_DIR}/lib --boost-python-libname=boost_python3  --disable-static --enable-shared --cxxflags=-stdlib=libc++ --ldflags=-stdlib=libc++
     make
     python setup.py install

Note to look closely at the result of the configure command -- it
errors frequently and has trouble finding Boost?

Check your python installation:

     $> ipython
     import numpy as np
     import matplotlib as plt
     import scipy
     import rasterio
     import fiona
     import shapely
     import meshpy.triangle


A first test
--------------

A good way to get started is to simply run the __main__ in workflow.triangulate.

    python workflow/triangulate.py

This downloads a HUC file (not small, so takes a minute) then
extracts, smooths, and triangulates some HUCs in it.