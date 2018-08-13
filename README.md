Workflow for building HUC-based meshes in python.

Installation and Setup
========================
All code is in python3.  Recommended is to download anaconda3, as this
makes it fairly easy to get all the required packages.

Note that this package expects you to place the top-level directory in your PYTHONPATH:

    export PYTHONPATH=`pwd`:${PYTHONPATH}

Required Packages
-------------------

Standard packages needed (should be included in all distributions?) include argparse and subprocess, and for testing, pytest and dist_utils.
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


Alternate Setup
==================
Alternative, use a docker container.

On a mac:
     docker build -t ats-meshing -f ./Dockerfile ./

     brew install socat
     nohup socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
     IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
     docker run -it --name container1 --net=host --env DISPLAY=${IP}:0  -v ~/codes/ats/ats-meshing/:/ats ats-meshing
     cd ats-meshing


A first example
----------------

A good way to get started is to simply run go:

    python3 bin/mesh_hucs.py -c 060102080102

This downloads a HUC file (not small, so takes a minute) then
extracts, smooths, and triangulates some HUCs in it.

A set of examples
-------------------

Basic triangulation of an existing HUC 12:

    python3 bin/mesh_hucs.py -c 060102080102

Triangulate a HUC 10, ensuring that HUC12 edges are included:

    python3 bin/mesh_hucs.py -c 0601020801

Triangulate with refinement, grading the max area to higher res near the river:

    python3 bin/mesh_hucs.py --refine-distance 0 1000 1000 10000 -c 060102080102

Triangulate a given user-provided shapefile, for instance a subwatershed from the Coweeta basin:

    python3 bin/mesh_shape.py --refine-distance 10 100 1000 1000 --hint=06 --center --shape-index=0 data/hydrologic_units/others/Coweeta/coweeta_subwatersheds.shp            
