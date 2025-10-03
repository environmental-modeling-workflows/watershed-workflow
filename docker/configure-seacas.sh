# NOTE: this requires that you have defined CONDA_PREFIX to point to
# your Anaconda environment's installation, but this is set by default
# on conda activate.

CC=${COMPILERS}/bin/gcc
CXX=${COMPILERS}/bin/g++
FC=${COMPILERS}/bin/gfortran

SEACAS_SRC_DIR=${SEACAS_DIR}/src/seacas

echo "Building SEACAS:"
echo " at: ${SEACAS_DIR}"
echo " with conda: ${CONDA_PREFIX}"
echo " with CC: ${CC}"
echo " with CXX: ${CXX}"
echo " with FC: ${FC}"

cmake \
    -D Seacas_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D Seacas_ENABLE_SEACASExodus:BOOL=ON \
    -D CMAKE_INSTALL_PREFIX:PATH=${SEACAS_DIR} \
    -D CMAKE_BUILD_TYPE=Debug \
    -D BUILD_SHARED_LIBS:BOOL=ON \
    \
    -D CMAKE_CXX_COMPILER:FILEPATH=${CXX} \
    -D CMAKE_C_COMPILER:FILEPATH=${CC} \
    -D CMAKE_Fortran_COMPILER:FILEPATH=${FC} \
    -D CMAKE_AR:FILEPATH=${COMPILERS}/bin/gcc-ar \
    -D CMAKE_RANLIB:FILEPATH=${COMPILERS}/bin/gcc-ranlib \
    -D Seacas_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON \
    -D TPL_ENABLE_Netcdf:BOOL=ON \
    -D TPL_ENABLE_HDF5:BOOL=ON \
    -D TPL_ENABLE_Matio:BOOL=OFF \
    -D TPL_ENABLE_MPI=OFF \
    -D TPL_ENABLE_CGNS:BOOL=OFF \
    \
    -D Netcdf_LIBRARY_DIRS:PATH=${CONDA_PREFIX}/lib \
    -D Netcdf_INCLUDE_DIRS:PATH=${CONDA_PREFIX}/include \
    -D HDF5_ROOT:PATH=${CONDA_PREFIX} \
    -D HDF5_NO_SYSTEM_PATHS=ON \
${SEACAS_SRC_DIR}


