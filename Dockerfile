FROM continuumio/anaconda3

LABEL maintainer="Ethan Coon <coonet@ornl.gov>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y \
    emacs \
    g++ \
    git \
    libgdal-dev \
    libgl-dev \
    unzip

RUN conda install -c conda-forge boost

RUN pip --no-cache-dir install --upgrade \
    fiona \
    rasterio \
    shapely \
    pyproj

EXPOSE 7745

RUN mkdir ats
ENV HOME=/ats
ENV SHELL=/bin/bash
ENV ANACONDA_DIR=/opt/conda
ENV BOOST_ROOT=/opt/conda
ENV ATS_MESHING_DIR=/ats/ats-meshing

VOLUME /ats
WORKDIR /ats

RUN git clone https://github.com/ecoon/meshpy.git && \
    cd meshpy && \
    ./configure.py --python-exe=${ANACONDA_DIR}/bin/python --boost-inc-dir=${ANACONDA_DIR}/include  --boost-lib-dir=${ANACONDA_DIR}/lib --boost-python-libname=boost_python36  --disable-static --enable-shared && \
    python setup.py build && \
    python setup.py install && \
    cd

RUN git clone https://github.com/ecoon/ideal-octo-waffle.git 
ENV PYTHONPATH=.

CMD ["/bin/bash"]