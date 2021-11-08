FROM continuumio/miniconda3 AS environ
#
# Stage 1 -- setup environment
#
LABEL Description="Watershed Workflow CI container for dependencies/TPLs and environment setup"

# set the workdir as the user's home directory
WORKDIR /ww

COPY environment.yml /ww/environment.yml

# create an environment based on standard process -- yml not working?
#RUN conda create -n watershed_workflow -c conda-forge -c defaults python=3 ipython ipykernel jupyter notebook nb_conda_kernels nb_conda numpy matplotlib scipy pandas geopandas meshpy fiona rasterio shapely cartopy pyepsg descartes pyproj requests sortedcontainers attrs pip libarchive h5py netCDF4 pytest papermill

# create an environment based on yml file
RUN conda env create -f environment.yml && \
    conda clean -afy

# add extras pip and conda-pack
RUN conda install -c conda-forge --yes --freeze-installed conda-pack pip

# pack the environment into a new directory
RUN conda-pack -n watershed_workflow -o /tmp/env.tar && \
    mkdir /ww_env && cd /ww_env && tar xf /tmp/env.tar && \
    rm /tmp/env.tar 

# We've put env in same path it'll be in final image,
# so now fix up paths:
RUN /ww_env/bin/conda-unpack


#
# Stage 2 -- clone repo run
#
FROM ubuntu:20.04 AS watershed_workflow

RUN apt-get -y update
RUN apt-get -y install git

COPY --from=environ /opt/conda/envs/watershed_workflow /opt
ENV PATH="/opt/bin:${PATH}"
ENV WATERSHED_WORKFLOW_DIR=/ww/watershed_workflow

# clone WW repo
#
# NOTE: fixme -- this include branch info!
WORKDIR /ww
RUN git clone -b setup_py https://github.com/ecoon/watershed-workflow.git watershed_workflow

# set the workdir as the newly cloned repo
WORKDIR /ww/watershed_workflow

# get pip and install this package (which also installs rosetta)
RUN python -m pip install -e .

# create a watershed_workflowrc that will be picked up
RUN cp watershed_workflowrc .watershed_workflowrc

RUN python -m pytest watershed_workflow/test/

CMD [ "python", "-m", "pytest", "watershed_workflow/test/" ]
