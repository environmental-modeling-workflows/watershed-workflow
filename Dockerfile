FROM continuumio/anaconda3

#LABEL Description="Watershed Workflow CI container based on Anaconda3"

ENV WATERSHED_WORKFLOW_DIR=/ww/watershed_workflow

# set the workdir as the user's home directory
WORKDIR /ww

# clone WW repo
RUN git clone -b setup_py https://github.com/ecoon/watershed-workflow.git watershed_workflow

# set the workdir as the newly cloned repo
WORKDIR /ww/watershed_workflow

# create an environment based on standard process -- yml not working?
#RUN conda create -n watershed_workflow -c conda-forge -c defaults python=3 ipython ipykernel jupyter notebook nb_conda_kernels nb_conda numpy matplotlib scipy pandas geopandas meshpy fiona rasterio shapely cartopy pyepsg descartes pyproj requests sortedcontainers attrs pip libarchive h5py netCDF4 pytest papermill

# create an environment based on yml file
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate watershed_workflow" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# get pip and install this package (which also installs rosetta)
RUN conda install pip && \
    python -m pip install -e .

# create a watershed_workflowrc that will be picked up
RUN cat watershed_workflowrc > .watershed_workflowrc



