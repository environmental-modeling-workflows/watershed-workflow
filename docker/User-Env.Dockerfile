#
# Sets up baseline watershed workflow container for a user/jupyterlab
#

#
# Stage 1 -- setup base CI environment
#
FROM quay.io/jupyter/minimal-notebook AS ww_env_base_user
LABEL Description="Base env for CI of Watershed Workflow"

ARG env_name=watershed_workflow
ENV CONDA_BIN=mamba

# Fix mamba cache permissions
USER root
RUN mkdir -p /home/jovyan/.cache/mamba && \
    chown -R jovyan:users /home/jovyan/.cache
USER jovyan

WORKDIR ${HOME}/tmp
RUN mkdir ${HOME}/tmp/environments
COPY environments/create_envs.py environments/create_envs.py 

# # linux-arm64 does not have a pycares package -- build it locally
# # Detect architecture and build if needed
# # note, these CANNOT be mamba, so use conda explicitly
# RUN arch=$(uname -m) && \
#     if [ "$arch" = "aarch64" ]; then \
#         echo "Building pycares from source for aarch64..."; \
#         conda install -y conda-build; \
#         conda skeleton pypi pycares; \
#         awk '/^requirements:/ { \
#                 print; \
#                 print "  build:"; \
#                 print "    - python"; \
#                 print "    - pip"; \
#                 print "    - setuptools"; \
#                 print "    - cffi >=1.5.0"; \
#                 print "    - wheel"; \
#                 print "    - gcc"; \
#                 next \
#             } \
#             { print }' pycares/meta.yaml > meta.new.yaml; \
#         mv meta.new.yaml pycares/meta.yaml; \
#         conda build pycares; \
#     fi


# compilers
USER root
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends gcc gfortran g++ make cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y
USER jovyan

# Create the base environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python environments/create_envs.py --OS=Linux --manager=${CONDA_BIN}  \
    --env-type=STANDARD --use-local ${env_name}

# install the kernel on base's jupyterlab
USER root
RUN conda run -n ${env_name} python -m ipykernel install \
        --name watershed_workflow --display-name "Python3 (watershed_workflow)"
USER jovyan

#
# Stage 2 -- add in the pip
#
FROM ww_env_base_user AS ww_env_pip_user

WORKDIR ${HOME}/tmp
COPY requirements.txt ${HOME}/tmp/requirements.txt
RUN ${CONDA_BIN} run -n ${env_name} python -m pip install -r requirements.txt

RUN ${CONDA_BIN} run -n ${env_name} python -c 'import geopandas; import meshpy; meshpy.__file__'

#
# Stage 3 -- add in Exodus
#
FROM ww_env_pip_user AS ww_env_user

ENV SEACAS_DIR="/opt/conda/envs/${env_name}"
ENV CONDA_ENV_PREFIX="/opt/conda/envs/${env_name}"

# get the source
WORKDIR /opt/conda/envs/${env_name}/src
COPY environments/exodus_py.patch /opt/conda/envs/${env_name}/src/exodus_py.patch
RUN git clone -b v2021-10-11 --depth=1 https://github.com/gsjaardema/seacas/ seacas

WORKDIR /opt/conda/envs/${env_name}/src/seacas
RUN git apply ../exodus_py.patch
RUN sed -i '/const int NC_SZIP_NN =/d' packages/seacas/libraries/exodus/src/ex_utils.c

# configure
ENV COMPILERS=/usr

WORKDIR ${HOME}/tmp
COPY --chown=jovyan:jovyan docker/configure-seacas.sh ${HOME}/tmp/configure-seacas.sh
RUN chmod +x ${HOME}/tmp/configure-seacas.sh
WORKDIR ${HOME}/tmp/seacas-build
RUN ../configure-seacas.sh
RUN make -j4 install

# exodus installs its wrappers in an invalid place for python...
# -- get and save the python version
RUN SITE_PACKAGES=$(conda run -n ${env_name} python -c "import site; print(site.getsitepackages()[0])") && \
    cp /opt/conda/envs/${env_name}/lib/exodus3.py ${SITE_PACKAGES}

RUN ${CONDA_BIN} run -n ${env_name} python -c "import exodus3; print(exodus3.__file__)"

# clean up
RUN rm -rf ${HOME}/tmp

# unclear where this comes from, must be in the jupyter/minimal-notebook?
RUN rm -rf ${HOME}/work 

#
# Stage 6 -- copy over source and run tests
#
FROM ww_env_user AS ww_user


WORKDIR ${HOME}/watershed_workflow

# copy over source code
COPY --chown=jovyan:jovyan . ${HOME}/watershed_workflow
RUN ${CONDA_BIN} run -n watershed_workflow python -m pip install -e .

# change the default port to something not used by ATS container
ENV NOTEBOOK_ARGS="--NotebookApp.port=9999"

# Set up the workspace.
#
# create a watershed_workflowrc that will be picked up
RUN cp watershed_workflowrc ${HOME}/.watershed_workflowrc

# create a directory for data -- NOTE, the user should mount a
# persistent volume at this location!
RUN mkdir ${HOME}/data

# create a working directory -- NOTE, the user should mount a
# persistent volume at this location!
RUN mkdir ${HOME}/workdir
WORKDIR ${HOME}/workdir


