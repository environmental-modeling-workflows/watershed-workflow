#
# In addition to that from User-Env.Dockerfile, this adds layers for
# Amanzi-ATS source code (not an executable), ats_input_spec python
# package, and amanzi_xml python package.
#

#
# Stage 1 -- setup base CI environment
#
ARG USER_ENV_DOCKER_TAG
FROM ecoon/watershed_workflow:${USER_ENV_DOCKER_TAG}
LABEL Description="ATS layers on top of WW"

ARG env_name=watershed_workflow
ARG user=jovyan
ARG ats_version=1.6
ENV CONDA_BIN=mamba

# get Amanzi-ATS source
RUN mkdir /home/${user}/ats
WORKDIR /home/${user}/ats
RUN git clone -b amanzi-${ats_version} --recursive --depth=1 https://github.com/amanzi/amanzi amanzi-ats
WORKDIR /home/${user}/ats/amanzi-ats/src/physics/ats
RUN git checkout -b ats-${ats_version}
RUN git pull

# install amanzi_xml
WORKDIR /home/${user}/ats/amanzi-ats/tools/amanzi_xml
RUN ${CONDA_BIN} run -n ${env_name} python -m pip install -e .

# set up environment for ats
ENV AMANZI_SRC_DIR=/home/${user}/ats/amanzi-ats
ENV ATS_SRC_DIR=/home/${user}/ats/amanzi-ats/src/physics/ats
ENV PYTHONPATH=/home/${user}/ats/amanzi-ats/src/physics/ats/tools/utils

# get ats_input_spec and install 
WORKDIR /home/${user}/ats
RUN git clone --depth=1 https://github.com/ecoon/ats_input_spec ats_input_spec
WORKDIR /home/${user}/ats/ats_input_spec
RUN ${CONDA_BIN} run -n ${env_name} python -m pip install -e .

# leave us in the right spot
WORKDIR /home/${user}/workdir
