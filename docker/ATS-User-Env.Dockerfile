#
# In addition to that from User-Env.Dockerfile, this adds layers for
# Amanzi-ATS source code (not an executable), ats_input_spec python
# package, and amanzi_xml python package.
#

#
# Stage 1 -- setup base CI environment
#
ARG GIT_BRANCH
ARG env_name=watershed_workflow
ARG user=jovyan
ENV CONDA_BIN=mamba

FROM ecoon/watershed_workflow:${GIT_BRANCH}
LABEL Description="ATS layers on top of WW"


RUN mkdir /home/${user}/ats
WORKDIR /home/${user}/ats

# get Amanzi-ATS source and install amanzi_xml
RUN git clone --recursive --depth=1 https://github.com/amanzi/amanzi amanzi-ats
WORKDIR /home/${user}/ats/amanzi-ats/tools/amanzi_xml
RUN ${CONDA_BIN} run -n ${env_name} python -m pip install -e .
ENV AMANZI_SRC_DIR=/home/${user}/ats/amanzi-ats
ENV ATS_SRC_DIR=/home/${user}/ats/amanzi-ats/src/physics/ats

ENV PYTHONPATH=/home/${user}/ats/amanzi-ats/src/physics/ats/tools/utils

# get ats_input_spec and install 
WORKDIR /home/${user}/ats/
RUN git clone --depth=1 https://github.com/ecoon/ats_input_spec ats_input_spec
WORKDIR /home/${user}/ats/ats_input_spec
RUN ${CONDA_BIN} run -n ${env_name} python -m pip install -e .


# leave us in the right spot
WORKDIR /home/${user}/workdir
