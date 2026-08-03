# Does everything except running tests...
#
# Stage 1 -- setup base CI environment
#
FROM condaforge/miniforge3:latest AS ww_env_base_ci
LABEL Description="Base env for CI of Watershed Workflow"

ARG env_name=watershed_workflow_CI
ENV CONDA_BIN=mamba

# figure out and print out conda platform info
ARG TARGETARCH
ARG TARGETOS

RUN echo "TARGETARCH=${TARGETARCH}" && \
    echo "TARGETOS=${TARGETOS}" && \
    uname -m && \
    conda info | grep platform

# copy over create_envs and requirements.txt (pip-installed by create_envs.py itself)
WORKDIR /ww/tmp
COPY environments/create_envs.py /ww/tmp/create_envs.py
COPY requirements.txt /ww/tmp/requirements.txt
RUN mkdir environments

# set compilers from watershed_workflow_tools environment -- this must happen
# BEFORE create_envs.py runs, since it also pip-installs requirements.txt
# (meshpy has no linux-aarch64 wheel and must build from source there, which
# needs g++/gcc from the tools env on PATH)
ENV COMPILERS=/opt/conda/envs/watershed_workflow_tools
ENV PATH="${COMPILERS}/bin:${PATH}"

# Create the environment (also pip-installs requirements.txt, constrained
# against the conda env's own package versions -- see create_envs.py)
RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --OS=Linux --manager=${CONDA_BIN}  \
    --env-type=CI --with-tools-env=watershed_workflow_tools ${env_name}

# test the environment
RUN ${CONDA_BIN} run --name ${env_name} python -c "import pymetis; import geopandas; import meshpy; import rosetta; import hf_hydrodata"

#
# Stage 2 -- add in Exodus
#
FROM ww_env_base_ci AS ww_env_exodus_ci

ENV SEACAS_DIR="/opt/conda/envs/${env_name}"
ENV CONDA_PREFIX="/opt/conda/envs/${env_name}"

# get the source
WORKDIR /opt/conda/envs/${env_name}/src
RUN git clone -b v2025-08-28 --depth=1 https://github.com/gsjaardema/seacas/ seacas

# apply the patch
COPY environments/exodus_py.patch /opt/conda/envs/${env_name}/src/exodus_py.patch
WORKDIR /opt/conda/envs/${env_name}/src/seacas
RUN git apply ../exodus_py.patch

# configure
WORKDIR /ww/tmp
COPY docker/configure-seacas.sh /ww/tmp/configure-seacas.sh
RUN chmod +x /ww/tmp/configure-seacas.sh
WORKDIR /ww/tmp/seacas-build
RUN ${CONDA_BIN} run -n watershed_workflow_CI ../configure-seacas.sh
RUN make -j4 install

# exodus installs its wrappers in an invalid place for python...
# -- get and save the python version
RUN SITE_PACKAGES=$(conda run -n ${env_name} python -c "import site; print(site.getsitepackages()[0])") && \
    cp /opt/conda/envs/${env_name}/lib/exodus3.py ${SITE_PACKAGES}

# test the environment
RUN ${CONDA_BIN} run --name ${env_name} python -c "import exodus3"
