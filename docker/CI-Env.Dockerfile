# Does everything except running tests...
#
# Stage 1 -- setup base CI environment
#
FROM condaforge/miniforge3:latest AS ww_env_base_ci
LABEL Description="Base env for CI of Watershed Workflow"

ARG env_name=watershed_workflow_CI
ENV CONDA_BIN=mamba

# copy over create_envs
WORKDIR /ww/tmp
COPY environments/create_envs.py /ww/tmp/create_envs.py 
RUN mkdir environments

# Create the environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --OS=Linux --manager=${CONDA_BIN}  \
    --env-type=CI --with-tools-env=watershed_workflow_tools ${env_name}

ENV COMPILERS=/opt/conda/envs/watershed_workflow_tools 
ENV PATH="$COMPILERS/bin:$PATH"

#
# Stage 2 -- add in the pip
#
FROM ww_env_base_ci AS ww_env_pip_ci

WORKDIR /ww/tmp
COPY requirements.txt /ww/tmp/requirements.txt

RUN ${CONDA_BIN} run --name ${env_name} python -m pip install -r requirements.txt

#
# Stage 3 -- add in Exodus
#
FROM ww_env_pip_ci AS ww_env_exodus_ci

ENV PATH=/opt/conda/envs/watershed_workflow_tools/bin:${PATH}
ENV SEACAS_DIR="/opt/conda/envs/${env_name}"
ENV CONDA_PREFIX="/opt/conda/envs/${env_name}"

# get the source
WORKDIR /opt/conda/envs/${env_name}/src
RUN apt-get install git
RUN git clone -b v2021-10-11 --depth=1 https://github.com/gsjaardema/seacas/ seacas \
  && sed -i '/const int NC_SZIP_NN/ i\#ifdef NC_SZIP_NN\n#undef NC_SZIP_NN\n#endif' \
    /opt/conda/envs/${env_name}/src/seacas/packages/seacas/libraries/exodus/src/ex_utils.c

# configure
WORKDIR /ww/tmp
COPY docker/configure-seacas.sh /ww/tmp/configure-seacas.sh
RUN chmod +x /ww/tmp/configure-seacas.sh
WORKDIR /ww/tmp/seacas-build
RUN ${CONDA_BIN} run -n watershed_workflow_tools ../configure-seacas.sh
RUN make -j4 install

# exodus installs its wrappers in an invalid place for python...
# -- get and save the python version
RUN SITE_PACKAGES=$(conda run -n ${env_name} python -c "import site; print(site.getsitepackages()[0])") && \
    cp /opt/conda/envs/${env_name}/lib/exodus3.py ${SITE_PACKAGES}

