# Does everything through running tests...
#
# Stage 1 -- setup base CI environment
#
FROM continuumio/miniconda3 AS ww_env_base_ci
LABEL Description="Base env for CI of Watershed Workflow"

ARG env_name=watershed_workflow_CI

WORKDIR /ww/tmp
COPY environments/create_envs.py /ww/tmp/create_envs.py 
RUN mkdir environments
RUN --mount=type=cache,target=/opt/conda/pkgs \
    python create_envs.py --CI --env=${env_name} --tools-env=watershed_workflow_tools Linux

#
# Stage 2 -- add in the pip
#
FROM ww_env_base_ci AS ww_env_pip_ci

WORKDIR /ww/tmp
COPY requirements.txt /ww/tmp/requirements.txt
RUN conda run -n ${env_name} python -m pip install -r requirements.txt


#
# Stage 3 -- add in Exodus
#
FROM ww_env_pip_ci AS ww_env_exodus_ci

ENV PATH=/opt/conda/envs/watershed_workflow_tools/bin:${PATH}
ENV SEACAS_DIR="/opt/conda/envs/${env_name}"
ENV CONDA_PREFIX="/opt/conda/envs/${env_name}"

# get the source
WORKDIR /opt/conda/envs/${env_name}/src
RUN git clone -b v2021-10-11 --depth=1 https://github.com/gsjaardema/seacas/ seacas

# configure
WORKDIR /ww/tmp
COPY docker/configure-seacas.sh /ww/tmp/configure-seacas.sh
RUN chmod +x /ww/tmp/configure-seacas.sh
WORKDIR /ww/tmp/seacas-build
RUN conda run -n watershed_workflow_tools ../configure-seacas.sh
RUN make -j install

# exodus installs its wrappers in an invalid place for python...
RUN mv /opt/conda/envs/${env_name}/lib/exodus3.py \
       /opt/conda/envs/${env_name}/lib/python3.9/site-packages/exodus3.py

#
# Stage 4 -- move the whole thing to make simpler containers
#
FROM ww_env_exodus_ci AS ww_env_ci_moved

# add conda-pack to the base env
RUN conda install -n base -c conda-forge --yes --freeze-installed conda-pack 
RUN conda-pack -n ${env_name} -o /tmp/env.tar && \
    mkdir /ww_env && cd /ww_env && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
RUN /ww_env/bin/conda-unpack


#
# Stage 5 -- copy over just what we need for CI
#
FROM ubuntu:20.04 AS ww_env_ci
COPY --from=ww_env_ci_moved /ww_env /ww_env
ENV PATH="/ww_env/bin:${PATH}"

# #
# # Stage 6 -- run tests!
# #
# FROM ww_env_ci AS ww_ci

# WORKDIR /ww

# # copy over source code
# COPY . /ww
# RUN python -m pip install -e .

# # create a watershed_workflowrc that will be picked up
# RUN cp watershed_workflowrc .watershed_workflowrc

# # run the tests
# RUN python -m pytest watershed_workflow/test/




