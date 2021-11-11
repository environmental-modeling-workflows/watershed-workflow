#
# Does everything to set up for User
#

#
# Stage 1 -- setup base CI environment
#
FROM jupyter/minimal-notebook AS ww_env_base_user
LABEL Description="Base env for CI of Watershed Workflow"

ARG env_name=watershed_workflow
ARG user=jovyan

WORKDIR /home/${user}/tmp
COPY environments/create_envs.py /home/${user}/tmp/create_envs.py 
RUN mkdir environments
RUN unset SUDO_UID
RUN --mount=type=cache,target=/opt/conda/pkgs \
    python create_envs.py --env-name=${env_name} \
        --with-tools-env --tools-env-name=watershed_workflow_tools \
        --with-user-env --user-env-name=default Linux

# dump the environments to disk so they can be recovered if desired
RUN mkdir /home/${user}/environments
RUN conda env export -n ${env_name} > /home/${user}/environments/environment-Linux.yml

# install the kernel on base's jupyterlab
USER root
RUN conda run -n ${env_name} python -m ipykernel install \
        --name watershed_workflow --display-name "Python3 (watershed_workflow)"
USER ${user}

#
# Stage 2 -- add in the pip
#
FROM ww_env_base_user AS ww_env_pip_user

WORKDIR /home/${user}/tmp
COPY requirements.txt /home/${user}/tmp/requirements.txt
RUN conda run -n ${env_name} python -m pip install -r requirements.txt


#
# Stage 3 -- add in Exodus
#
FROM ww_env_pip_user AS ww_env_user

ENV PATH="/opt/conda/envs/watershed_workflow_tools/bin:${PATH}"
ENV SEACAS_DIR="/opt/conda/envs/${env_name}"
ENV CONDA_PREFIX="/opt/conda/envs/${env_name}"

# get the source
WORKDIR /opt/conda/envs/${env_name}/src
RUN git clone -b v2021-10-11 --depth=1 https://github.com/gsjaardema/seacas/ seacas

# configure
WORKDIR /home/${user}/tmp
COPY --chown=${user}:${user} docker/configure-seacas.sh /home/${user}/tmp/configure-seacas.sh
RUN chmod +x  /home/${user}/tmp/configure-seacas.sh
WORKDIR /home/${user}/tmp/seacas-build
RUN conda run -n watershed_workflow_tools ../configure-seacas.sh
RUN make -j install

# exodus installs its wrappers in an invalid place for python...
RUN mv /opt/conda/envs/${env_name}/lib/exodus3.py \
       /opt/conda/envs/${env_name}/lib/python3.9/site-packages/exodus3.py


# clean up
RUN rm -rf /home/${user}/tmp

# unclear where this comes from, must be in the jupyter/minimal-notebook?
RUN rm -rf /home/${user}/work 

#
# Stage 6 -- run tests!
#
FROM ww_env_user AS ww_user

WORKDIR /home/${user}/watershed_workflow

# copy over source code
COPY  --chown=${user}:${user} . /home/${user}/watershed_workflow
RUN python -m pip install -e .

# run the tests
RUN conda run -n watershed_workflow python -m pytest watershed_workflow/test/

# Set up the workspace.
#
# create a watershed_workflowrc that will be picked up
RUN cp watershed_workflowrc /home/${user}/.watershed_workflowrc

# create a directory for data -- NOTE, the user should mount a
# persistent volume at this location!
RUN mkdir /home/${user}/data

# create a working directory -- NOTE, the user should mount a
# persistent volume at this location!
RUN mkdir /home/${user}/workdir
WORKDIR /home/${user}/workdir

# set the command
CMD [ "jupyter", "lab", "--port=8888" ]




