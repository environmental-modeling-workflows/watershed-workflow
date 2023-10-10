# Simply creates envs and dumps them to disk, so that I can save them
# for the repo.
#
# To be used by WW maintainer only...

FROM condaforge/mambaforge:4.12.0-0
WORKDIR /ww
COPY environments/create_envs.py /ww/create_envs.py
RUN mkdir environments
ENV CONDA_BIN=mamba

RUN ${CONDA_BIN} install -n base -y -c conda-forge python=3

RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --manager=${CONDA_BIN} --env-type=CI --OS=Linux watershed_workflow_ci 

RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --manager=${CONDA_BIN} --env-type=STANDARD --OS=Linux watershed_workflow 

RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --manager=${CONDA_BIN} --env-type=DEV --OS=Linux watershed_workflow_dev


