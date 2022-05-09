# Simply creates envs and dumps them to disk, so that I can save them
# for the repo.
#
# To be used by WW maintainer only...

FROM mambaorg/micromamba:0.23.0
WORKDIR /ww
COPY environments/create_envs.py /ww/create_envs.py
RUN mkdir environments

RUN micromamba install -n base -y -c conda-forge python=3

RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --manager=micromamba --env-name=watershed_workflow_ci --env-type=CI Linux

RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --manager=micromamba --env-name=watershed_workflow --env-type=STANDARD Linux

RUN --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/python create_envs.py --manager=mamba --env-name=watershed_workflow_dev --env-type=DEV Linux


