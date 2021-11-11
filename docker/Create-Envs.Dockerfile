# Simply creates envs and dumps them to disk, so that I can save them
# for the repo.
#
# To be used by WW maintainer only...

FROM continuumio/miniconda3
WORKDIR /ww
COPY environments/create_envs.py /ww/create_envs.py
RUN mkdir environments
RUN --mount=type=cache,target=/opt/conda/pkgs \
    python create_envs.py --env-name=ww_ci --env-type=CI Linux

RUN --mount=type=cache,target=/opt/conda/pkgs \
    python create_envs.py --env-name=ww --env-type=STANDARD Linux

RUN --mount=type=cache,target=/opt/conda/pkgs \
    python create_envs.py --env-name=ww_dev --env-type=DEV Linux


