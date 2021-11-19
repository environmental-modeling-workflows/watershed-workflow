#!/bin/bash
docker build --progress=plain -f docker/Create-Envs.Dockerfile -t ww_linux_envs .
docker create -it --name ww_linux_envs_tmp ww_linux_envs 
docker cp ww_linux_envs_tmp:/ww/environments/environment-Linux.yml ./environments/environment-Linux.yml
docker cp ww_linux_envs_tmp:/ww/environments/environment-CI-Linux.yml ./environments/environment-CI-Linux.yml
docker cp ww_linux_envs_tmp:/ww/environments/environment-DEV-Linux.yml ./environments/environment-DEV-Linux.yml

# definitely want to remove the container
docker container rm -f ww_linux_envs_tmp

# also remove the image -- we have to rebuild to get new repos, and we
# only use this image once to generate the files
docker image rm -f ww_linux_envs
