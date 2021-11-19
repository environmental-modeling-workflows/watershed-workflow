#!/bin/bash
# copy back the environment.yml files from the docker container
docker create -it --name watershed_workflow_env-ci-linux bash
docker cp watershed_workflow_env-ci-linux:/environment.yml ./
docker rm -f watershed_workflow_env-ci-linux
