#!/bin/bash

# how to (re-)create the local environments
ENV_NAME="watershed_workflow-`date +%F`"
python environments/create_envs.py --manager=mamba --OS=OSX --with-user-env=watershed_workflow_user --with-tools-env=watershed_workflow_tools ${ENV_NAME}
conda run -n ${ENV_NAME} python -m ipykernel install \
        --name ${ENV_NAME} --display-name "Python3 ${ENV_NAME}"
conda env export -n ${ENV_NAME} --no-builds > environments/environment-OSX.yml

CI_ENV_NAME="watershed_workflow_CI-`date +%F`"
python environments/create_envs.py --manager=mamba  --OS=OSX --env-type=CI ${CI_ENV_NAME}
conda run -n ${CI_ENV_NAME} python -m ipykernel install \
        --name ${CI_ENV_NAME} --display-name "Python3 ${CI_ENV_NAME}"
conda env export -n ${CI_ENV_NAME} --no-builds > environments/environment-CI-OSX.yml

DEV_ENV_NAME="watershed_workflow_DEV-`date +%F`"
python environments/create_envs.py --manager=mamba --OS=OSX --env-type=DEV ${DEV_ENV_NAME}
conda run -n ${DEV_ENV_NAME} python -m ipykernel install \
        --name ${DEV_ENV_NAME} --display-name "Python3 ${DEV_ENV_NAME}"
conda env export -n ${DEV_ENV_NAME} --no-builds > environments/environment-DEV-OSX.yml

