#!/bin/bash
#
# This script simply launches a docker container containing Watershed Workflow.
#
if [[ -z "${WATERSHED_WORKFLOW_DATA_DIRECTORY}" ]]; then
    echo "Warning: data directory not set, so downloaded data will not be saved."
    echo "To keep data across sessions, set the env variable:"
    echo "  $> export WATERSHED_WORKFLOW_DATA_DIRECTORY=/path/to/data"
    echo "and rerun this script."
    echo ""
    while true; do
        printf '%s ' 'Continue without persistent data? [y/n]'
        read yn
        case $yn in
            [Yy]* ) docker run --rm -p 8888:8888 --name=watershed_workflow watershed_workflow; break;;
            [Nn]* ) exit 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
else
    docker run --rm -p 8888:8888 -v ${WATERSHED_WORKFLOW_DATA_DIRECTORY}:/home/jovyan/watershed_workflow_data:delegated --name=watershed_workflow watershed_workflow
fi
