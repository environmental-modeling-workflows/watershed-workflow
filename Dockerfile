FROM continuumio/anaconda3

#LABEL Description="Watershed Workflow CI container based on Anaconda3"

#ENV WW_DIR=/ww/watershed_workflow

# set the workdir as the user's home directory
WORKDIR /ww

# clone WW repo
RUN git clone https://github.com/ecoon/watershed-workflow.git watershed_workflow

# set the workdir as the newly cloned repo
WORKDIR /ww/watershed_workflow

RUN echo "Where am i? ${PWD}"
RUN echo `ls ${PWD}/watershed_workflow_env.yml`

# create an environment based on the environment.yml file
RUN conda env create --name watershed_workflow -f watershed_workflow_env.yml

# activate the environment
RUN conda activate watershed_workflow

# get pip and install this package (which also installs rosetta)
RUN conda install pip && \
    python -m pip install -e .



