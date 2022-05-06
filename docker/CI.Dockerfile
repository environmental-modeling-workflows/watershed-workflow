#
# Stage 2 -- clone repo run
#
ARG GIT_BRANCH

FROM ecoon/watershed_workflow-ci_env:${GIT_BRANCH} AS watershed_workflow

WORKDIR /ww

# copy over source code
COPY . /ww
RUN python -m pip install -e .

# create a watershed_workflowrc that will be picked up
RUN cp watershed_workflowrc .watershed_workflowrc

# run the tests
RUN python -m pytest watershed_workflow/
# run the notebook example
RUN pytest --nbmake examples/mesh_coweeta.ipynb


