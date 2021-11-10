#
# Stage 2 -- clone repo run
#
FROM ecoon/watershed-workflow_env:latest AS watershed_workflow

WORKDIR /ww

# copy over source code
COPY . /ww
RUN python -m pip install -e .

# create a watershed_workflowrc that will be picked up
RUN cp watershed_workflowrc .watershed_workflowrc

# run the tests
RUN python -m pytest watershed_workflow/test/
