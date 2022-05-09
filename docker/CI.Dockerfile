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

# run the library tests
RUN python -m pytest watershed_workflow/test/

# run the test for those sources which can reasonably be expected to work and are most crucial
RUN python -m pytest watershed_workflow/sources/test/test_file_manager_daymet.py
RUN python -m pytest watershed_workflow/sources/test/test_file_manager_nhdplus.py
RUN python -m pytest watershed_workflow/sources/test/test_file_manager_shape.py

# run the notebook example
RUN pytest --nbmake --nbmake-kernel=default examples/mesh_coweeta.ipynb


