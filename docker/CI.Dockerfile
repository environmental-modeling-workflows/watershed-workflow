#
# Stage 2 -- clone repo run
#
ARG CI_ENV_DOCKER_TAG

FROM ecoon/watershed_workflow-ci_env:${CI_ENV_DOCKER_TAG} AS watershed_workflow

WORKDIR /ww

# copy over source code
COPY . /ww
RUN python -m pip install -e .

# create a watershed_workflowrc that will be picked up
RUN cp watershed_workflowrc .watershed_workflowrc
RUN cat .watershed_workflowrc
RUN echo "data_directory : /ww/examples/Coweeta/input_data" >> .watershed_workflowrc
RUN cat .watershed_workflowrc

# note it ALSO needs to be in the examples directory to be picked up there, and with the right path
RUN cp watershed_workflowrc examples/.watershed_workflowrc
RUN echo "data_directory : /ww/examples/Coweeta/input_data" >> examples/.watershed_workflowrc

# run the library tests
RUN conda run -n watershed_workflow_CI python -m pytest watershed_workflow/

# run the notebook examples
RUN conda run -n watershed_workflow_CI pytest --nbmake --nbmake-kernel=python3 examples/mesh_coweeta.ipynb
RUN conda run -n watershed_workflow_CI pytest --nbmake --nbmake-kernel=python3 examples/get_Daymet.ipynb
RUN conda run -n watershed_workflow_CI pytest --nbmake --nbmake-kernel=python3 examples/mesh_mixed_element_toy_problem.ipynb
RUN conda run -n watershed_workflow_CI pytest --nbmake --nbmake-kernel=python3 examples/mesh_mixed_element_coweeta.ipynb


