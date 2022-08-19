docker build --progress=plain -f docker/User-Env.Dockerfile -t ecoon/watershed_workflow:master . && \
docker build --progress=plain -f docker/ATS-User-Env.Dockerfile -t ecoon/watershed_workflow-ats:master .


