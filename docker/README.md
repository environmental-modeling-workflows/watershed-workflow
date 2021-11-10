There is a lot going on with our docker -- this helps to clarify what.

I have multiple ways of using WW:

1. local, non-docker environments
2. docker builds for CI, created via GitHub actions
3. docker builds for users, created locally
4. docker builds for users, created via GitHub actions

Local environments
------------------

Build via the create_env.py script, including the user env:

    python environments/create_env.py --user-env=default --user-env-extras OSX

Or without the user env:

    python environments/create_env.py OSX
    


Docker builds for CI
--------------------

This builds on many stages to keep things shared across as much as
possible.  The stages are:

1. Build an environment for the conda packages.
2. Layer in pip package requirements
3. Layer in an exodus build
4. Migrate this to /opt for smaller container -- this gets pushed to DockerHub
5. Layer in source code and run tests





Docker builds for users, created locally
----------------------------------------

1. Build an environment for the conda packages, also build user environment
2. Layer in pip package requirements
3. Layer in an exodus build
4. Layer in source code and run tests


Docker builds users, created via GitHub actions
-----------------------------------------------

This builds on many stages to keep things shared across as much as
possible.  The stages are:

1. Build an environment for the conda packages.
2. Layer in pip package requirements
3. Layer in an exodus build
5. Layer in source code and run tests



