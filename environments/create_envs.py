"""A script to help (re-)generate conda environments for Watershed Workflow."""

# packages required for running WW
PACKAGES_BASE=['python=3',
              'numpy',
              'matplotlib',
              'scipy',
              'pandas',
              'meshpy',
              'fiona',
              'rasterio',
              'shapely',
              'cartopy',
              'descartes',
              'pyproj',
              'requests',
              'sortedcontainers',
              'attrs',
              'pip',
              'libarchive',
              'h5py',
              'netCDF4',
              'pytest',
               'cmake',
              ]

# extra packages needed in the WW env when building for a user
PACKAGES_EXTRAS=['ipykernel',]

# a user environment that can use the WW env
PACKAGES_USER_BASE=['ipython',
               'jupyterlab',
               'ipykernel',
               'nb_conda',
               'nb_conda_kernels',
               'papermill',
               'sphinx',
               'numpydoc',
               'sphinx_rtd_theme',
               'nbsphinx',
               ]

# my personal extras that go in my user env
PACKAGES_USER_EXTRAS=['numpy',
                      'matplotlib',
                      'gitpython',
                      'pandas',
                      'h5py',
                      'netCDF4',
                      'scipy',
                      ]

PACKAGES_TOOLS=['cmake',
                'make',
                ]

# channels needed to find these packages
CHANNELS=['conda-forge',
          'defaults',
          ]


DOCKER_TEMPLATE = \
"""# Docker container to create the env
FROM continuumio/miniconda3
RUN conda create -n {env_name_arg} {channels_arg} {packages_arg}
RUN conda env export -n {env_name_arg} --no-builds > environment.yml
"""

import datetime
import subprocess
import os

def date_str():
    """Gets the date in a preferred format for writing env names"""
    return datetime.datetime.today().strftime('%Y-%m-%d')

def get_env_name(name):
    """Standard format for the name of an environment."""
    return name+'-'+date_str()

def get_env_prefix(env_type=None):
    if env_type is None or env_type == 'USER':
        return 'watershed_workflow'
    elif env_type == 'CI':
        return 'watershed_workflow_CI'
    elif env_type == 'TOOLS':
        return 'watershed_workflow_tools'
    else:
        raise ValueError(f'Unknown env type {env_type}')

def get_packages(env_type=None, include_my_extras=False, os_name=None):
    """Get the list of packages to build."""
    if env_type == 'USER':
        packages = PACKAGES_USER_BASE.copy()
        if include_my_extras:
            packages.extend(PACKAGES_USER_EXTRAS)
    elif env_type == 'TOOLS':
        packages = PACKAGES_TOOLS.copy()
        if os_name == 'OSX':
            packages.append('clang_osx-64')
            packages.append('clangxx_osx-64')
            packages.append('gfortran_osx-64')
        elif os_name == 'Linux':
            packages.append('gcc_linux-64')
            packages.append('gxx_linux-64')
            packages.append('gfortran_linux-64')
    else:
        packages = PACKAGES_BASE.copy()
        if env_type != 'CI':
            packages.extend(PACKAGES_EXTRAS)
    return packages

def dump_env_local(env_type, os_name, env_name, env_filename=None):
    """Dumps the env into an appropriate filename, attempting to 'clean it up'.

    This would be improved by a solution to:
      https://github.com/conda/conda/issues/9399
    but instead we manually hack to remove some known OS-specific libraries.
    """
    if env_filename is None:
        if env_type is None:
            env_filename = os.path.join('environments', f'environment-{os_name}.yml')
        else:
            env_filename = os.path.join('environments', f'environment-{env_type}-{os_name}.yml')

    if env_name is None:
        env_name = get_env_name(get_env_prefix(env_type))

    result = subprocess.run(['conda','env','export','--no-builds','-n',env_name],
                            check=True, capture_output=True)

    # try to strip matches that break OSX dependent code...
    stdout = result.stdout.decode('utf-8').split('\n')
    lines = []
    # the first line includes the name -- strip the date
    lines.append(stdout[0].split('-')[0])
    lines.extend(stdout[1:])

    with open(env_filename, 'w') as fid:
        fid.write('\n'.join(lines))

def create_env_local(env_type, os_name, packages, env_name=None):
    """Creates the environment locally."""
    env_prefix = get_env_prefix(env_type)

    # build up the conda env create command
    cmd = ['conda', 'create', '--yes']
    cmd.append('-n')
    if env_name is None:
        env_name = get_env_name(env_prefix)
    cmd.append(env_name)

    for channel in CHANNELS:
        cmd.append('-c')
        cmd.append(channel)

    cmd.extend(packages)

    # call conda env create
    subprocess.run(cmd, check=True)
    return env_name

def create_and_dump_env_local(env_type, os_name, packages, env_name=None, dump_only=False):
    if not dump_only:
        env_name = create_env_local(env_type, os_name, packages, env_name)
    dump_env_local(env_type, os_name, env_name)
    
def create_and_dump_env_docker(env_type, os_name, packages, env_name=None, dump_only=False):
    """Creates and dumps the env in a docker container."""
    if dump_only:
        raise ValueError('Cannot only dump docker environments, must create as well.')

    env_name_arg = env_name
    if env_name_arg is None:
        env_name_arg = get_env_prefix(env_type)

    channels_arg = ' '.join([f'-c {c}' for c in CHANNELS])
    packages_arg = ' '.join(packages)
    dockerfile = DOCKER_TEMPLATE.format(env_name_arg=env_name_arg,
                                        channels_arg=channels_arg,
                                        packages_arg=packages_arg)
    docker_filename = f'docker/{os_name}-Env-Dockerfile'
    with open(docker_filename, 'w') as fid:
        fid.write(dockerfile)

    env_type_str = ''
    if env_type is not None:
        env_type_str = '-'+env_type
        
    docker_image_name = 'watershed_workflow_env'+env_type_str
    docker_image_name += f'-{os_name}'
    docker_image_name = docker_image_name.lower()

    # build the env
    subprocess.run(['docker', 'build', '--progress=plain', '-f', docker_filename, '-t', docker_image_name, '.'],
                   check=True)

    # start the dummy layer
    subprocess.run(['docker', 'create', '-it', '--name', 'dummy', docker_image_name, 'bash'],
                   check=True)

    # copy the file out
    subprocess.run(['docker', 'cp', 'dummy:/environment.yml',
                    os.path.join('environments', f'environment{env_type_str}-{os_name}.yml')],
                   check=True)

    # remove the image
    subprocess.run(['docker', 'rm', '-f', 'dummy'])

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helper script to (re-)create environments and dump them to file.')
    parser.add_argument('--dump-only', action='store_true', help='Only write the .yml file')
    parser.add_argument('--CI', action='store_true', help='Use only the CI set of packages.')
    parser.add_argument('--env', default=None, type=str, help='Name for this environement')
    parser.add_argument('--user-env', default=None, type=str, help='Create an environment for the User packages.')
    parser.add_argument('--user-env-extras', action='store_true', help='User build with my personal extras.')
    parser.add_argument('--tools-env', default=None, type=str, help='Create an environment for compiling things.')
    parser.add_argument('--docker', action='store_true', help='Build in docker container')
    parser.add_argument('OS', type=str, help='Operating system flag for filename, likely OSX or Linux')

    args = parser.parse_args()

    if args.docker:
        func = create_and_dump_env_docker
    else:
        func = create_and_dump_env_local
    
    if args.CI:
        # if CI, we only need to make the CI environment, no need for user envs
        func('CI', args.OS, get_packages('CI'), args.env, dump_only=args.dump_only)
    else:
        # create the WW env
        func(None, args.OS, get_packages(), args.env, dump_only=args.dump_only)

        # create the default user environment
        if args.user_env is not None:
            func('USER', args.OS, get_packages('USER', args.user_env_extras),
                 args.user_env, dump_only=args.dump_only)

    if args.tools_env is not None:
        func('TOOLS', args.OS, get_packages('TOOLS', os_name=args.OS),
             args.tools_env, dump_only=args.dump_only)
        
