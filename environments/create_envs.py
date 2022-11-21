"""A script to help (re-)generate conda environments for Watershed Workflow."""

# packages required for running WW
PACKAGES_BASE=['python=3.10',
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
              'py7zr',
              'h5py',
              'netCDF4',
              'pytest',
              'nbmake' 
              ]

# extra packages needed in the WW env when building for a user
PACKAGES_EXTRAS_USER=['ipykernel',
                      'papermill',
                      ]

# extras packages needed in the WW env when building for development
PACKAGES_EXTRAS_DEV=[
               'sphinx',
               'numpydoc',
               'sphinx_rtd_theme',
               'nbsphinx',
               'ipython',
               ]

# packages for the base environment
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
#          'defaults',
          ]

PACKAGE_MANAGER = 'conda'


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
    elif env_type == 'DEV':
        return 'watershed_workflow_DEV'
    elif env_type == 'TOOLS':
        return 'watershed_workflow_tools'
    else:
        raise ValueError(f'Unknown env type {env_type}')

def get_packages(env_type, os_name=None, extras=False):
    """Get the list of packages to build."""
    if env_type == 'USER':
        packages = PACKAGES_USER_BASE.copy()
        if extras:
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
            packages.extend(PACKAGES_EXTRAS_USER)
        if env_type == 'DEV':
            packages.extend(PACKAGES_EXTRAS_DEV)
    return packages

def dump_env_local(env_type, os_name, env_name, env_filename=None, new_env_name=None):
    """Dumps the env into an appropriate filename.

    This would be improved by a solution to:
      https://github.com/conda/conda/issues/9399
    but instead we must assume this is OS specific.
    """
    if env_filename is None:
        if env_type is None:
            env_filename = os.path.join('environments', f'environment-{os_name}.yml')
        else:
            env_filename = os.path.join('environments', f'environment-{env_type}-{os_name}.yml')


    if new_env_name is None:
        if env_name is None:
            new_env_name = get_env_prefix(env_type)
        else:    
            new_env_name = env_name

    if env_name is None:
        env_name = get_env_name(get_env_prefix(env_type))


    args = ['env', 'export',]
    if PACKAGE_MANAGER == 'conda':
        args.append('--no-builds')
    result = subprocess.run([PACKAGE_MANAGER,]+args+['--name',env_name],
                            check=True, capture_output=True)

    # try to strip matches that break OSX dependent code...
    stdout = result.stdout.decode('utf-8').split('\n')
    lines = [f'name: {new_env_name}',] + stdout[1:]

    with open(env_filename, 'w') as fid:
        fid.write('\n'.join(lines))

def create_env_local(env_type, os_name, packages, env_name=None):
    """Creates the environment locally."""
    if env_name is None:
        env_prefix = get_env_prefix(env_type)
        env_name = get_env_name(env_prefix)

    # build up the conda env create command
    cmd = [PACKAGE_MANAGER, 'create', '--yes', '--name', env_name]
    for channel in CHANNELS:
        cmd.append('-c')
        cmd.append(channel)
    cmd.extend(packages)

    # call conda env create
    subprocess.run(cmd, check=True)

    # set an environment variable so the user can figure out what we just made
    if env_type is None:
        os.environ[f'WATERSHED_WORKFLOW_ENV'] = env_name
    else:
        os.environ[f'WATERSHED_WORKFLOW_{env_type}_ENV'] = env_name
    return env_name

def create_and_dump_env_local(env_type, os_name, packages, env_name=None, dump_only=False):
    if not dump_only:
        create_env_local(env_type, os_name, packages, env_name)
    dump_env_local(env_type, os_name, env_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helper script to (re-)create environments and dump them to file.')
    parser.add_argument('--env-type', type=str, default='STANDARD', choices=['STANDARD', 'CI', 'DEV'],
                        help='Type of environment to build, one of "STANDARD", "CI" (for minimal build), or "DEV" '
                             '(for development tools)')
    parser.add_argument('--env-name', default=None, type=str, help='Name for this environement')
    parser.add_argument('--without-ww-env', action='store_true', help='Skip building the workflow environment.')
    parser.add_argument('--with-user-env', action='store_true', help='Build a user environment.')
    parser.add_argument('--user-env-name', default=None, type=str, help='Name for the user environment.')
    parser.add_argument('--user-env-extras', action='store_true', help='User build with my personal extras.')
    parser.add_argument('--with-tools-env', action='store_true', help='Build an environment for compiling things.')
    parser.add_argument('--tools-env-name', default=None, type=str, help='Name for the tools environment.')
    parser.add_argument('--dump-only', action='store_true', help='Only write the .yml file')
    parser.add_argument('--manager', default='conda', type=str,
                        help='Package manager, one of conda, mamba, or micromamba')
    parser.add_argument('OS', type=str, choices=['OSX', 'Linux'],
                        help='Operating system flag for filename, likely OSX or Linux')
    args = parser.parse_args()

    PACKAGE_MANAGER = args.manager
    
    if args.env_type == 'STANDARD':
        args.env_type = None
    if not args.without_ww_env:
        # create the workflow environment
        packages = get_packages(args.env_type, args.OS)
        create_and_dump_env_local(args.env_type, args.OS, packages, args.env_name, args.dump_only)

    if args.with_user_env:
        # create the user environment
        packages = get_packages('USER', args.OS, args.user_env_extras)
        create_and_dump_env_local('USER', args.OS, packages, args.user_env_name, args.dump_only)

    if args.with_tools_env:
        # create the tools environment
        packages = get_packages('TOOLS', args.OS)
        create_and_dump_env_local('TOOLS', args.OS, packages, args.tools_env_name, args.dump_only)
        
