"""A simple script to help (re-)generate conda environments."""

PACKAGES_ALL=['python=3',
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
              ]

PACKAGES_DEV=['sphinx',
              'numpydoc',
              'sphinx_rtd_theme',
              'nbsphinx',
              ]

PACKAGES_USER=['ipython',
               'jupyterlab',
               'ipykernel',
               'nb_conda',
               'nb_conda_kernels',
               'papermill',
               ]

TO_STRIP=['libgfort',
          'libglib',
          'libcx',
          'python.app',
          ]

CHANNELS=['conda-forge',
          'defaults',
          ]

import datetime
import subprocess

def date_str():
    """Gets the date in a preferred format for writing env names"""
    return datetime.datetime.today().strftime('%Y-%m-%d')

def get_env_name(name):
    """Standard format for the name of an environment."""
    return name+'-'+date_str()

def get_env_prefix(env_type=None):
    if env_type is None:
        return 'watershed_workflow'
    elif env_type == 'CI':
        return 'watershed_workflow_CI'
    elif env_type == 'dev':
        return 'watershed_workflow_dev'
    else:
        raise ValueError(f'Unknown env type {env_type}')

def dump_env(env_type=None):
    """Dumps the env into an appropriate filename, attempting to 'clean it up'.

    This would be improved by a solution to:
      https://github.com/conda/conda/issues/9399
    but instead we manually hack to remove some known OS-specific libraries.
    """
    env_prefix = get_env_prefix(env_type)
    env_name = get_env_name(env_prefix)
    if env_type is None:
        dump_filename = 'environment.yml'
    else:
        dump_filename = f'environment-{env_type}.yml'

    result = subprocess.run(['conda','env','export','--no-builds','-n',env_name],
                            check=True, capture_output=True)

    # try to strip matches that break OSX dependent code...
    lines = []
    stdout = result.stdout.decode('utf-8').split('\n')
    # the first line includes the name -- strip the date
    lines.append(stdout[0].split('-')[0])

    for line in stdout[1:]:
        if not any(line.strip().startswith(m) for m in TO_STRIP):
            lines.append(line)

    with open(dump_filename, 'w') as fid:
        fid.write('\n'.join(lines))

def create_env(env_type=None):
    packages = PACKAGES_ALL.copy()
    env_prefix = get_env_prefix(env_type)

    if env_type != 'CI':
        packages.extend(PACKAGES_USER)
        if env_type == 'dev':
            packages.extend(PACKAGES_DEV)

    # build up the conda env create command
    cmd = ['conda', 'create', '--yes']
    cmd.append('-n')
    cmd.append(get_env_name(env_prefix))

    for channel in CHANNELS:
        cmd.append('-c')
        cmd.append(channel)

    cmd.extend(packages)

    # call conda env create
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    import sys,os
    import argparse
    parser = argparse.ArgumentParser(description='Helper script to (re-)create environments and dump them to file.')
    parser.add_argument('--CI', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--dump-only', action='store_true')

    args = parser.parse_args()

    env_type = None
    if args.CI:
        env_type = 'CI'
    elif args.dev:
        env_type = 'dev'

    if not args.dump_only:
        create_env(env_type)
    dump_env(env_type)
    
    

    
