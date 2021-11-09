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

# this isn't too robust
TO_STRIP=[]

CHANNELS=['conda-forge',
          'defaults',
          ]


DOCKER_TEMPLATE = """# Docker container to create the env
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
    if env_type is None:
        return 'watershed_workflow'
    elif env_type == 'CI':
        return 'watershed_workflow_CI'
    elif env_type == 'dev':
        return 'watershed_workflow_dev'
    else:
        raise ValueError(f'Unknown env type {env_type}')

def get_packages(env_type=None):
    """Get the list of packages to build."""
    packages = PACKAGES_ALL.copy()
    if env_type != 'CI':
        packages.extend(PACKAGES_USER)
        if env_type == 'dev':
            packages.extend(PACKAGES_DEV)
    return packages

def create_env_local(env_type=None):
    """Creates the environment locally."""
    env_prefix = get_env_prefix(env_type)
    packages = get_packages(env_type)

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

def dump_env_local(env_type, os_name):
    """Dumps the env into an appropriate filename, attempting to 'clean it up'.

    This would be improved by a solution to:
      https://github.com/conda/conda/issues/9399
    but instead we manually hack to remove some known OS-specific libraries.
    """
    env_prefix = get_env_prefix(env_type)
    env_name = get_env_name(env_prefix)
    if env_type is None:
        dump_filename = os.path.join('environments', f'environment-{os_name}.yml')
    else:
        dump_filename = os.path.join('environments', f'environment-{env_type}-{os_name}.yml')

    result = subprocess.run(['conda','env','export','--no-builds','-n',env_name],
                            check=True, capture_output=True)

    # try to strip matches that break OSX dependent code...
    lines = []
    stdout = result.stdout.decode('utf-8').split('\n')
    # the first line includes the name -- strip the date
    lines.append(stdout[0].split('-')[0])

    for line in stdout[1:]:
        if not any(line.strip(' -').startswith(m) for m in TO_STRIP):
            lines.append(line)

    with open(dump_filename, 'w') as fid:
        fid.write('\n'.join(lines))

    
def create_and_dump_env_docker(env_type, os_name):
    """Creates and dumps the env in a docker container."""
    env_name_arg = get_env_prefix(env_type)
    channels_arg = ' '.join([f'-c {c}' for c in CHANNELS])
    packages_arg = ' '.join(get_packages(env_type))
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
    parser.add_argument('--os', type=str, default='OSX', help='Operating system flag for filename.')
    parser.add_argument('--dump-only', action='store_true', help='Only write the .yml file')
    parser.add_argument('--CI', action='store_true', help='Use the CI set of packages.')
    parser.add_argument('--dev', action='store_true', help='Use the dev set of packages.')

    args = parser.parse_args()

    env_type = None
    if args.CI:
        env_type = 'CI'
    elif args.dev:
        env_type = 'dev'

    if args.os == 'OSX':
        if not args.dump_only:
            create_env_local(env_type)
        dump_env_local(env_type, args.os)
    elif args.os == 'Linux':
        create_and_dump_env_docker(env_type, args.os)
    else:
        raise ValueError('Invalid os type: must be one of {OSX,Linux}')
        
    
    

    
