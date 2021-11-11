"""Starts Watershed Workflow Jupyterlab in Docker container"""

epilog = """
A confusing aspect of running Watershed Workflow in a Docker container
is that paths in the host are not the same as paths in the container,
and files must be explicitly shared with the container.

Therefore, when run in the container, `~/.watershed_workflowrc` will not
be the same as that in the host system.  To avoid issues like this, we
always generate an rc file in the working directory using the
parameters supplied in the host system files.  This generated config
should not be modified -- instead create/modify
`~/.watershed_workflowrc` or `WORKDIR/watershed_workflowrc`.

"""

import os
import subprocess
import configparser


def set_up_docker_config(workdir, data_library):
    # read config, not including the dockerfile's config
    rc = configparser.ConfigParser()
    rc.read([os.path.join(os.path.expanduser('~'), '.watershed_workflowrc'),
             os.path.join(workdir, '.watershed_workflowrc'),
             os.path.join(workdir, 'watershed_workflowrc')])

    if data_library is None:
        if 'data_directory' in rc['DEFAULT']:
            data_library = rc['DEFAULT']['data_directory']
        else:
            data_library = os.path.join(workdir, 'data_library')
            if not os.path.isdir(data_library):
                os.mkdir(data_library)

    # set the config's data library to the location, in the container,
    # where we will mount the data_library volume
    rc['DEFAULT']['data_directory'] = '/home/jovyan/data'
    with open(os.path.join(workdir, '.docker_watershed_workflowrc'), 'w') as fid:
        rc.write(fid)

    return data_library
    
def start_docker(data_library, workdir, port):
    abspath_data_library = os.path.abspath(data_library)
    if not os.path.isdir(abspath_data_library):
        raise FileNotFoundError(f'Data library directory {abspath_data_library} does not exist.')
    
    abspath_workdir = os.path.abspath(workdir)
    if not os.path.isdir(abspath_workdir):
        raise FileNotFoundError(f'Working directory {abspath_workdir} does not exist.')

    cmd = ['docker', 'run', '-it', '--rm', '-p', f'{port}:8888',
                    '-v', f'{abspath_data_library}:/home/jovyan/data:delegated',
                    '-v', f'{abspath_workdir}:/home/jovyan/workdir:delegated',
                    'ecoon/watershed-workflow:latest']
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog=epilog)
    parser.add_argument('--rc', type=str, default=None, help='Configuration file, see below.')
    parser.add_argument('--data-library', type=str, default=None, help='Location of data library.')
    parser.add_argument('-p', '--port', type=int, default='8888', help='Port to open for jupyterlab.')
    parser.add_argument('WORKDIR', type=str, help='Where to store output files.')
    args = parser.parse_args()

    if not os.path.isdir(args.WORKDIR):
        raise FileNotFoundError(f'Invalid working directory: {args.WORKDIR}')

    data_library = set_up_docker_config(args.WORKDIR, args.data_library)
    start_docker(data_library, args.WORKDIR, args.port)
