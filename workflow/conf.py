"""Configuration and global defaults."""

import os
import subprocess

rcParams = {'packages data dir' : 'packages',
            'digits' : 7, # roundoff precision
            }
try:
    rcParams['data dir'] = os.path.join(os.environ['ATS_MESHING_DIR'], 'data')
except KeyError:
    rcParams['data dir'] = os.path.join(os.getcwd(), 'data')


def get_git_revision_hash():
    """Returns the git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')





