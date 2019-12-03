"""Configuration and global defaults."""

import os
import subprocess
import configparser

def home():
    return os.path.expanduser('~')

def defaults():
    rcParams = { 'packages data dir' : 'packages' }
    try:
        rcParams['data_directory'] = os.path.join(os.environ['WATERSHED_WORKFLOW_DIR'], 'data')
    except KeyError:
        rcParams['data_directory'] = os.path.join(os.getcwd(), 'data')
    return rcParams
    

def parse():
    """Parse config files and set defaults.

    Returns
    -------
    rcParams : configparser.ConfigParser
      A dict-like object containing parameters.
    """
    rcParams = configparser.ConfigParser(defaults=defaults())
    rcParams.read([os.path.join(os.getcwd(), 'watershed_workflowrc'),
                   os.path.join(os.getcwd(), '.watershed_workflowrc'),
                   os.path.join(home(), '.watershed_workflowrc')])
    return rcParams

def get_git_revision_hash():
    """Returns the git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')

# global config
rcParams = parse()




