"""Configuration and global defaults."""

import os
import subprocess
import configparser

def home():
    return os.path.expanduser('~')

def config():
    """Parse config files and set defaults.

    Returns
    -------
    rcParams : configparser.ConfigParser
      A dict-like object containing parameters.
    """
    rcParams = configparser.ConfigParser()
    try:
        data_directory = os.path.join(os.environ['WATERSHED_WORKFLOW_DIR'], 'data')
    except KeyError:
        data_directory = os.path.join(os.getcwd(), 'data')
    rcParams['DEFAULT']['data_directory'] = data_directory
    
    rcParams.read([os.path.join(os.getcwd(), 'watershed_workflowrc'),
                   os.path.join(os.getcwd(), '.watershed_workflowrc'),
                   os.path.join(home(), '.watershed_workflowrc')])
    return rcParams

def get_git_revision_hash():
    """Returns the git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')


def set_data_directory(path):
    """Sets the directory in which all data is stored."""
    rcParams['DEFAULT']['data_directory'] = path

# global config
rcParams = config()






