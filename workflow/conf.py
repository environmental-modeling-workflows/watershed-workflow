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
        data_directory = os.path.join(os.environ['WATERSHED_WORKFLOW_DATA_DIR'])
    except KeyError:
        try:
            data_directory = os.path.join(os.environ['WATERSHED_WORKFLOW_DIR'], 'data')
        except KeyError:
            data_directory = os.path.join(os.getcwd(), 'data')

    # defaults
    rcParams['DEFAULT']['data_directory'] = data_directory
    rcParams['DEFAULT']['ssl_cert'] = "True"  # note this can be True,
                                            # False (bad
                                            # idea/permissive) or a
                                            # path to ssl certs,
                                            # e.g. /etc/ssl/cert.perm
                                            # or similar
    rcParams['DEFAULT']['proj_network'] = "False"

    # rosetta
    try:
        rosetta_path = os.path.join(os.environ['WATERSHED_WORKFLOW_DIR'], 'workflow_tpls',
                                       'rosetta')
    except KeyError:
        workflow_dir = os.path.split(os.path.split(__file__)[0])[0]
        rosetta_path = os.path.join(workflow_dir, 'workflow_tpls',
                                       'rosetta')                                       
    rcParams['DEFAULT']['rosetta_path'] = rosetta_path
    rcParams['DEFAULT']['rosetta_db_path'] = os.path.join(rosetta_path, 'sqlite', 'rosetta.sqlite')
                                            
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






