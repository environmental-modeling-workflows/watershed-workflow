from . import _version

__version__ = _version.get_versions()['version']

import os
from watershed_workflow.config import rcParams

if rcParams['DEFAULT']['proj_network'] == "True":
    os.environ['PROJ_NETWORK'] = 'ON'
elif rcParams['DEFAULT']['proj_network'] == "False":
    os.environ['PROJ_NETWORK'] = 'OFF'

from .hilev import *
