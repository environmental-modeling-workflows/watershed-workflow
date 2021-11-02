from . import _version
__version__ = _version.get_versions()['version']

import watershed_workflow.conf
if watershed_workflow.conf.rcParams['DEFAULT']['proj_network'] == "True":
    import os
    os.environ['PROJ_NETWORK'] = 'ON'
elif watershed_workflow.conf.rcParams['DEFAULT']['proj_network'] == "False":
    import os
    os.environ['PROJ_NETWORK'] = 'OFF'

from .hilev import *


