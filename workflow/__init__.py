import workflow.conf
if workflow.conf.rcParams['DEFAULT']['proj_network'] == "True":
    import os
    os.environ['PROJ_NETWORK'] = 'ON'
elif workflow.conf.rcParams['DEFAULT']['proj_network'] == "False":
    import os
    os.environ['PROJ_NETWORK'] = 'OFF'

from .hilev import *

