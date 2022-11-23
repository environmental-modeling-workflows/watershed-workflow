# Import libraries

import watershed_workflow.ui
watershed_workflow.ui.setup_logging(1)
import watershed_workflow.sources.manager_nhdplusv21


fm = watershed_workflow.sources.manager_nhdplusv21.FileManagerNHDPlusV21() # File manager
profile,huc = fm.get_huc('14010001') # <--- this currently does not work

profile,reaches = fm.get_reaches('14010001') # <--- this is the next step to add this function