# Import libraries

import watershed_workflow.ui
watershed_workflow.ui.setup_logging(1)
import watershed_workflow.sources.manager_nhdplusv21
import matplotlib.pyplot as plt
import watershed_workflow

fm = watershed_workflow.sources.manager_nhdplusv21.FileManagerNHDPlusV21() # File manager
# profile,huc = fm.get_huc('14010001')

# profile_hucs,hucs = fm.get_hucs('14010001',10)

# crs_daymet = watershed_workflow.crs.daymet_crs() 
# _, hucs_split  = watershed_workflow.hilev.get_split_form_hucs(fm, '14010001',10,crs_daymet)

# crs = watershed_workflow.crs.from_fiona(profile['crs'])

# huc_shply = watershed_workflow.utils.shply(huc) 
# hucs_shply = [watershed_workflow.utils.shply(shp) for shp in hucs]

# fig, ax = plt.subplots()
# [ax.plot(*gg.exterior.xy) for gg in hucs_shply]
# ax.plot(*huc_shply.exterior.xy, c='k')

# # fig = plt.figure(figsize=figsize)
# # ax = watershed_workflow.plot.get_ax(crs,fig)
# # watershed_workflow.plot.hucs(watershed, crs, ax=ax, color='k', linewidth=1)
# # watershed_workflow.plot.rivers(river, crs, ax=ax, color='red', linewidth=1)
# # plt.show()

profile, reaches = fm.get_hydro(
                  '14010001',
                  bounds=None,
                  bounds_crs=None,
                  in_network=True,
                  properties=True,
                  include_catchments=False,
                  force_download=False)

profile,reaches = fm.get_reaches('14010001') # <--- this is the next step to add this function