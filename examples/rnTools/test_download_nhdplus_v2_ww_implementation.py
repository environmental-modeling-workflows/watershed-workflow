# Import libraries

import watershed_workflow.ui
watershed_workflow.ui.setup_logging(1)
import watershed_workflow.sources.manager_nhdplusv21
import matplotlib.pyplot as plt
import watershed_workflow
import watershed_workflow.plot 

## Test Highlevel Functions
# set up a dictionary of source objects
sources = watershed_workflow.source_list.get_default_sources()
sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHDPlus_MidRes']
sources['HUC'] = watershed_workflow.source_list.huc_sources['NHDPlus_MidRes']



## Test FileManager directly
fm = watershed_workflow.sources.manager_nhdplusv21.FileManagerNHDPlusV21() # File manager
test_huc = '14010001'
test_hucs_level = 10

profile_huc,huc = fm.get_huc(huc = test_huc)

profile_hucs_10,hucs_10 = fm.get_hucs(
                huc = test_huc,
                level = test_hucs_level)

profile_reaches, reaches = fm.get_hydro(
                huc = test_huc,
                bounds=None,
                bounds_crs=None,
                in_network=True,
                properties=True,
                include_catchments=False,
                force_download=False)

profile_bodies, bodies = fm.get_waterbodies(
                huc = test_huc, 
                bounds=None, 
                bounds_crs=None, 
                force_download=False)

fig, ax = plt.subplots(1)
watershed_workflow.plot.shapes(shps = [huc], crs = profile_huc, color='g', ax=ax)
watershed_workflow.plot.shapes(shps = hucs_10, crs = profile_hucs_10, color='r', ax=ax)
watershed_workflow.plot.shapes(shps = bodies, crs = profile_bodies, color='m', ax=ax)
watershed_workflow.plot.shapes(shps = reaches, crs = profile_reaches, color='b', ax=ax)

# profile,reaches = fm.get_reaches('14010001') # <--- this is the next step to add this function
print('Done')




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


