# Import libraries

import watershed_workflow.ui
watershed_workflow.ui.setup_logging(1)
import watershed_workflow.sources.manager_nhdplusv21
import matplotlib.pyplot as plt
import watershed_workflow
import watershed_workflow.plot 

# Test Highlevel Functions

test_huc = '14010001'
test_hucs_level = 10

## set up a dictionary of source objects
sources = watershed_workflow.source_list.get_default_sources()
sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHDPlus_MidRes']
sources['HUC'] = watershed_workflow.source_list.huc_sources['NHDPlus_MidRes']
sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
watershed_workflow.source_list.log_sources(sources)
crs = watershed_workflow.crs.daymet_crs() # Use the same projection than Daymet

profile_ws, ws = watershed_workflow.get_huc(
            source=sources['HUC'], 
            huc=test_huc, 
            out_crs=crs)

profile_ws_10, ws_10 = watershed_workflow.get_hucs(
            source=sources['HUC'], 
            huc=test_huc,
            level=10, 
            out_crs=crs)

profile_ws_12, ws_12 = watershed_workflow.get_hucs(
            source=sources['HUC'], 
            huc=test_huc,
            level=12, 
            out_crs=crs)



fig, ax = plt.subplots(1)
[watershed_workflow.plot.huc(huc=hu, crs=profile_ws_12, ax=ax, color='lightgrey') \
    for hu in ws_12]
[watershed_workflow.plot.huc(huc=hu, crs=profile_ws_10, ax=ax, color='grey') \
    for hu in ws_10]
watershed_workflow.plot.huc(huc=ws, crs=profile_ws, ax=ax, color='black')



# Test FileManager directly
fm = watershed_workflow.sources.manager_nhdplusv21.FileManagerNHDPlusV21() # File manager
test_huc = '14010001'
test_hucs_level = 12

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


