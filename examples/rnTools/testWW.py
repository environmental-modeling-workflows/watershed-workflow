# Genral packages to import ----------------------------------------------
import os,sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
import logging
import pandas
import copy
import time
pandas.options.display.max_columns = None

# watershed_workflow packages and modules to import ----------------------------------------------
import watershed_workflow
import watershed_workflow.source_list
import watershed_workflow.ui
import watershed_workflow.colors
import watershed_workflow.condition
import watershed_workflow.mesh
import watershed_workflow.split_hucs
import watershed_workflow.create_river_mesh
import watershed_workflow.densify_rivers_hucs

# Change working directory and import watershed analysis functions ---------------
os.chdir('/Users/8n8/Documents/myRepos/watershed-workflow/examples/rnTools')
from watershed_analysis_functions import *

# Basic information for the coordinate reference system and data sources for watershed_workflow -------------

## Coordinate reference system
crs = watershed_workflow.crs.daymet_crs()
## Dictionary of source objects
sources = watershed_workflow.source_list.get_default_sources()
sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD Plus']
sources['HUC'] = watershed_workflow.source_list.huc_sources['NHD Plus']
sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
watershed_workflow.source_list.log_sources(sources)

# Test the watershed analysis routines

huc = '060102070302' # This is the huc 12-digit Hydrologic Unit for East Fork Poplar Creek

## Get the river network and watershed boundary
# get the start time
st = time.time()
watershed, river = get_RN_WB(huc = huc, crs=crs, sources = sources)
print('Execution time:', time.time() - st, 'seconds')

## Plot network
# Plot the main river
fig, axs = plt.subplots(1,1,figsize=[10,10])

watershed_workflow.plot.hucs(watershed, crs, 'k', axs)
axs.set_title('Main River',fontsize=16)
for node in river.preOrder():
    x,y=node.segment.xy 
    axs.plot(x,y,'-o',markersize=2)
#plt.show()
fig.savefig('./results/RN.pdf',bbox_inches='tight')

#my_watershed = get_WightedWidthFunction(hucs=hucs)

d_to_outlet, river_lto = get_WightedWidthFunction(river)

weights = np.ones_like(d_to_outlet)
n_bins = 100
bin_size = np.ceil(np.max(d_to_outlet)/n_bins)
bins_edges = np.arange(n_bins+1)*bin_size
bin_center = (bins_edges[1:]+bins_edges[:-1])/2

histogram_data, _ = np.histogram(d_to_outlet, bins=bins_edges, weights=weights)
x_wf = np.concatenate(([bins_edges[0]], bin_center, [bins_edges[-1]]))
y_wf = np.concatenate(([0], histogram_data, [0]))

area_wf = np.trapz(y=y_wf, x=x_wf)
y_wf = y_wf/area_wf

fig, axs = plt.subplots(1,1,figsize=[10,10])
axs.plot(x_wf,y_wf)
plt.xlabel('Distance to the outlet (m)')
fig.savefig('./results/WidthFunction.pdf',bbox_inches='tight')




#plt.show()

river
print('Done!')

# import os,sys
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import cm as pcm
# import shapely
# import logging
# import pandas
# pandas.options.display.max_columns = None


# import watershed_workflow
# import watershed_workflow.source_list
# import watershed_workflow.ui
# import watershed_workflow.colors
# import watershed_workflow.condition
# import watershed_workflow.mesh
# import watershed_workflow.split_hucs


# ##-----------------------------
# hucs = ['060102070302'] # This is the huc for East Fork Poplar Creek

# # set up a dictionary of source objects
# sources = watershed_workflow.source_list.get_default_sources()
# sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD Plus']
# sources['HUC'] = watershed_workflow.source_list.huc_sources['NHD Plus']
# sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
# watershed_workflow.source_list.log_sources(sources)
# crs = watershed_workflow.crs.daymet_crs() # Use the same projection than Daymet

# _, ws = watershed_workflow.get_huc(sources['HUC'], hucs[0], crs)

# my_hucs = []
# for huc in hucs:
#     _, ws = watershed_workflow.get_huc(sources['HUC'], huc, crs)
#     my_hucs.append(ws)
    
# watershed = watershed_workflow.split_hucs.SplitHUCs(my_hucs)

# include_rivers=True
# ignore_small_rivers = False
# prune_by_area_fraction = False

# if include_rivers:
#     # download/collect the river network within that shape's bounds    
#     _, reaches = watershed_workflow.get_reaches(sources['hydrography'], huc, 
#                                                 watershed.exterior(), crs, crs,
#                                                 in_network=True, properties=True, 
#                                                 include_catchments=True)
    
#     rivers = watershed_workflow.construct_rivers(watershed, reaches, method='hydroseq',
#                                                 ignore_small_rivers=ignore_small_rivers,
#                                                 prune_by_area_fraction=prune_by_area_fraction,
#                                                 remove_diversions=True,
#                                                 remove_braided_divergences=True)


# figsize = (5,5)
# fig = plt.figure(figsize=figsize)
# ax = watershed_workflow.plot.get_ax(crs,fig)
# watershed_workflow.plot.hucs(watershed, crs, ax=ax, color='k', linewidth=1)
# watershed_workflow.plot.rivers(rivers[1], crs, ax=ax, color='red', linewidth=1)
# plt.show()

# river_tmp = rivers[1]
# river_tmp[1]
# node = river_tmp.preOrder().list()

 
# river_tmp.getRoot()

# watershed_workflow.simplify()
