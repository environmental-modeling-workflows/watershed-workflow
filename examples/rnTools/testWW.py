import os,sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
import logging
import pandas
pandas.options.display.max_columns = None


import watershed_workflow
import watershed_workflow.source_list
import watershed_workflow.ui
import watershed_workflow.colors
import watershed_workflow.condition
import watershed_workflow.mesh
import watershed_workflow.split_hucs


##-----------------------------
hucs = ['060102070302'] # This is the huc for East Fork Poplar Creek

# set up a dictionary of source objects
sources = watershed_workflow.source_list.get_default_sources()
sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD Plus']
sources['HUC'] = watershed_workflow.source_list.huc_sources['NHD Plus']
sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
watershed_workflow.source_list.log_sources(sources)
crs = watershed_workflow.crs.daymet_crs() # Use the same projection than Daymet

_, ws = watershed_workflow.get_huc(sources['HUC'], hucs[0], crs)

my_hucs = []
for huc in hucs:
    _, ws = watershed_workflow.get_huc(sources['HUC'], huc, crs)
    my_hucs.append(ws)
    
watershed = watershed_workflow.split_hucs.SplitHUCs(my_hucs)

include_rivers=True
ignore_small_rivers = False
prune_by_area_fraction = False

if include_rivers:
    # download/collect the river network within that shape's bounds    
    _, reaches = watershed_workflow.get_reaches(sources['hydrography'], huc, 
                                                watershed.exterior(), crs, crs,
                                                in_network=True, properties=True, 
                                                include_catchments=True)
    
    rivers = watershed_workflow.construct_rivers(watershed, reaches, method='hydroseq',
                                                ignore_small_rivers=ignore_small_rivers,
                                                prune_by_area_fraction=prune_by_area_fraction,
                                                remove_diversions=True,
                                                remove_braided_divergences=True)


figsize = (5,5)
fig = plt.figure(figsize=figsize)
ax = watershed_workflow.plot.get_ax(crs,fig)
watershed_workflow.plot.hucs(watershed, crs, ax=ax, color='k', linewidth=1)
watershed_workflow.plot.rivers(rivers[1], crs, ax=ax, color='red', linewidth=1)
plt.show()

river_tmp = rivers[1]
river_tmp[1]
node = river_tmp.preOrder().list()

 
river_tmp.getRoot()

watershed_workflow.simplify()
