# Genral packages to import ----------------------------------------------
from distutils.command.install_egg_info import to_filename
import os,sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
from shapely import Point
import logging
import pandas
import copy
import time
from scipy import integrate
import pickle
import rasterio
import fiona
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
import watershed_workflow.daymet

# Change working directory and import watershed analysis functions ---------------
os.chdir('/Users/8n8/Documents/myRepos/watershed-workflow/examples/rnTools')
from watershed_analysis_functions import *
from daymet_watershed_analysis_functions import *



# Test the watershed analysis routines

huc = '060102070302' # This is the huc 12-digit Hydrologic Unit for East Fork Poplar Creek
toRead_Preprocessed = True

# Get the river network and watershed boundary

if (not toRead_Preprocessed):

    # Basic information for the coordinate reference system and data sources for watershed_workflow -------------

    ## Coordinate reference system
    crs = watershed_workflow.crs.daymet_crs()
    ## Dictionary of source objects
    sources = watershed_workflow.source_list.get_default_sources()
    sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD Plus']
    #sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD']
    sources['HUC'] = watershed_workflow.source_list.huc_sources['NHD Plus']
    sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
    watershed_workflow.source_list.log_sources(sources)

    # get the start time
    st = time.time()
    watershed, river = get_RN_WB(huc = huc, crs=crs, sources = sources)
    print('Execution time:', time.time() - st, 'seconds')

    rn_wb = {'watershed':watershed, 'river':river, 'huc':huc,
            'crs':crs, 'sources':sources}

    # Save watershed and river information in a pickle file
    pickle_name = './results/rn_wb_huc_'+ huc + '.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(rn_wb, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    pickle_name = './results/rn_wb_huc_'+ huc + '.pickle'
    with open(pickle_name, 'rb') as handle:
        rn_wb = pickle.load(handle)

    watershed = rn_wb['watershed']
    river = rn_wb['river']
    huc = rn_wb['huc']
    crs = rn_wb['crs']
    sources = rn_wb['sources']

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

# Get width function ------------------------------------------------
d_to_outlet, weights, width_function, river_lto = get_WightedWidthFunction(river)


fig, axs = plt.subplots(2,1,figsize=[5,10])
axs[0].plot(width_function['distance'],width_function['pdf'])
axs[0].set(xlabel='Distance to the outlet (m)', ylabel=r'Width Function (m$^{-1}$)')

axs[1].plot(width_function['distance'],width_function['cdf'])
axs[1].set(xlabel='Distance to the outlet (m)', ylabel=r'Cumulative Width Function (-)')
fig.savefig('./results/WidthFunction.pdf',bbox_inches='tight')
#plt.show()

# Get daymet data ----------------------------------------------------



## returned raw data has dim(nband, ncol, nrow)
startdate = "1-2011"
enddate = "365-2021"
bounds = watershed.exterior()

# Download Daymet data
daymet_data_raw, x_raw, y_raw = watershed_workflow.daymet.collectDaymet(bounds, crs=crs, 
                                                    start=startdate, end=enddate)

# Reproject Daymet data to the watershed CRS
x_daymet, y_daymet, extent, daymet_data, daymet_profile = \
        watershed_workflow.daymet.reproj_Daymet(x_raw, y_raw, daymet_data_raw, dst_crs=crs)

nday, nrow, ncol = daymet_data['prcp'].shape

# Get (i,j)s of the Daymet pixels with data within the watershed
i_grid, j_grid, idx_grid, x_grid, y_grid = \
    get_Pixels_Inside_Watershed(x_daymet, y_daymet, watershed)

ivar = 'prcp'
idx_time = np.arange(nday)

plt.figure()
plt.bar(idx_time,time_total)
plt.show()


fig, axs = plt.subplots(1,1,figsize=[10,10])
xb,yb = bounds.exterior.xy
axs.plot(xb,yb,'-k')
axs.plot(x_grid,y_grid,'.r')
axs.plot(x_grid[idx_grid],y_grid[idx_grid],'.b')
plt.show()

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
