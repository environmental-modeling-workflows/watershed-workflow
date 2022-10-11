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

# Basic information for the coordinate reference system and data sources for watershed_workflow -------------

## Coordinate reference system
crs = watershed_workflow.crs.daymet_crs()
## Dictionary of source objects
sources = watershed_workflow.source_list.get_default_sources()
sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD Plus']
sources['HUC'] = watershed_workflow.source_list.huc_sources['NHD Plus']
sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
watershed_workflow.source_list.log_sources(sources)

#  Read the watershed boundary ----------------------------------------

profile_ws, ws = watershed_workflow.get_huc(sources['HUC'], huc, crs)
watershed = watershed_workflow.split_hucs.SplitHUCs([ws])

fig, axs = plt.subplots(1,1,figsize=[10,10])
watershed_workflow.plot.hucs(watershed, crs, 'k', axs)
plt.show()



##################

# Step 1: Get the bounds from the HUC
watersheds_shapely = list(watershed.polygons())
watersheds_shapely[0].bounds

bounds = watersheds_shapely[0].bounds
bounds_crs = profile_ws 



path_NHDPlusGlobalData = "/Users/8n8/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/ornl/01_projects/01_active/IDEAS/data/gis_data/nhd_plusv21/NHDPlusGlobalData"
filename_boundary_units_NHDPlusV21 = "BoundaryUnit.shp"

downloadfile = os.path.join(path_NHDPlusGlobalData, filename_boundary_units_NHDPlusV21)

# boundary_units = fiona.open(downloadfile)

with fiona.open(downloadfile) as fid:
    boundary_units_crs = watershed_workflow.crs.from_fiona(fid.profile['crs'])

    bounds = watershed_workflow.warp.bounds(
        bounds, bounds_crs, boundary_units_crs)
    bu = [r for (i, r) in fid.items(bbox=bounds)]

UnitType = []
UnitID = []
DrainageID = []
for pp in bu:
    UnitType.append(pp['properties']['UnitType'])    
    UnitID.append(pp['properties']['UnitID']) 
    DrainageID.append(pp['properties']['DrainageID'])

UnitType = np.array(UnitType)
UnitID = np.array(UnitID)
DrainageID = np.array(DrainageID)

# Find RPUs and VPUs
dID_vpu_rpu = []
dID_unique = np.unique(DrainageID)
for dd in dID_unique:

    vpu_unique = np.unique(UnitID[np.argwhere((UnitType == 'VPU') & (DrainageID == dd))])
    
    for vv in vpu_unique:
        dID_vpu_rpu += [[dd, vv, UnitID[ii]] for ii in range(len(UnitType)) if (('RPU' in UnitType[ii]) & (vv[0:2] in UnitID[ii]))]



## Identify the unique VPUs



# idx_rpu = UnitType.index('RPU')
# idx_vpu = UnitType.index('VPU')
kk =0
component_name = ['']

"NHDPlusV21_" + dID_vpu_rpu[kk][0] + "_" + dID_vpu_rpu[kk][1] + "_" + component_name + "_02"
"NHDPlusV21_" + dID_vpu_rpu[kk][0] + "_" + dID_vpu_rpu[kk][1] + "_" + dID_vpu_rpu[kk][2] + "_" + component_name + "_02"


"NHDPlusV21_" + dID_vpu_rpu[kk][0] + "_" + dID_vpu_rpu[kk][1] + "_" + dID_vpu_rpu[kk][2] + "_" + component_name + "_02"
# CatSeed_02.7z(6.4 MB)
# FdrFac_02.7z(172.6 MB)
# FdrNull_02.7z(53.5 MB)
# HydroDem_02.7z(339.4 MB)
# NEDSnapshot_03.7z(456.2 MB)

"NHDPlusV21_" + dID_vpu_rpu[kk][0] + "_" + dID_vpu_rpu[kk][1] + "_" + component_name + "_02"
# EROMExtension_07.7z(127.2 MB)
# NHDPlusAttributes_10.7z(25.4 MB)
# NHDPlusBurnComponents_05.7z(71.4 MB)
# NHDPlusCatchment_02.7z(347.1 MB)
# NHDSnapshotFGDB_07.7z(89.6 MB)
# NHDSnapshot_07.7z(161.1 MB)
# VPUAttributeExtension_05.7z(134.9 MB)
# VogelExtension_02.7z(1.4 MB)
# WBDSnapshot_04.7z(53.3 MB)