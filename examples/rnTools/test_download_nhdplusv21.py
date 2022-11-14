# Genral packages to import ----------------------------------------------
from bs4 import BeautifulSoup
import urllib.request
from requests_html import HTMLSession
import watershed_workflow.daymet
import watershed_workflow.densify_rivers_hucs
import watershed_workflow.create_river_mesh
import watershed_workflow.split_hucs
import watershed_workflow.mesh
import watershed_workflow.condition
import watershed_workflow.colors
import watershed_workflow.ui
import watershed_workflow.source_list
import watershed_workflow
import ssl
import urllib
import py7zr
import requests
from distutils.command.install_egg_info import to_filename
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
import logging
import pandas
import copy
import time
from scipy import integrate
import pickle
import rasterio
import fiona

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
import watershed_workflow.sources.utils as source_utils

# Change working directory and import watershed analysis functions ---------------
os.chdir('/Users/8n8/Documents/myRepos/watershed-workflow/examples/rnTools')
from watershed_analysis_functions import *
from daymet_watershed_analysis_functions import *
from functions_NHDPlusV2_io import *


# General variables and paths for the NHDPlus information

## Component names for the NHDPlus V2 datasets
componentnames_vpu_wide_all = ["EROMExtension", "NHDPlusAttributes", "NHDPlusBurnComponents", "NHDPlusCatchment", \
    "NHDSnapshotFGDB", "NHDSnapshot", "VPUAttributeExtension", "VogelExtension", "WBDSnapshot"]

componentnames_rpu_wide_all = ["CatSeed", "FdrFac", "FdrNull", "FilledAreas", "HydroDem", "NEDSnapshot"]

componentnames_vpu_wide_main = ["EROMExtension", "NHDPlusCatchment", "NHDPlusAttributes", "NHDSnapshot", \
    "VPUAttributeExtension", "WBDSnapshot"]

componentnames_rpu_wide_main = ["NEDSnapshot"]

## Global data provided by NHDPlusV2 (https://www.epa.gov/waterdata/nhdplus-global-data)

path_BoundaryUnitsNHDPlusV21 = "/Users/8n8/Dropbox/myworkfiles_ornl/01_projects/01_active/IDEAS/data/gis_data/nhd_plusv21/NHDPlusGlobalData/"
filename_BoundaryUnitsNHDPlusV21 = "BoundaryUnit.shp"

BoundaryUnitFile = os.path.join(path_BoundaryUnitsNHDPlusV21,filename_BoundaryUnitsNHDPlusV21)

## Test the watershed analysis routines
hucs = ['14020001', '14020002', '14020003', '14020004', '14020005', '14020006'] # Gunnison
# huc = '060102070302'  # This is the huc 12-digit Hydrologic Unit for East Fork Poplar Creek
# hucs = ['020401040301', '020401040302'] # Never Sink

# Basic information for the coordinate reference system and data sources for watershed_workflow -------------

# Coordinate reference system
crs = watershed_workflow.crs.daymet_crs()
# Dictionary of source objects
sources = watershed_workflow.source_list.get_default_sources()
sources['hydrography'] = watershed_workflow.source_list.hydrography_sources['NHD Plus']
sources['HUC'] = watershed_workflow.source_list.huc_sources['NHD Plus']
sources['DEM'] = watershed_workflow.source_list.dem_sources['NED 1/3 arc-second']
watershed_workflow.source_list.log_sources(sources)

# Step 1: Read the watershed (from HUCs) boundary and its bounds  ----------------------------------------

watershed, bounds, bounds_crs = get_watershed_boundaries(hucs, sources, crs)

# fig_size_x = 10
# fig_size_y = fig_size_x*((bounds[3]-bounds[1])/(bounds[2]-bounds[0])) 
# fig, axs = plt.subplots(1, 1, figsize=[fig_size_x, fig_size_y])
# watershed_workflow.plot.hucs(watershed, crs, 'k', axs)
# watershed_workflow.plot.shply(watershed.exterior(), crs, 'r', axs)
# plt.show()

# Step 2: Download data  ----------------------------------------
data_dir = '/Users/8n8/Downloads/testDownloadNHDPlusV2/dowloaded_datasets'
daID_vpu_rpu, filenames = download_NHDPlusV2_datasets(
                        data_dir = data_dir, 
                        componentnames_vpu_wide = componentnames_vpu_wide_main, 
                        componentnames_rpu_wide = componentnames_rpu_wide_main,
                        bounds = bounds, 
                        bounds_crs = bounds_crs, 
                        BoundaryUnitFile = BoundaryUnitFile, 
                        enforce_VPUs = hucs, 
                        force=False)

# Load the information into a WW tree

_nhdplus_vaa = dict({
    'StreamOrder': 'StreamOrde',
    'StreamLevel': 'StreamLeve',
    'HydrologicSequence': 'Hydroseq',
    'DownstreamMainPathHydroSeq': 'DnHydroseq',
    'UpstreamMainPathHydroSeq': 'UpHydroseq',
    'DivergenceCode': 'Divergence',
    'CatchmentAreaSqKm': 'AreaSqKM',
    'TotalDrainageAreaSqKm': 'TotDASqKM',
    'FromNode': 'FromNode',
    'ToNode': 'ToNode'
})

_nhdplus_elevslope = dict({
    'MinimumElevationSmoothed': 'MINELEVSMO',
    'MaximumElevationSmoothed': 'MAXELEVSMO',
    'MinimumElevationRaw': 'MINELEVRAW',
    'MaximumElevationRaw': 'MAXELEVRAW',
    'Slope': 'SLOPE',
})


_nhdplus_eromma = dict({
    'MeanAnnualFlow': 'Q0001E',
    'MeanAnnualVelocity': 'V0001E',
    'MeanAnnualIncrementalFlow': 'Qincr0001E',
    'MeanAnnualFlowGaugeAdj': 'Q0001E'
})

# in_network : bool, optional
#     If True (default), remove reaches that are not "in" the NHD network

in_network = True
properties = list(_nhdplus_vaa.keys()) + list(_nhdplus_elevslope.keys()) + list(_nhdplus_eromma.keys())
include_catchments=False

# Step 1: find and open the hydrography layer

path_nhdsnapshot = [pp for pp in filenames if "NHDSnapshot" in pp][0]
layer = 'NHDFlowline'
filename = os.path.join(path_nhdsnapshot,'Hydrography',layer+'.shp')

logging.info(
    f"Opening '{filename}' for streams in '{bounds}'")
with fiona.open(filename, mode='r') as fid:
    profile = fid.profile
    bounds = watershed_workflow.warp.bounds(
        bounds, bounds_crs, watershed_workflow.crs.from_fiona(profile['crs']))
    reaches = [r for (i, r) in fid.items(bbox=bounds)]
    logging.info(f"Found total of {len(reaches)} in bounds.")

# Step 2: filter not in network --> Unlike NHD Plus HR, we use 'FLOWDIR' to identify these features
if in_network:
    logging.info("  Filtering reaches not in-network")
    reaches = [
        r for r in reaches
        if ('FLOWDIR' in r['properties']) and not ('Uninitialized' in r['properties']['FLOWDIR'])
    ]



'''
For NHDPlus HR: InNetwork = “Yes” means that the reach is part of the networked NHD flowlines. In NHDPlus HR, some networked flowlines were intentionally removed from the set of features used for catchment generation. Examples included pipelines, elevated canals, headwater flowlines that conflicted with the WBD, and some other limited data conditions. 

For NHD Plus V2: there is not attribute InNetwork. We can use the FlowDir feature. In general, all the InNetwork = 0 would also have 
FlowDir = 0

NHDFlowline Features with "Known Flow" vs. Features with “Unknown
Flow”
There are approximately three million NHDFlowline features in NHDPlusV2. Most, but not all of these features have a known flow direction. Flow direction information is contained in the attribute “FlowDir” in the NHDFlowline feature class attribute table. FlowDir can have the values “With Digitized” (known flow direction) or “Uninitialized” (unknown flow direction). The features having unknown flow direction are primarily: isolated stream segments, canal/ditches, and some channels inside braided networks. The features with known flow direction are the subset of the NHDFlowline feature class which makeup the NHDPlusV2 surface water network. The “Plus” part of NHDPlusV2 is constructed for the flowlines with known flow direction. Catchments and associated catchment area attributes are only populated in NHDPlusV2 features with known flow direction. When using NHDPlusV2, it is useful to symbolize the NHDFlowline feature class using the FlowDir attribute. This helps eliminate displaying of features considered to be in the NHDPlusV2 surface water network. In Figure 12:, the dark blue lines indicate NHDFlowline features with known flow direction and, consequently, are included in the “plus” part of NHDPlusV2. The cyan lines are NHDFlowline features with unknown flow direction and, consequently, are not part of the “Plus” portion of NHDPlusV2.
'''

# Step 3: associate catchment areas with the reaches if NHDPlus
if properties != None:
    reach_dict = dict((r['properties']['ComID'], r) for r in reaches) # The ComID is the key for this dictionary

    # validation of properties
    valid_props = list(_nhdplus_vaa.keys()) + list(
        _nhdplus_eromma.keys()) + ['catchment', ]

    if include_catchments:
        path_nhdcatchment = [pp for pp in filenames if "NHDPlusCatchment" in pp][0]
        layer = 'Catchment'
        filename = os.path.join(path_nhdcatchment,layer+'.shp')

        f"Opening '{filename}' for catchments in '{bounds}'")
        for r in reaches:
            r['properties']['catchment'] = None
        
        with fiona.open(filename, mode='r') as fid:
            for catchment in fid.values():
                reach = reach_dict.get(catchment['properties']['FEATUREID'])
                if reach is not None:
                    reach['properties']['catchment'] = catchment

        # VAA 

        if len(set(_nhdplus_vaa.keys()).intersection(set(properties))) > 0:
            
            path_nhdattributes = [pp for pp in filenames if "NHDPlusAttributes" in pp][0]
            layer = 'PlusFlowlineVAA'
            filename = os.path.join(path_nhdattributes,layer+'.dbf')

            logging.info(
                f"Opening '{filename}' for river network properties in '{bounds}'"
            )
            with fiona.open(filename, mode='r') as fid:
                for flowline in fid.values():
                    reach = reach_dict.get(flowline['properties']['ComID'])
                    if reach is not None:
                        for prop in properties:
                            if prop in list(_nhdplus_vaa.keys()):
                                prop_code = _nhdplus_vaa[prop]
                                reach['properties'][prop] = flowline['properties'][prop_code]

        # Elevslope 

        if len(set(_nhdplus_elevslope.keys()).intersection(set(properties))) > 0:
            
            path_nhdattributes = [pp for pp in filenames if "NHDPlusAttributes" in pp][0]
            layer = 'elevslope'
            filename = os.path.join(path_nhdattributes,layer+'.dbf')

            logging.info(
                f"Opening '{filename}' for river network properties in '{bounds}'"
            )
            with fiona.open(filename, mode='r') as fid:
                for flowline in fid.values():
                    reach = reach_dict.get(flowline['properties']['COMID'])
                    if reach is not None:
                        for prop in properties:
                            if prop in list(_nhdplus_elevslope.keys()):
                                prop_code = _nhdplus_elevslope[prop]
                                reach['properties'][prop] = flowline['properties'][prop_code]

        # EROM 

        if len(set(_nhdplus_eromma.keys()).intersection(set(properties))) > 0:
            
            path_nhderom = [pp for pp in filenames if "EROMExtension" in pp][0]
            name_files = os.listdir(path_nhderom)

            layer = [pp for pp in name_files if "EROM_MA" in pp][0].split('.DBF')[0]
            filename = os.path.join(path_nhderom,layer+'.DBF')

            logging.info(
                f"Opening '{filename}' for river network properties in '{bounds}'"
            )
            with fiona.open(filename, mode='r') as fid:
                for flowline in fid.values():
                    reach = reach_dict.get(flowline['properties']['ComID'])
                    if reach is not None:
                        for prop in properties:
                            if prop in list(_nhdplus_eromma.keys()):
                                prop_code = _nhdplus_eromma[prop]
                                reach['properties'][prop] = flowline['properties'][prop_code]





























# url_vpu_wide = get_NHDPlusV2_component_url("https://www.epa.gov/waterdata/nhdplus-upper-colorado-data-vector-processing-unit-14", componentnames_vpu_wide_all)

# get_url_NHD_dataset(data_links, componentnames_vpu_wide_all[5])

# data_links = get_NHDPlusV2_URLs_from_EPA_url("https://www.epa.gov/waterdata/nhdplus-upper-colorado-data-vector-processing-unit-14", verify=False)
# url_vpu_wide = get_NHDPlusV2_component_url(data_links, componentnames_vpu_wide_main)
# # Step 2: Get the boundary Units that intersect with the watershed

# daID_vpu_rpu = get_BoundaryUnit_Info(bounds, bounds_crs,BoundaryUnitFile, enforce_VPUs=hucs)
# print('Tiles needed: ' )
# print(daID_vpu_rpu)
# print('------------------------------')

# # Step 3: Get the URLs for the sites to download the data

# URLs = get_URLs_VPU(daID_vpu_rpu)



# # for each URL 
# kk = 0
# for kk in range(daID_vpu_rpu):
#     data_links = get_NHDPlusV2_URLs_from_EPA_url(URLs[kk], verify=False)
#     url_vpu_wide = get_NHDPlusV2_component_url(data_links, componentnames_vpu_wide_main)
#     url_rpu_wide = get_NHDPlusV2_component_url(data_links, componentnames_rpu_wide_main)
#     daID, vpu, rpu = daID_vpu_rpu[kk]
# # Step 4: Download the data

#     # cc = 1
#     # url = url_vpu_wide[cc]
#     # component_name = componentnames_vpu_wide_main[cc]
#     # filename = download_NHDPlusV2_datasets(component_name = component_name, url = url, data_dir = data_dir, vpu = vpu)

#     filenames = []
#     for cc in range(len(componentnames_vpu_wide_main)):    
#         url = url_vpu_wide[cc]
#         component_name = componentnames_vpu_wide_main[cc]
#         filenames.append(
#             download_NHDPlusV2_datasets_component(url, data_dir, vpu, force=False)
#         )
#     for cc in range(len(componentnames_rpu_wide_main)):    
#         url = url_rpu_wide[cc]
#         component_name = componentnames_rpu_wide_main[cc]
#         filenames.append(
#             download_NHDPlusV2_datasets_component(url, data_dir, vpu, force=False)
#         )










## To download file https://www.codingem.com/python-download-file-from-url/











# # Global data provided by NHDPlusV2
# # https://www.epa.gov/waterdata/nhdplus-global-data

# path_BoundaryUnitsNHDPlusV21 = "/Users/8n8/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/ornl/01_projects/01_active/IDEAS/data/gis_data/nhd_plusv21/NHDPlusGlobalData"
# filename_BoundaryUnitsNHDPlusV21 = "BoundaryUnit.shp"

# BUfile = os.path.join(path_BoundaryUnitsNHDPlusV21,
#                       filename_BoundaryUnitsNHDPlusV21)


# with fiona.open(BUfile) as fid:
#     # Get the CRS for the Boundary Units
#     BoundaryUnits_crs = watershed_workflow.crs.from_fiona(fid.profile['crs'])
#     # Project the watershed boundary to the CRS for the Boundary Units
#     bounds = watershed_workflow.warp.bounds(
#         bounds, bounds_crs, BoundaryUnits_crs)
#     # Get the boundary Units that intersect with the watershed
#     BUs = [r for (i, r) in fid.items(bbox=bounds)]

# # Consolidate information from the selected Boundary Units
# UnitType = []
# UnitID = []
# DrainageID = []
# for pp in BUs:
#     UnitType.append(pp['properties']['UnitType'])
#     UnitID.append(pp['properties']['UnitID'])
#     DrainageID.append(pp['properties']['DrainageID'])

# UnitType = np.array(UnitType)
# UnitID = np.array(UnitID)
# DrainageID = np.array(DrainageID)

# # Find tuples of Drainage Areas, VPUs, and RPUs
# daID_vpu_rpu = []  # list of lists with the Drainage Areas, VPUs, and RPUs
# daID_unique = np.unique(DrainageID)

# for dd in daID_unique:

#     vpu_unique = np.unique(
#         UnitID[np.argwhere((UnitType == 'VPU') & (DrainageID == dd))])

#     for vv in vpu_unique:
#         daID_vpu_rpu += [[dd, vv, UnitID[ii]] for ii in range(len(UnitType))
#                          if (('RPU' in UnitType[ii]) & (vv[0:2] in UnitID[ii]))]

# File names to download


# component_name = ['']
# version_component = ['']

# for kk, vars in enumerate(daID_vpu_rpu):
#     # "NHDPlusV21_" + vars[0] + "_" + vars[1] + "_" + component_name + "_" + version_component
#     # "NHDPlusV21_" + vars[0] + "_" + vars[1] + "_" + vars[2] + "_" + component_name + "_" + version_component

#     # Create folder named:
# folder_name = "NHDPlus" + daID_vpu_rpu[kk][1]

# # RPU components
# "NHDPlusV21_" + vars[0] + "_" + vars[1] + "_" + \
#     vars[2] + "_" + component_name + "_" + version_component
# # CatSeed_02 -- 01, 02
# # FdrFac_02 -- 01, 03,
# # FdrNull_02 -- 01, 03
# # HydroDem_02 -- 01, 02
# # NEDSnapshot_03 -- 01, 03

# # VPU-Wide components
# "NHDPlusV21_" + vars[0] + "_" + vars[1] + "_" + \
#     component_name + "_" + version_component
# # EROMExtension_07 -- 05, 06, 07, 11,
# # NHDPlusAttributes_10 -- 07, 09, 10, 14,
# # NHDPlusBurnComponents_05 -- 02, 03, 05, 07,
# # NHDPlusCatchment_02 -- 01, 05
# # NHDSnapshotFGDB_07 -- 04, 06, 07, 08, 09
# # NHDSnapshot_07 -- 04, 06, 07, 08, 09
# # VPUAttributeExtension_05 -- 03, 04, 05, 07
# # VogelExtension_02 -- 01, 04, 06
# # WBDSnapshot_04 -- 03, 04, 06


# theURL = 'https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusMS/NHDPlus06/NHDPlusAttributes_10.7z'
# # Download the data behind the URL
# theURL = 'https://www.epa.gov/waterdata/nhdplus-tennessee-data-vector-processing-unit-06'
# # I am not adding a certificate verification (i.e., verify=False). This is NOT good practice!
# response = requests.get(theURL, verify=False)
# status_code = response.status_code  # A status code of 200 means it was accepted
# response.raise_for_status()
# print("Status code:" + str(status_code))

# response.json()
# response.content.decode


# theURL = 'https://www.epa.gov/waterdata/nhdplus-tennessee-data-vector-processing-unit-06'
# session = HTMLSession()
# response = session.get(theURL, verify=False)
# response.raise_for_status()
# status_code = response.status_code  # A status code of 200 means it was accepted
# print("Status code:" + str(status_code))

# all_links = response.html.links
# all_links = response.html.absolute_links
# all_links

# html = response.html
# for html in r.html:
#     print(html)

# theURL = 'https://www.epa.gov/waterdata/nhdplus-tennessee-data-vector-processing-unit-06'

# # I am not adding a certificate verification (i.e., verify=False). This is NOT good practice!
# r = requests.get(theURL, verify=False)
# r.raise_for_status()
# htmltext_for_data = r.json()
# htmltext_for_data.find('NHDPlusAttributes')


# req = urllib.request.Request(theURL)
# # To bypass the certificate issues "urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:997)>"
# gcontext = ssl.SSLContext()
# html = urllib.request.urlopen(req, context=gcontext).read()  # str(

# html.find('NHDPlusAttributes')
# html[28728-10:28728+10]


# # a Request object that specifies the URL you want to fetch
# req = urllib.request.Request(theURL)
# # a ssl.SSLContext instance describing the various SSL options
# gcontext = ssl.SSLContext()
# with urllib.request.urlopen(req, context=gcontext) as response:
#     html = response.read().decode("utf8")

# mystr.find('NHDPlusAttributes')


# '''
# Note about certificates: 
# gcontext = ssl.SSLContext()  # Is used to bypass the certificate issues "urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:997)>" This is NOT a good practice, and we need a better way to do this without the risks involved witht eh gcontext selected. 
# '''

# ####################
# pathDataOut = '/Users/8n8/Downloads/testDownloadNHDPlusV2'
# filename_out = os.path.join(pathDataOut, theURL.split('/')[-1])
# open(filename_out, "wb").write(response.content)

# # Unzip file
# with py7zr.SevenZipFile(filename_out, 'r') as archive:
#     archive.extractall(path=pathDataOut)

#     # cwd = os.getcwd()
#     # try:
#     #     os.chdir(to_location)
#     #     libarchive.extract_file(filename)

# # https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusMS/NHDPlus10U/NHDPlusV21_MS_10U_EROMExtension_07.7z
# # https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusGB/NHDPlusV21_GB_16_EROMExtension_04.7z
# # https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusGB/NHDPlusV21_GB_16_EROMExtension_04.7z


# ########################

# def save_html(html, path):
#     with open(path, 'wb') as f:
#         f.write(html)


# def open_html(path):
#     with open(path, 'rb') as f:
#         return f.read()



# r = requests.get(url, verify=False)

# print(r.content[:100])

# soup = BeautifulSoup(r.content, 'html.parser')
# rows = soup.select('div a')
# rows
# # pathDataOut = '/Users/8n8/Downloads/test_ettp_epa/testDataFromPython'
# # save_html(r.content, pathDataOut)
# # html = open_html(pathDataOut)


# # from lxml import etree









#  [get_url_NHD_dataset(data_links, cc)[0] for cc  in componentnames]