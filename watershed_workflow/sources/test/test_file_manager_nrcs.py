import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.sources.manager_nhd
import watershed_workflow.sources.manager_nrcs
import watershed_workflow
    
# def test_nrcs1():
#     # single file covers it
#     nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
#     hprofile, huc = nhd.get_huc('020401010101')
#     hucly = watershed_workflow.utils.shply(huc['geometry'])
    
#     # get imgs
#     nrcs = watershed_workflow.sources.manager_nrcs.FileManagerNRCS()
#     profile, shps = nrcs.get_shapes(hucly.bounds, watershed_workflow.crs.from_fiona(hprofile['crs']))
#     assert(type(shps[0]) is dict)
#     assert('id' in shps[0])

#     # check coordinates got flipped
#     coord0 = next(watershed_workflow.utils.generate_coords(shps[0]))
#     assert(-80 < coord0[0] < -70)
#     assert(42 < coord0[1] < 43)


def test_nrcs2():
    # single file covers it
    coweeta_shapefile = "./data/hydrologic_units/others/Coweeta/coweeta_basin.shp"
    crs, coweeta = watershed_workflow.get_split_form_shapes(coweeta_shapefile)
    target_bounds = coweeta.exterior().bounds

    # get imgs
    nrcs = watershed_workflow.sources.manager_nrcs.FileManagerNRCS()
    profile, shps, df = nrcs.get_shapes_and_properties(target_bounds,crs, force_download=True)

    # check df
    # mukeys = set([int(s['properties']['id']) for s in shps])
    # assert(len(df) == len(mukeys)) # one per unique key
    # assert(set(df['mukey'].values) == mukeys) # same mukeys




    
