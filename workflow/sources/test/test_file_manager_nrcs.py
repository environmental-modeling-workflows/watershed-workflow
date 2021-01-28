import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.utils
import workflow.sources.manager_nhd
import workflow.sources.manager_nrcs
import workflow
    
# def test_nrcs1():
#     # single file covers it
#     nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
#     hprofile, huc = nhd.get_huc('020401010101')
#     hucly = workflow.utils.shply(huc['geometry'])
    
#     # get imgs
#     nrcs = workflow.sources.manager_nrcs.FileManagerNRCS()
#     profile, shps = nrcs.get_shapes(hucly.bounds, workflow.crs.from_fiona(hprofile['crs']))
#     assert(type(shps[0]) is dict)
#     assert('id' in shps[0])

#     # check coordinates got flipped
#     coord0 = next(workflow.utils.generate_coords(shps[0]))
#     assert(-80 < coord0[0] < -70)
#     assert(42 < coord0[1] < 43)


def test_nrcs2():
    # single file covers it
    coweeta_shapefile = "./data/hydrologic_units/others/Coweeta/coweeta_basin.shp"
    crs, coweeta = workflow.get_split_form_shapes(coweeta_shapefile)
    target_bounds = coweeta.exterior().bounds

    # get imgs
    nrcs = workflow.sources.manager_nrcs.FileManagerNRCS()
    profile, shps, df = nrcs.get_shapes_and_properties(target_bounds,crs)

    # check df
    mukeys = set([int(s['properties']['id']) for s in shps])
    mukeys_df = df.index
    assert(len(mukeys_df) == len(mukeys)) # one per unique key
    assert(set(mukeys_df) == mukeys) # same mukeys

    print(df)
    for m in mukeys:
        df.loc[m] # this throws if it doesn't work



    
