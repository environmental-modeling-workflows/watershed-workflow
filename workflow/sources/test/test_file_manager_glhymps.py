import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.utils
import workflow.sources.manager_glhymps
import workflow
    

def test_glhymps():
    # single file covers it
    coweeta_shapefile = "./data/hydrologic_units/others/Coweeta/coweeta_basin.shp"
    crs, coweeta = workflow.get_split_form_shapes(coweeta_shapefile)
    target_bounds = coweeta.exterior().bounds

    # get imgs
    glhymps = workflow.sources.manager_glhymps.FileManagerGLHYMPS()
    profile, shps, df = glhymps.get_shapes_and_properties(target_bounds,crs)

    # check df
    ids = set([int(s['properties']['id']) for s in shps])
    assert(len(df) == len(ids)) # one per unique key
    assert(set(df['id'].values) == ids) # same ids
    


    
