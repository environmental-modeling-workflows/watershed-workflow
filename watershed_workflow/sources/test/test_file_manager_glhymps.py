import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.sources.manager_glhymps
import watershed_workflow


def test_glhymps():
    # single file covers it
    coweeta_shapefile = "./examples/Coweeta/input_data/coweeta_basin.shp"
    crs, coweeta = watershed_workflow.get_split_form_shapes(coweeta_shapefile)
    target_bounds = coweeta.exterior().bounds

    # get imgs
    glhymps = watershed_workflow.sources.manager_glhymps.FileManagerGLHYMPS()
    profile, shps, df = glhymps.get_shapes_and_properties(target_bounds, crs)

    # check df
    ids = set([int(s['properties']['id']) for s in shps])
    assert (len(df) == len(ids))  # one per unique key
    assert (set(df['id'].values) == ids)  # same ids
