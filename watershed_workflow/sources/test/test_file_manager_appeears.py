import pytest

import os
import shapely
import numpy as np
import datetime
import time

import watershed_workflow.config
import watershed_workflow.sources.manager_shape
import watershed_workflow.sources.manager_modis_appeears

def test_shape1():
    """Tests a forced download"""
    ms = watershed_workflow.sources.manager_shape.FileManagerShape(
        os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
    profile, shape = ms.get_shape()
    shply = watershed_workflow.utils.create_shply(shape['geometry'])

    app = watershed_workflow.sources.manager_modis_appeears.FileManagerMODISAppEEARS()

    START = datetime.date(2010, 7, 1)
    END = datetime.date(2010, 8, 30)

    count = 0
    success = False
    res = None
    while count < 100 and not success:
        res = app.get_data(shply, profile['crs'], start=START, end=END, force_download=True, task=res)
        if type(res) is tuple:
            success = True
            return
        else:
            time.sleep(120)
            count += 1

def test_shape2():
    """Tests a pre-existing download"""
    ms = watershed_workflow.sources.manager_shape.FileManagerShape(
        os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
    profile, shape = ms.get_shape()
    shply = watershed_workflow.utils.create_shply(shape['geometry'])

    app = watershed_workflow.sources.manager_modis_appeears.FileManagerMODISAppEEARS()

    START = datetime.date(2010, 7, 1)
    END = datetime.date(2010, 8, 30)

    count = 0
    success = False
    res = app.get_data(shply, profile['crs'], start=START, end=END)
    assert(type(res) is tuple)
    assert(len(res[1]) == 2)

    print(res[1][0].shape)
    print(res[1][1].shape)

