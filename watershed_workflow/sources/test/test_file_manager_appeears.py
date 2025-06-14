import pytest

import os
import shapely
import numpy as np
import datetime
import time
import xarray as xr

import watershed_workflow.config
import watershed_workflow.sources.manager_shape
import watershed_workflow.sources.manager_modis_appeears

# def test_shape1():
#     """Tests a forced download"""
#     ms = watershed_workflow.sources.manager_shape.FileManagerShape(
#         os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
#     profile, shape = ms.get_shape()
#     shply = watershed_workflow.utils.create_shply(shape['geometry'])

#     app = watershed_workflow.sources.manager_modis_appeears.FileManagerMODISAppEEARS()

#     START = datetime.date(2010, 7, 1)
#     END = datetime.date(2010, 8, 30)

#     res = app.get_data(shply, profile['crs'], start=START, end=END,
#                        force_download=True)
#     res = app.wait(res)
#     assert(isinstance(res, xr.Dataset))
#     assert(len(res) == 2)
#     assert(len(res.collections) == 2)

#     lc = res['LULC']
#     assert(lc.data.shape == (1,17,20))
#     lai = res['LAI']
#     assert(lai.data.shape == (16,17,20))
#     assert(lai.profile['height'] == 17)
#     assert(lai.profile['width'] == 20)
#     assert(len(lai.times) == 16)


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
    assert (len(res) == 2)
    assert (len(res.collections) == 2)

    lc = res['LULC']
    assert (lc.data.shape == (1, 17, 20))
    lai = res['LAI']
    assert (lai.data.shape == (16, 17, 20))
    assert (lai.profile['height'] == 17)
    assert (lai.profile['width'] == 20)
    assert (len(lai.times) == 16)
