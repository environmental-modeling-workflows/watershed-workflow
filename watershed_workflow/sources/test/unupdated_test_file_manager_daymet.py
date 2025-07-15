import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.crs
import watershed_workflow.sources.manager_nhd
import watershed_workflow.sources.manager_daymet


def test_daymet1():
    # single file covers it
    ms = watershed_workflow.sources.manager_shape.FileManagerShape(
        os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
    hprofile, huc = ms.get_shape()
    hucly = watershed_workflow.utils.create_shply(huc['geometry'])
    native_crs = watershed_workflow.crs.from_fiona(hprofile['crs'])

    # get imgs
    daymet = watershed_workflow.sources.manager_daymet.FileManagerDaymet()
    state = daymet.get_data(hucly,
                            native_crs,
                            '1999-1-1',
                            '1999-2-1', ['prcp'],
                            force_download=True)

    prcp = state['prcp']
    assert (watershed_workflow.crs.isEqual(watershed_workflow.crs.daymet_crs(), prcp.profile['crs']))
    assert (prcp.data.shape == (31, 9, 9))
    assert (prcp.profile['height'] == 9)
    assert (prcp.profile['width'] == 9)
    assert (len(prcp.times) == 31)
