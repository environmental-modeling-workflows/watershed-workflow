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
    nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
    hprofile, huc = nhd.get_huc('020401010101')
    hucly = watershed_workflow.utils.create_shply(huc['geometry'])
    native_crs = watershed_workflow.crs.from_fiona(hprofile['crs'])

    # get imgs
    daymet = watershed_workflow.sources.manager_daymet.FileManagerDaymet()
    state = daymet.get_data(hucly, native_crs,
                            '1999-1-1', '1999-2-1', ['prcp'],
                            force_download=True)

    prcp = state['prcp']
    assert(watershed_workflow.crs.equal(watershed_workflow.crs.daymet_crs(),
                                        prcp.profile['crs']))
    assert(prcp.data.shape == (31,19,17))
    assert(prcp.profile['height'] == 19)
    assert(prcp.profile['width'] == 17)
    assert(len(prcp.times) == 31)
