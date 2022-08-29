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
    hucly = watershed_workflow.utils.shply(huc['geometry'])
    native_crs = watershed_workflow.crs.from_fiona(hprofile['crs'])

    # get imgs
    daymet = watershed_workflow.sources.manager_daymet.FileManagerDaymet()

    filename = daymet.get_meteorology('prcp', 1999, hucly, native_crs, force_download=True)
