import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.utils
import workflow.crs
import workflow.sources.manager_nhd
import workflow.sources.manager_daymet

    
def test_daymet1():
    # single file covers it
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    hprofile, huc = nhd.get_huc('020401010101')
    hucly = workflow.utils.shply(huc['geometry'])
    native_crs = workflow.crs.from_fiona(hprofile['crs'])
    
    # get imgs
    daymet = workflow.sources.manager_daymet.FileManagerDaymet()

    filename = daymet.get_meteorology('prcp', 1999, hucly, native_crs)
    
    
