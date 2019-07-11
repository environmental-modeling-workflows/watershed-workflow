import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.sources.manager_nhd
import workflow.sources.manager_nrcs

    
def test_ned1():
    # single file covers it
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    hprofile, huc = nhd.get_huc('020401010101')
    hucly = workflow.utils.shply(huc['geometry'])
    
    # get imgs
    nrcs = workflow.sources.manager_nrcs.FileManagerNRCS()
    profile, shps, ids = nrcs.get_shapes_in_bounds(hucly.bounds, hprofile['crs'])

    
