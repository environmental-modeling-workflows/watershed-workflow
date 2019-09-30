import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.utils
import workflow.sources.manager_nhd
import workflow.sources.manager_nrcs

    
def test_nrcs1():
    # single file covers it
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    hprofile, huc = nhd.get_huc('020401010101')
    hucly = workflow.utils.shply(huc['geometry'])
    
    # get imgs
    nrcs = workflow.sources.manager_nrcs.FileManagerNRCS()
    profile, shps = nrcs.get_shapes(hucly.bounds, workflow.crs.from_fiona(hprofile['crs']))
    assert(type(shps[0]) is dict)
    assert('id' in shps[0])

    # check coordinates got flipped
    coord0 = next(workflow.utils.generate_coords(shps[0]))
    assert(-80 < coord0[0] < -70)
    assert(42 < coord0[1] < 43)

    
