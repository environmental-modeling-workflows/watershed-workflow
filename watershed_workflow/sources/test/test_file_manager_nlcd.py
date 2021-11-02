import pytest
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio

import watershed_workflow.conf
import watershed_workflow.sources.manager_nlcd as manager_nlcd
import watershed_workflow.sources.manager_nhd



bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])

@pytest.fixture
def nlcd():
    return manager_nlcd.FileManagerNLCD()

def test_nlcd_downloads_plots(nlcd):
    f = nlcd._download()

def test_nlcd(nlcd):
    # requires tiles
    nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    dem_prof, dem = nlcd.get_raster(huc, watershed_workflow.crs.from_fiona(profile['crs']))
    #plt.imshow(dem)
    #plt.show()
    
    assert(dem.shape == (3808,2460))
