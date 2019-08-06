import pytest
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio

import workflow.conf
import workflow.sources.manager_nlcd as manager_nlcd
import workflow.sources.manager_nhd



bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])

@pytest.fixture
def nlcd():
    return manager_nlcd.FileManagerNLCD(layer='Land_Cover')

def test_nlcd_downloads_plots(nlcd):
    f = nlcd.download(bounds4_ll, workflow.conf.latlon_crs())
    with rasterio.open(f, 'r') as fid:
        d = fid.read(1)

    plt.imshow(d)
    plt.show()


def test_nlcd(nlcd):
    # requires tiles
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    dem_prof, dem = nlcd.get_raster(huc, profile['crs'])
    plt.imshow(dem)
    plt.show()
    
    assert(dem.shape == (3449, 3580))
