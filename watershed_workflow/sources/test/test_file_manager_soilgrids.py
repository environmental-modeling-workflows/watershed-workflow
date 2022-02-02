import pytest
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio

import workflow.conf
import workflow.sources.manager_soilgrids_2017 as manager_soilgrids
import workflow.sources.manager_nhd


bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])

#@pytest.fixture
def soilgrids():
    return manager_soilgrids.FileManagerSoilGrids2017()


def test_soilgrids(soilgrids):
    # requires tiles
    nhd = workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    dem_prof, dem = soilgrids.get_layer7(huc, workflow.crs.from_fiona(profile['crs']))
    #plt.imshow(dem)
    #plt.show()
    
    assert(dem.shape == (3808,2460))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sg = soilgrids()
    test_soilgrids(sg)
