import pytest
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio

import watershed_workflow.conf
import watershed_workflow.sources.manager_soilgrids_2017 as manager_soilgrids
import watershed_workflow.sources.manager_nhd


bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])

@pytest.fixture
def soilgrids():
    return manager_soilgrids.FileManagerSoilGrids2017('US')


def test_soilgrids(soilgrids):
    # requires tiles
    nhd = watershed_workflow.sources.manager_nhd.FileManagerNHDPlus()
    profile, huc = nhd.get_huc('02040101')

    # get imgs
    dem_prof, data = soilgrids.get_layer7(huc, watershed_workflow.crs.from_fiona(profile['crs']))
    #plt.imshow(dem)
    #plt.show()
    
    assert(data['bulk density [kg m^-3]'].shape == (468,488))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sg = soilgrids()
    test_soilgrids(sg)
