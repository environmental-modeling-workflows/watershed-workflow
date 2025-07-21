import pytest
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio

import watershed_workflow.config
import watershed_workflow.sources.manager_nlcd as manager_nlcd
import watershed_workflow.sources.manager_wbd

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])


@pytest.fixture
def nlcd():
    return manager_nlcd.ManagerNLCD()


def test_nlcd(nlcd):
    # requires tiles
    wbd = watershed_workflow.sources.manager_wbd.ManagerWBD()
    huc = wbd.getShapesByID('02040101')

    # get imgs
    data = nlcd.getDataset(huc.geometry[0].buffer(0.01), huc.crs)

    # data.plot.imshow()
    # plt.show()

    assert data.rio.crs is not None
    assert data.shape == (3683, 2880)
