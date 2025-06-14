import pytest
import numpy as np
import pandas as pd
import xarray as xr
import watershed_workflow

import watershed_workflow.land_cover_properties


def test_crosswalk():
    modis = np.array([[1, 2], [2, 2], [2, 3]])
    nlcd = np.array([[4, 6], [4, 5], [4, 7]])
    x = np.array([1,2,3])
    y = np.array([1,2])

    modis_da = xr.DataArray(name='modis', data=modis, coords={'x':x, 'y':y})
    nlcd_da = xr.DataArray(name='nlcd', data=nlcd, coords={'x':x, 'y':y})

    crosswalk = watershed_workflow.land_cover_properties.computeCrosswalk(
        modis_da, nlcd_da, plot=False, warp=False)
    assert (crosswalk[4][0][0] == 2)
    assert (crosswalk[5][0][0] == 2)
    assert (crosswalk[6][0][0] == 2)
    assert (crosswalk[7][0][0] == 3)
