import pytest
import numpy as np
import pandas as pd
import watershed_workflow

import watershed_workflow.land_cover_properties


def test_crosswalk():
    modis = np.array([[1, 2], [2, 2], [2, 3]])
    nlcd = np.array([[4, 6], [4, 5], [4, 7]])

    crosswalk = watershed_workflow.land_cover_properties.computeMaximalCrosswalkCorrelation(
        dict(nodata=-1), modis, dict(nodata=-1), nlcd, plot=False, warp=False)
    assert (crosswalk[4] == 2)
    assert (crosswalk[5] == 2)
    assert (crosswalk[6] == 2)
    assert (crosswalk[7] == 3)
