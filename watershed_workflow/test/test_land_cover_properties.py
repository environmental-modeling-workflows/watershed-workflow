import pytest
import numpy as np
import pandas as pd
import watershed_workflow

import watershed_workflow.land_cover_properties_Soumendra


def test_crosswalk():
    modis = np.ma.MaskedArray([[1,2],
                               [2,2],
                               [2,3]])
    nlcd = np.ma.MaskedArray([[4,6],
                              [4,5],
                              [4,7]])

    crosswalk = watershed_workflow.land_cover_properties_Soumendra.compute_crosswalk_correlation(modis, None, nlcd, None, plot=False, warp=False)
    assert(crosswalk[4] == 2)
    assert(crosswalk[5] == 2)
    assert(crosswalk[6] == 2)
    assert(crosswalk[7] == 3)
    
def test_average_time_series():
    
    data = np.array([[1,2],
                    [45,2],
                    [10,56]])
    data_one = data.ravel()
    
    average_time = watershed_workflow.land_cover_properties_Soumendra.average_time_series(data_one, smooth_filter=True, nyears=2)
    print(average_time)
