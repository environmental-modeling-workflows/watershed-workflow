import numpy as np

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs

from watershed_workflow.sources.manager_3dep import Manager3DEP

def test_3dep(coweeta):
    ned = Manager3DEP(60)
    dem = ned.getDataset(coweeta.geometry[0], coweeta.crs)
    assert ((99,98) == dem.shape)
    assert abs(np.nanmean(dem.values) - 993) < 1
    
