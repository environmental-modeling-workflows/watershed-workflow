import pytest

import os
import shapely
import numpy as np

import workflow.conf
import workflow.sources.manager_shape

    
def test_shape1():
    ms = workflow.sources.manager_shape.FileManagerShape(
        os.path.join(workflow.conf.rcParams['data dir'], 'hydrologic_units', 'others', 'Coweeta', 'coweeta_basin.shp'))
    profile, shape = ms.get_shape()
    bounds = workflow.utils.shply(shape['geometry']).bounds
    print(bounds)
    assert(np.allclose(
        np.array([273971.0911428096, 3878839.6361173145, 279140.9150949494, 3883953.7853134344]),
        np.array(bounds), 1.e-4))

def test_shape2():
    ms = workflow.sources.manager_shape.FileManagerShape(
        os.path.join(workflow.conf.rcParams['data dir'], 'hydrologic_units', 'others', 'Coweeta', 'coweeta_subwatersheds.shp'))
    profile, shape = ms.get_shape(filter=lambda i,a: i == 3)
    bounds = workflow.utils.shply(shape['geometry']).bounds
    print(bounds)
    assert(np.allclose(
        np.array([274433.5915278848, 3878839.6361189918, 279050.0001343766, 3883530.4727734774]),
        np.array(bounds), 1.e-4))
    
