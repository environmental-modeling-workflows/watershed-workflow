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
    bounds = shapely.geometry.shape(shape['geometry']).bounds
    print(bounds)
    assert(np.allclose(
        np.array([1129295.2795668286, 1404623.5896358239, 1134448.7938438908, 1410000.8473600561]),
        np.array(bounds), 1.e-4))

def test_shape2():
    ms = workflow.sources.manager_shape.FileManagerShape(
        os.path.join(workflow.conf.rcParams['data dir'], 'hydrologic_units', 'others', 'Coweeta', 'coweeta_subwatersheds.shp'))
    profile, shape = ms.get_shape(filter=lambda i,a: i == 3)
    bounds = shapely.geometry.shape(shape['geometry']).bounds
    print(bounds)
    assert(np.allclose(
        np.array([1129887.28006847, 1404623.58964961, 1134344.01901746, 1409691.14880939]),
        np.array(bounds), 1.e-4))
    
