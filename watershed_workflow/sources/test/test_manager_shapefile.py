import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
from watershed_workflow.sources.manager_shapefile import ManagerShapefile


def test_shape1():
    ms = ManagerShapefile(os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
    shp = ms.getShapes()
    bounds = shp.bounds
    print(bounds)
    assert (np.allclose(
        np.array([273971.0911428096, 3878839.6361173145, 279140.9150949494, 3883953.7853134344]),
        np.array(bounds), 1.e-4))
