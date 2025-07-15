import pytest
import os

from watershed_workflow.sources.manager_shapefile import ManagerShapefile
import watershed_workflow.crs

@pytest.fixture
def coweeta():
    ms = ManagerShapefile(os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
    shp = ms.getShapes()
    shp.to_crs(watershed_workflow.crs.latlon_crs)
    return shp
