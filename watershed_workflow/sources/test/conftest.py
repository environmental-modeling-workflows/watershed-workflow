import pytest
import os

from watershed_workflow.sources.manager_shapefile import ManagerShapefile
import watershed_workflow.crs


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'network: marks tests that require live network access (deselect with -m "not network")'
    )


@pytest.fixture
def coweeta():
    ms = ManagerShapefile(
        os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_simplified.shp'))
    shp = ms.getShapes()
    return shp.to_crs(watershed_workflow.crs.from_epsg(4269))
