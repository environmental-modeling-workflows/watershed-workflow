import pytest

import os
import shapely
import numpy as np

import watershed_workflow.crs
import watershed_workflow.sources.manager_shapefile
import watershed_workflow.sources.manager_daymet


pytest.skip("skipping DayMet module -- DayMet THREDDS is down", allow_module_level=True)



def test_daymet1():
    # single file covers it
    ms = watershed_workflow.sources.manager_shapefile.ManagerShapefile(
        os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp'))
    coweeta = ms.getShapes()

    # get imgs
    daymet = watershed_workflow.sources.manager_daymet.ManagerDaymet()
    data = daymet.getDataset(coweeta.geometry[0],
                            coweeta.crs,
                            '1999-1-1',
                            '1999-2-1',
                             ['prcp'],
                            force_download=True)

    prcp = data['prcp']
    assert watershed_workflow.crs.isEqual(watershed_workflow.crs.daymet_crs, data.rio.crs)
    assert prcp.data.shape == (31, 9, 9)
    assert len(prcp.x) == 9
    assert len(prcp.y) == 9
    assert len(prcp.time) == 31
