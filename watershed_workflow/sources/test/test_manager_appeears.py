import pytest

import os
import shapely
import numpy as np
import datetime
import time
import xarray as xr

import watershed_workflow.sources.manager_modis_appeears

from fixtures import coweeta


def test_shape2(coweeta):
    """Tests a pre-existing download"""
    app = watershed_workflow.sources.manager_modis_appeears.ManagerMODISAppEEARS()

    START = datetime.date(2010, 7, 1)
    END = datetime.date(2010, 8, 30)

    count = 0
    success = False
    # res = app.getDataset(coweeta.geometry[0], coweeta.crs, start=START, end=END)
