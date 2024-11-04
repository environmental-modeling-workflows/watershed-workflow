import pytest

import os
import numpy as np

from watershed_workflow.sources.manager_nhd import FileManagerWBD

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8_ll = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

#bounds4_crs = np.array(list(watershed_workflow.warp.xy(bounds4[0], bounds4[1], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())) + list(watershed_workflow.warp.xy(bounds4[2], bounds4[3], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())))

#bounds8_crs = np.array(list(watershed_workflow.warp.xy(bounds8[0], bounds8[1], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())) + list(watershed_workflow.warp.xy(bounds8[2], bounds8[3], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())))


def test_wbd_get() -> None:
    wbd = FileManagerWBD()
    huc = wbd.getShapes('02040101')
    bounds = huc.loc['02040101'].geometry.bounds
    assert (np.allclose(bounds8_ll, np.array(bounds), 1.e-6))


def test_wbd_get_many() -> None:
    wbd = FileManagerWBD()
    huc = wbd.getShapes('02040101', 12)
    print(huc)
    assert len(huc) == len(set(huc.index)) # unique
    assert all(l.startswith('02040101') for l in huc.index) # all in the HUC8
    assert (len(huc) == 38) # right number
    

