import pytest

import os
import numpy as np
import shapely

import geopandas as gpd
from matplotlib import pyplot as plt

from watershed_workflow.sources.manager_wbd import ManagerWBD
import watershed_workflow.crs

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8_ll = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

def test_wbd_get() -> None:
    wbd = ManagerWBD()
    huc = wbd.getShapesByID('02040101')
    bounds = huc.loc['02040101'].geometry.bounds
    assert (np.allclose(bounds8_ll, np.array(bounds), 1.e-6))

def test_wbd_get_many() -> None:
    wbd = ManagerWBD()
    wbd.setLevel(12)
    huc = wbd.getShapesByID('02040101')
    print(huc)
    assert len(huc) == len(set(huc.index)) # unique
    assert all(l.startswith('02040101') for l in huc.index) # all in the HUC8
    assert (len(huc) == 38) # right number

def test_wbd_get_geometry() -> None:
    wbd = ManagerWBD()
    wbd.setLevel(8)
    shp = shapely.geometry.box(*bounds8_ll)
    huc = wbd.getShapesByGeometry(shp, watershed_workflow.crs.latlon_crs)
    huc = huc.to_crs(watershed_workflow.crs.latlon_crs)
    huc = huc[[shp.buffer(0.001).contains(h) for h in huc.geometry]]
    assert len(huc) == 1
    


def test_wbd_waterdata_get() -> None:
    wbd = ManagerWBD('WaterData')
    huc = wbd.getShapesByID('02040101')
    bounds = huc.loc['02040101'].geometry.bounds
    assert (np.allclose(bounds8_ll, np.array(bounds), 1.e-6))

def test_wbd_waterdata_get_many() -> None:
    wbd = ManagerWBD('WaterData')
    wbd.setLevel(12)
    huc = wbd.getShapesByID('02040101')
    print(huc)
    assert len(huc) == len(set(huc.index)) # unique
    assert all(l.startswith('02040101') for l in huc.index) # all in the HUC8
    assert (len(huc) == 38) # right number

def test_wbd_waterdata_get_geometry() -> None:
    wbd = ManagerWBD('WaterData')
    wbd.setLevel(8)
    shp = shapely.geometry.box(*bounds8_ll)
    huc = wbd.getShapesByGeometry(shp, watershed_workflow.crs.latlon_crs)
    huc = huc.to_crs(watershed_workflow.crs.latlon_crs)
    huc = huc[[shp.buffer(0.001).contains(h) for h in huc.geometry]]
    assert len(huc) == 1
    

