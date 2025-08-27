import pytest

import os
import numpy as np
import shapely

import geopandas as gpd
from matplotlib import pyplot as plt

from watershed_workflow.sources.manager_wbd import ManagerWBD
import watershed_workflow.crs
import watershed_workflow.sources.standard_names as names

bounds4_ll = np.array([-76.3955534, 36.8008194, -73.9026218, 42.4624454])
bounds8_ll = np.array([-75.5722117, 41.487746, -74.5581047, 42.4624454])

def test_wbd_get() -> None:
    wbd = ManagerWBD()
    huc = wbd.getShapesByID('02040101')
    bounds = huc[huc.ID=='02040101'].geometry.bounds
    assert (np.allclose(bounds8_ll, np.array(bounds), 1.e-6))

def test_wbd_get_many() -> None:
    wbd = ManagerWBD()
    wbd.setLevel(12)
    huc = wbd.getShapesByID('02040101')
    print(huc)
    assert len(huc) == len(set(huc.ID)) # unique
    assert all(l.startswith('02040101') for l in huc.ID) # all in the HUC8
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
    wbd = ManagerWBD(protocol_name='WaterData')
    huc = wbd.getShapesByID('02040101')
    bounds = huc[huc.ID=='02040101'].geometry.bounds
    assert (np.allclose(bounds8_ll, np.array(bounds), 1.e-6))

def test_wbd_waterdata_get_many() -> None:
    wbd = ManagerWBD(protocol_name='WaterData')
    wbd.setLevel(12)
    huc = wbd.getShapesByID('02040101')
    print(huc)
    assert len(huc) == len(set(huc.ID)) # unique
    assert all(l.startswith('02040101') for l in huc.ID) # all in the HUC8
    assert (len(huc) == 38) # right number

def test_wbd_waterdata_get_geometry() -> None:
    wbd = ManagerWBD(protocol_name='WaterData')
    wbd.setLevel(8)
    shp = shapely.geometry.box(*bounds8_ll)
    huc = wbd.getShapesByGeometry(shp, watershed_workflow.crs.latlon_crs)
    huc = huc.to_crs(watershed_workflow.crs.latlon_crs)
    huc = huc[[shp.buffer(0.001).contains(h) for h in huc.geometry]]
    assert len(huc) == 1
    

def test_constructor_properties() -> None:
    """Test that constructor sets properties correctly"""
    wbd = ManagerWBD()
    assert wbd.name == 'WBD'  # Name should be WBD regardless of protocol
    assert wbd.source == 'HyRiver.WBD'
    assert wbd.native_crs_in == watershed_workflow.crs.latlon_crs
    assert wbd._protocol_name == 'WBD'  # Protocol name is the string
    assert wbd._protocol.__name__ == 'WBD'  # _protocol is the class

def test_constructor_waterdata_properties() -> None:
    """Test that constructor with WaterData protocol sets properties correctly"""
    wbd = ManagerWBD(protocol_name='WaterData')
    assert wbd.name == 'WBD'  # Name should still be WBD
    assert wbd.source == 'HyRiver.WaterData'
    assert wbd._protocol_name == 'WaterData'
    assert wbd._protocol.__name__ == 'WaterData'

def test_standard_naming_applied() -> None:
    """Test that standard naming is properly applied"""
    wbd = ManagerWBD()
    wbd.setLevel(8)
    huc = wbd.getShapesByID('02040101')
    
    # Check required standard columns
    assert names.ID in huc.columns
    assert names.HUC in huc.columns
    assert names.AREA in huc.columns
    
    # Check that ID values are strings (as specified in _addStandardNames)
    assert huc[names.ID].dtype == 'string'
    
    # Check metadata
    assert huc.attrs['name'] == wbd.name
    assert huc.attrs['source'] == wbd.source

def test_geodataframe_input() -> None:
    """Test getShapesByGeometry with GeoDataFrame input"""
    wbd = ManagerWBD()
    wbd.setLevel(8)
    
    # Create GeoDataFrame from bounds
    shp = shapely.geometry.box(*bounds8_ll)
    gdf = gpd.GeoDataFrame([{'test': 1}], geometry=[shp], crs=watershed_workflow.crs.latlon_crs)
    
    huc = wbd.getShapesByGeometry(gdf)
    
    assert isinstance(huc, gpd.GeoDataFrame)
    assert len(huc) >= 1
    assert names.ID in huc.columns
    assert names.HUC in huc.columns

