import pytest
import os
import numpy as np
import shapely.geometry
import geopandas as gpd

import watershed_workflow.config
import watershed_workflow.crs
import watershed_workflow.sources.standard_names as names


pytest.skip("Skipping all Basin-3D tests -- Basin-3D is not in the default environment.", allow_module_level=True)

import watershed_workflow.sources.manager_basin3d as manager_basin3d


@pytest.fixture
def basin3d_mgr():
    """Create Basin3D manager with USGS plugin."""
    return manager_basin3d.ManagerBasin3D()


@pytest.fixture 
def basin3d_mgr_multi():
    """Create Basin3D manager with multiple feature types."""
    return manager_basin3d.ManagerBasin3D(feature_types=['point', 'site'])


def test_constructor(basin3d_mgr):
    """Test basic constructor and properties."""
    assert basin3d_mgr.name == 'Basin3D-usgs'
    assert 'Basin3D with plugins: usgs' in basin3d_mgr.source
    assert basin3d_mgr.native_crs_in == watershed_workflow.crs.from_epsg(4326)
    assert basin3d_mgr.native_id_field == 'id'
    assert basin3d_mgr.feature_types == ['point']
    assert basin3d_mgr.data_sources == ['usgs']


def test_constructor_multi_features(basin3d_mgr_multi):
    """Test constructor with multiple feature types."""
    assert basin3d_mgr_multi.feature_types == ['point', 'site']
    assert basin3d_mgr_multi.data_sources == ['usgs']


def test_constructor_invalid_feature_type():
    """Test constructor with invalid feature type."""
    with pytest.raises(ValueError, match="Invalid feature_type"):
        manager_basin3d.ManagerBasin3D(feature_types=['INVALID_TYPE'])


def test_constructor_invalid_data_source():
    """Test constructor with no valid plugins."""
    with pytest.raises(ValueError, match="No valid plugins found"):
        manager_basin3d.ManagerBasin3D(data_sources=['invalid_source'])


def test_plugin_registration(basin3d_mgr):
    """Test that Basin3D synthesizer was registered correctly."""
    assert hasattr(basin3d_mgr, 'synthesizer')
    assert basin3d_mgr.synthesizer is not None
    assert len(basin3d_mgr.plugin_classes) == 1
    assert 'basin3d.plugins.usgs.USGSDataSourcePlugin' in basin3d_mgr.plugin_classes


def test_getShapesByGeometry(basin3d_mgr):
    """Test geometry-based query returns proper GeoDataFrame."""

    polygon = shapely.geometry.box(-90.6, 34.4, -90.5, 34.6)
    result = basin3d_mgr.getShapesByGeometry(polygon, watershed_workflow.crs.latlon_crs)
    
    # Check result structure
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == basin3d_mgr.native_crs_in
    
    # Check expected columns
    expected_cols = ['id', 'name', 'feature_type', 'description', 'data_source', 'elevation', 'geometry']
    for col in expected_cols:
        assert col in result.columns
    
    # Check standard names were added
    if names.ID in result.columns:
        assert names.ID in result.columns
    if names.NAME in result.columns:
        assert names.NAME in result.columns

    assert len(result) == 2


def test_getShapesByID(basin3d_mgr):
    """Test geometry-based query returns proper GeoDataFrame."""

    result = basin3d_mgr.getShapesByID(['USGS-13010000', 'USGS-385508107021201',])
    
    # Check result structure
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == basin3d_mgr.native_crs_in
    
    # Check expected columns
    expected_cols = ['id', 'name', 'feature_type', 'description', 'data_source', 'elevation', 'geometry']
    for col in expected_cols:
        assert col in result.columns
    
    # Check standard names were added
    if names.ID in result.columns:
        assert names.ID in result.columns
    if names.NAME in result.columns:
        assert names.NAME in result.columns

    print(result)
    assert len(result) == 2

    
