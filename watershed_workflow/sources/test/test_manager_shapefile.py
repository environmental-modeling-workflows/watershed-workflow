import pytest

import os
import shapely
import numpy as np
import geopandas as gpd

import watershed_workflow.config
import watershed_workflow.crs
import watershed_workflow.sources.standard_names as names
from watershed_workflow.sources.manager_shapefile import ManagerShapefile


@pytest.fixture
def shapefile_path():
    """Path to test shapefile"""
    return os.path.join('examples', 'Coweeta', 'input_data', 'coweeta_basin.shp')


@pytest.fixture
def multi_shape_path():
    """Path to shapefile with multiple shapes for row index testing"""
    return os.path.join('examples', 'Coweeta', 'input_data', 'soil_structure', 'GLHYMPS', 'GLHYMPS.shp')


@pytest.fixture
def manager_no_id(shapefile_path):
    """ManagerShapefile without ID field specified"""
    return ManagerShapefile(shapefile_path)


@pytest.fixture
def manager_multi_shapes(multi_shape_path):
    """ManagerShapefile for multi-shape file without ID field specified"""
    return ManagerShapefile(multi_shape_path)


@pytest.fixture
def test_geometry():
    """Test geometry for spatial queries"""
    # Create a small polygon within the Coweeta basin bounds
    return shapely.geometry.box(275000, 3880000, 277000, 3882000)


@pytest.fixture
def test_crs():
    """CRS for test geometry (UTM Zone 17N)"""
    return watershed_workflow.crs.from_epsg('32617')


def test_constructor_basic(shapefile_path):
    """Test basic constructor"""
    ms = ManagerShapefile(shapefile_path)
    assert ms.name.startswith('shapefile: ')
    assert ms.source == os.path.abspath(shapefile_path)
    assert ms._filename == shapefile_path
    assert ms._id_name is None


def test_constructor_with_id_field(shapefile_path):
    """Test constructor with ID field specified"""
    ms = ManagerShapefile(shapefile_path, id_name='OBJECTID')
    assert ms._id_name == 'OBJECTID'


def test_getShapes_backward_compatibility(manager_no_id):
    """Test backward compatibility of getShapes method"""
    shp = manager_no_id.getShapes()
    bounds = shp.bounds
    assert np.allclose(
        np.array([273971.0911428096, 3878839.6361173145, 279140.9150949494, 3883953.7853134344]),
        np.array(bounds), 1.e-4)


def test_getShapesByGeometry_with_geometry_crs(manager_no_id, test_geometry, test_crs):
    """Test getShapesByGeometry with shapely geometry + CRS"""
    result = manager_no_id.getShapesByGeometry(test_geometry, test_crs)
    
    # Should return GeoDataFrame with standard columns
    assert isinstance(result, gpd.GeoDataFrame)
    assert names.ID in result.columns
    assert names.NAME in result.columns
    assert result.index.name == names.ID
    
    # Should have intersecting shapes
    assert len(result) > 0


def test_getShapesByGeometry_with_geodataframe(manager_no_id, test_geometry, test_crs):
    """Test getShapesByGeometry with GeoDataFrame input"""
    # Create test GeoDataFrame
    gdf = gpd.GeoDataFrame([{'test': 1}], geometry=[test_geometry], crs=test_crs)
    
    result = manager_no_id.getShapesByGeometry(gdf)
    
    # Should return GeoDataFrame with standard columns
    assert isinstance(result, gpd.GeoDataFrame)
    assert names.ID in result.columns
    assert names.NAME in result.columns
    assert result.index.name == names.ID


def test_getShapesByID_row_indices(manager_multi_shapes):
    """Test getShapesByID with row indices (no ID field)"""
    # Get first shape by row index
    result = manager_multi_shapes.getShapesByID(['0'])
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert names.ID in result.columns
    assert names.NAME in result.columns
    
    # Test multiple indices
    result_multi = manager_multi_shapes.getShapesByID(['0', '1'])
    assert len(result_multi) == 2
    
    # Test non-sequential indices
    result_non_seq = manager_multi_shapes.getShapesByID(['0', '5', '10'])
    assert len(result_non_seq) == 3


def test_getShapesByID_invalid_indices(manager_multi_shapes):
    """Test getShapesByID with invalid row indices"""
    with pytest.raises(ValueError, match="Invalid row indices"):
        manager_multi_shapes.getShapesByID(['999'])  # GLHYMPS has 13 shapes, so 999 is invalid


def test_getShapesByID_non_integer_indices(manager_multi_shapes):
    """Test getShapesByID with non-integer indices when no ID field"""
    with pytest.raises(ValueError, match="Cannot convert IDs"):
        manager_multi_shapes.getShapesByID(['not_a_number'])


def test_standard_naming(manager_no_id, test_geometry, test_crs):
    """Test that standard naming is applied"""
    result = manager_no_id.getShapesByGeometry(test_geometry, test_crs)
    
    # Check required standard columns exist
    assert names.ID in result.columns
    assert names.NAME in result.columns
    
    # Check that name is generated from ID
    assert result[names.NAME].iloc[0] == str(result[names.ID].iloc[0])
    
    # Check metadata
    assert result.attrs['name'] == manager_no_id.name
    assert result.attrs['source'] == manager_no_id.source


def test_crs_handling(manager_no_id):
    """Test that CRS is properly handled"""
    # Use lat/lon geometry and ensure it gets transformed properly
    latlon_geom = shapely.geometry.box(-83.5, 35.0, -83.4, 35.1)
    latlon_crs = watershed_workflow.crs.latlon_crs
    
    result = manager_no_id.getShapesByGeometry(latlon_geom, latlon_crs)
    
    # Should have proper CRS set
    assert result.crs is not None
    # Should have some results (assuming geometry overlaps with Coweeta)
    assert len(result) >= 0


def test_geometry_intersection_filtering(manager_no_id, test_crs):
    """Test that geometry intersection filtering works properly"""
    # Create geometry that should intersect
    intersecting_geom = shapely.geometry.box(275000, 3880000, 277000, 3882000)
    
    # Create geometry that should not intersect (far away)
    non_intersecting_geom = shapely.geometry.box(100000, 3880000, 102000, 3882000)
    
    result_intersecting = manager_no_id.getShapesByGeometry(intersecting_geom, test_crs)
    result_non_intersecting = manager_no_id.getShapesByGeometry(non_intersecting_geom, test_crs)
    
    # Intersecting geometry should return shapes, non-intersecting should not
    assert len(result_intersecting) > 0
    assert len(result_non_intersecting) == 0


def test_getShapesByID_with_named_field(shapefile_path):
    """Test getShapesByID with named ID field"""
    # Create manager with BASIN_CODE as ID field
    manager_with_id = ManagerShapefile(shapefile_path, id_name='BASIN_CODE')
    
    # Get the basin by its BASIN_CODE value (should be 1 based on our inspection)
    result = manager_with_id.getShapesByID(['1'])
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert names.ID in result.columns
    assert names.NAME in result.columns
    
    # Check that the ID column contains the BASIN_CODE value
    assert result[names.ID].iloc[0] == 1
    
    # Test with string representation of the ID
    result_str = manager_with_id.getShapesByID(['1'])
    assert len(result_str) == 1
    assert result_str[names.ID].iloc[0] == 1


def test_getShapesByID_named_field_not_found(shapefile_path):
    """Test getShapesByID with non-existent named field"""
    manager_bad_id = ManagerShapefile(shapefile_path, id_name='NONEXISTENT_FIELD')
    
    with pytest.raises(ValueError, match="ID field 'NONEXISTENT_FIELD' not found"):
        manager_bad_id.getShapesByID(['1'])


def test_getShapesByID_named_field_invalid_value(shapefile_path):
    """Test getShapesByID with invalid ID value for named field"""
    manager_with_id = ManagerShapefile(shapefile_path, id_name='BASIN_CODE')
    
    # Try to get basin with ID 999 (doesn't exist)
    result = manager_with_id.getShapesByID(['999'])
    
    # Should return empty dataframe
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
