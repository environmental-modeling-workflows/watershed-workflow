import pytest
import numpy as np
import xarray as xr
import shapely.geometry

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs
import watershed_workflow.warp
from watershed_workflow.crs import CRS

from watershed_workflow.sources.manager_nlcd import ManagerNLCD


@pytest.fixture
def small_test_geometry():
    """Small test geometry in WGS84 (native CRS) for fast testing."""
    # Small box in North Carolina in WGS84 coordinates
    return shapely.geometry.box(-83.0, 35.0, -82.9, 35.1)


def test_getDataset_single_variable(small_test_geometry):
    """Test getDataset with single variable returns proper Dataset."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Use the public interface
    result = mgr.getDataset(small_test_geometry, geometry_crs, variables=['cover'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 1
    assert 'cover' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(4326)
    
    # Check metadata (base class adds name/source, NLCD adds year/location)
    assert result.attrs['nlcd_year'] == 2019
    assert result.attrs['nlcd_location'] == 'L48'
    assert result.attrs['name'] == 'NLCD 2019 L48'
    assert result.attrs['source'] == 'pygeohydro'


def test_getDataset_multiple_variables(small_test_geometry):
    """Test getDataset with multiple variables returns Dataset."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Use the public interface with multiple variables
    result = mgr.getDataset(small_test_geometry, geometry_crs, 
                            variables=['cover', 'impervious'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 2
    assert 'cover' in result.data_vars
    assert 'impervious' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(4326)


def test_getDataset_all_variables(small_test_geometry):
    """Test getDataset with all NLCD variables."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    all_vars = ['cover', 'impervious', 'canopy', 'descriptor']
    result = mgr.getDataset(small_test_geometry, geometry_crs, variables=all_vars)
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 4
    for var in all_vars:
        assert var in result.data_vars


def test_different_years_and_locations():
    """Test different year and location combinations."""
    small_geom = shapely.geometry.box(-83.0, 35.0, -82.9, 35.1)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test different years for L48
    mgr_2021 = ManagerNLCD(location='L48', year=2021)
    mgr_2016 = ManagerNLCD(location='L48', year=2016)
    
    result_2021 = mgr_2021.getDataset(small_geom, geometry_crs, variables=['cover'])
    result_2016 = mgr_2016.getDataset(small_geom, geometry_crs, variables=['cover'])
    
    assert result_2021.attrs['nlcd_year'] == 2021
    assert result_2016.attrs['nlcd_year'] == 2016
    
    # Test Alaska (different location)
    ak_geom = shapely.geometry.box(-150.0, 64.0, -149.9, 64.1)
    mgr_ak = ManagerNLCD(location='AK', year=2016)
    
    result_ak = mgr_ak.getDataset(ak_geom, geometry_crs, variables=['cover'])
    assert result_ak.attrs['nlcd_location'] == 'AK'


def test_year_validation():
    """Test year validation for different locations."""
    # Valid years
    mgr_l48 = ManagerNLCD(location='L48', year=2019)  # Should work
    mgr_ak = ManagerNLCD(location='AK', year=2016)    # Should work
    
    # Invalid years should raise ValueError
    with pytest.raises(ValueError, match='NLCD invalid year'):
        ManagerNLCD(location='L48', year=1999)  # Too early
    
    with pytest.raises(ValueError, match='NLCD invalid year'):
        ManagerNLCD(location='AK', year=2019)   # Not available for AK


def test_location_validation():
    """Test location validation."""
    # Valid locations
    ManagerNLCD(location='L48')  # Should work
    ManagerNLCD(location='AK')   # Should work
    ManagerNLCD(location='HI')   # Should work
    ManagerNLCD(location='PR')   # Should work
    
    # Invalid location should raise ValueError
    with pytest.raises(ValueError, match='NLCD invalid location'):
        ManagerNLCD(location='INVALID')


def test_default_year_selection():
    """Test that None year selects most recent available."""
    mgr_l48 = ManagerNLCD(location='L48', year=None)
    mgr_ak = ManagerNLCD(location='AK', year=None)
    
    # L48 should default to most recent (2021)
    assert mgr_l48.year == 2021
    
    # AK should default to most recent (2016)
    assert mgr_ak.year == 2016


def test_native_properties():
    """Test native data properties."""
    mgr = ManagerNLCD()
    
    # Check native properties
    assert mgr.native_crs_in == CRS.from_epsg(4326)  # WGS84
    assert mgr.native_crs_out == CRS.from_epsg(4326) # WGS84
    assert mgr.native_resolution == 0.00027          # ~30m in degrees
    assert mgr.native_start is None                  # Non-temporal
    assert mgr.native_end is None                    # Non-temporal
    
    # Check variables
    expected_valid = ['cover', 'impervious', 'canopy', 'descriptor']
    expected_default = ['cover']
    assert mgr.valid_variables == expected_valid
    assert mgr.default_variables == expected_default


def test_getDataset_coweeta_integration(coweeta):
    """Test that getDataset works with existing coweeta fixture."""
    mgr = ManagerNLCD(location='L48', year=2019)
    
    # Use first geometry from coweeta with proper CRS
    geom = coweeta.geometry.iloc[0]
    geom_crs = coweeta.crs
    
    result = mgr.getDataset(geom, geom_crs, variables=['cover'])
    
    assert isinstance(result, xr.Dataset)
    assert 'cover' in result.data_vars
    assert result.attrs['nlcd_year'] == 2019
    
    # Check that we got reasonable land cover data
    cover_data = result['cover']
    assert not np.all(np.isnan(cover_data.values))
    # NLCD values should be in valid range (0-95 for land cover classes)
    valid_data = cover_data.values[~np.isnan(cover_data.values)]
    assert np.all((valid_data >= 0) & (valid_data <= 127))  # Allow for water/no-data values


def test_non_temporal_behavior(small_test_geometry):
    """Test that NLCD correctly handles non-temporal data."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Base class should handle None start/end for non-temporal data
    result = mgr.getDataset(small_test_geometry, geometry_crs, variables=['cover'])
    
    # Should not have time dimension
    assert 'time' not in result.dims
    assert 'time' not in result.coords
    
    # Should be a spatial-only dataset
    assert 'x' in result.dims or 'longitude' in result.dims
    assert 'y' in result.dims or 'latitude' in result.dims


def test_async_interface(small_test_geometry):
    """Test the async interface methods."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test requestDataset
    request = mgr.requestDataset(small_test_geometry, geometry_crs, variables=['cover'])
    assert isinstance(request, mgr.Request)
    assert request.manager == mgr
    assert request.geometry is not None
    assert request.variables == ['cover']
    
    # Test isReady - should be immediately ready for NLCD data
    assert mgr.isReady(request) == True
    
    # Test fetchRequest
    result = mgr.fetchRequest(request)
    assert isinstance(result, xr.Dataset)
    assert 'cover' in result.data_vars
    assert result.attrs['nlcd_year'] == 2019


def test_default_variables(small_test_geometry):
    """Test that default variables work correctly."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test with no variables specified (should use defaults)
    result = mgr.getDataset(small_test_geometry, geometry_crs)
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 1
    assert 'cover' in result.data_vars  # Default variable


def test_invalid_variable_error(small_test_geometry):
    """Test that invalid variables raise appropriate errors."""
    mgr = ManagerNLCD(location='L48', year=2019)
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test invalid variable
    with pytest.raises(ValueError, match="Invalid variable"):
        mgr.getDataset(small_test_geometry, geometry_crs, variables=['invalid_var'])
    
    # Test mix of valid and invalid variables
    with pytest.raises(ValueError, match="Invalid variable"):
        mgr.getDataset(small_test_geometry, geometry_crs, variables=['cover', 'invalid_var'])
