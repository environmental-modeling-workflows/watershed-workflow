import pytest
import numpy as np
import xarray as xr
import shapely.geometry

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs
import watershed_workflow.warp
from watershed_workflow.crs import CRS

from watershed_workflow.sources.manager_3dep import Manager3DEP


@pytest.fixture
def small_test_geometry():
    """Small test geometry in lat/lon for fast testing."""
    # Small box in Colorado in lat/lon (WGS84)
    return shapely.geometry.box(-105.5, 39.5, -105.4, 39.6)

@pytest.fixture  
def small_test_crs():
    """CRS for the small test geometry."""
    return CRS.from_epsg(4326)  # WGS84


def test_getDataset_single_variable(small_test_geometry, small_test_crs):
    """Test getDataset with single variable returns proper Dataset."""
    mgr = Manager3DEP(resolution=60)
    
    # Use the proper public API
    result = mgr.getDataset(small_test_geometry, small_test_crs, variables=['DEM'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 1
    assert 'dem' in result.data_vars  # Single variable gets converted to dataset
    assert result.rio.crs == CRS.from_epsg(5070)
    assert result.attrs['name'] == '3DEP'
    assert result.attrs['source'] == 'py3dep'


def test_getDataset_multiple_variables(small_test_geometry, small_test_crs):
    """Test getDataset with multiple variables returns Dataset."""
    mgr = Manager3DEP(resolution=60)
    
    # Use the proper public API with multiple variables
    result = mgr.getDataset(small_test_geometry, small_test_crs, 
                           variables=['DEM', 'Slope Degrees'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 2
    # py3dep returns different variable names for multi-variable requests
    assert 'elevation' in result.data_vars  
    assert 'slope_degrees' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(5070)
    assert result.attrs['name'] == '3DEP'
    assert result.attrs['source'] == 'py3dep'


def test_getDataset_resolution_consistency(small_test_geometry, small_test_crs):
    """Test that different resolution managers produce appropriately sized data."""
    mgr_30m = Manager3DEP(resolution=30)
    mgr_60m = Manager3DEP(resolution=60)
    
    result_30m = mgr_30m.getDataset(small_test_geometry, small_test_crs, variables=['DEM'])
    result_60m = mgr_60m.getDataset(small_test_geometry, small_test_crs, variables=['DEM'])
    
    # Higher resolution should have more pixels for same geometry
    assert result_30m.sizes['x'] > result_60m.sizes['x']
    assert result_30m.sizes['y'] > result_60m.sizes['y']
    
    # Both should have valid data
    assert not np.all(np.isnan(result_30m['dem'].values))
    assert not np.all(np.isnan(result_60m['dem'].values))


def test_getDataset_native_crs_usage(small_test_geometry, small_test_crs):
    """Test that getDataset correctly uses native CRS."""
    mgr = Manager3DEP(resolution=60)
    
    # Verify the manager has the expected native CRS
    assert mgr.native_crs_out == CRS.from_epsg(5070)
    
    result = mgr.getDataset(small_test_geometry, small_test_crs, variables=['DEM'])
    
    # Result should be in the native output CRS
    assert result.rio.crs == mgr.native_crs_out


def test_getDataset_aspect_and_slope_variables(small_test_geometry, small_test_crs):
    """Test specific 3DEP variables like aspect and slope."""
    mgr = Manager3DEP(resolution=60)
    
    # Test slope
    slope_result = mgr.getDataset(small_test_geometry, small_test_crs, variables=['Slope Degrees'])
    assert isinstance(slope_result, xr.Dataset)
    assert 'slope_degrees' in slope_result.data_vars
    
    # Test aspect  
    aspect_result = mgr.getDataset(small_test_geometry, small_test_crs, variables=['Aspect Degrees'])
    assert isinstance(aspect_result, xr.Dataset)
    assert 'aspect_degrees' in aspect_result.data_vars
    
    # Test multiple derived products
    multi_result = mgr.getDataset(small_test_geometry, small_test_crs, 
                                 variables=['DEM', 'Slope Degrees', 'Aspect Degrees'])
    assert len(multi_result.data_vars) == 3
    assert 'elevation' in multi_result.data_vars
    assert 'slope_degrees' in multi_result.data_vars  
    assert 'aspect_degrees' in multi_result.data_vars


def test_getDataset_coweeta_backward_compatibility(coweeta):
    """Test that getDataset works with existing coweeta fixture."""
    mgr = Manager3DEP(60)
    
    # Use the coweeta GeoDataFrame directly with the public API
    result = mgr.getDataset(coweeta, variables=['DEM'])
    
    assert isinstance(result, xr.Dataset)
    assert 'dem' in result.data_vars
    assert result.attrs['name'] == '3DEP'
    assert result.attrs['source'] == 'py3dep'
    
    # Extract DEM data for comparison
    dem_data = result['dem']
    # Shape may vary slightly due to different processing in new API
    assert dem_data.shape[0] > 90 and dem_data.shape[0] < 110
    assert dem_data.shape[1] > 90 and dem_data.shape[1] < 110
    assert abs(np.nanmean(dem_data.values) - 993) < 50  # Expected elevation range


def test_requestDataset_pattern(small_test_geometry, small_test_crs):
    """Test the non-blocking request/wait pattern."""
    mgr = Manager3DEP(resolution=60)
    
    # Request the dataset
    request = mgr.requestDataset(small_test_geometry, small_test_crs, variables=['DEM'])
    
    # Should be ready immediately for 3DEP
    assert mgr.isReady(request)
    
    # Fetch the data
    result = mgr.fetchRequest(request)
    
    assert isinstance(result, xr.Dataset)
    assert 'dem' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(5070)
    assert result.attrs['name'] == '3DEP'
    assert result.attrs['source'] == 'py3dep'


def test_waitForDataset_pattern(small_test_geometry, small_test_crs):
    """Test the request/wait pattern."""
    mgr = Manager3DEP(resolution=60)
    
    # Request the dataset
    request = mgr.requestDataset(small_test_geometry, small_test_crs, variables=['DEM'])
    
    # Wait for completion (should be immediate)
    result = mgr.waitForDataset(request, interval=1, tries=3)
    
    assert isinstance(result, xr.Dataset)
    assert 'dem' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(5070)
    assert result.attrs['name'] == '3DEP'
    assert result.attrs['source'] == 'py3dep'


def test_default_variables(small_test_geometry, small_test_crs):
    """Test that default variables work when none specified."""
    mgr = Manager3DEP(resolution=60)
    
    # Don't specify variables - should use defaults
    result = mgr.getDataset(small_test_geometry, small_test_crs)
    
    assert isinstance(result, xr.Dataset)
    assert 'dem' in result.data_vars
    assert len(result.data_vars) == 1  # Only DEM by default


def test_invalid_variable_error(small_test_geometry, small_test_crs):
    """Test that invalid variables raise appropriate errors."""
    mgr = Manager3DEP(resolution=60)
    
    with pytest.raises(ValueError, match="Invalid variable"):
        mgr.getDataset(small_test_geometry, small_test_crs, variables=['InvalidVariable'])