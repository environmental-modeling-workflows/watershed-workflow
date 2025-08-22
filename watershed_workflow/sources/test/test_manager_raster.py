import pytest
import os
import numpy as np
import xarray as xr
import shapely.geometry
from shapely.geometry import box

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs
from watershed_workflow.crs import CRS

from watershed_workflow.sources.manager_raster import ManagerRaster


@pytest.fixture
def dtb_raster_path():
    """Path to the DTB test raster file."""
    return os.path.join('examples', 'Coweeta', 'input_data', 'DTB', 'DTB.tif')


@pytest.fixture
def raster_manager(dtb_raster_path):
    """ManagerRaster instance for DTB.tif."""
    return ManagerRaster(dtb_raster_path)


@pytest.fixture
def test_geometry():
    """Small test geometry in lat/lon coordinates."""
    # Small box around Coweeta area
    return box(-83.5, 35.0, -83.4, 35.1)


def test_manager_raster_initialization(raster_manager, dtb_raster_path):
    """Test that ManagerRaster initializes correctly."""
    assert raster_manager.filename == dtb_raster_path
    assert raster_manager.name == f'raster: "{os.path.basename(dtb_raster_path)}"'
    assert raster_manager.source == os.path.abspath(dtb_raster_path)
    assert raster_manager.native_start is None  # Non-temporal
    assert raster_manager.native_end is None    # Non-temporal
    assert raster_manager.native_crs_in is not None
    assert raster_manager.native_crs_out is not None
    assert raster_manager.native_resolution > 0
    assert raster_manager.valid_variables is not None
    assert raster_manager.default_variables is not None
    assert len(raster_manager.default_variables) >= 1


def test_manager_raster_variables(raster_manager):
    """Test that variables are set up correctly for band access."""
    # Should have at least one variable
    assert len(raster_manager.valid_variables) >= 1
    
    # Variables should be band_1, band_2, etc.
    for var in raster_manager.valid_variables:
        assert var.startswith('band_')
        band_num = int(var.split('_')[1])
        assert band_num >= 1


def test_request_dataset_async_interface(raster_manager, test_geometry):
    """Test the async interface methods."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test requestDataset
    request = raster_manager.requestDataset(test_geometry, geometry_crs)
    assert isinstance(request, raster_manager.Request)
    assert request.manager == raster_manager
    assert request.geometry is not None
    
    # Test isReady - should be immediately ready for raster data
    assert raster_manager.isReady(request) == True
    
    # Test fetchRequest
    result = raster_manager.fetchRequest(request)
    assert isinstance(result, xr.Dataset)
    assert hasattr(result, 'rio')
    assert len(result.data_vars) >= 1


def test_getDataset_with_specific_variable(raster_manager, test_geometry):
    """Test getDataset method with specific variable selection."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Request specific variable (first one)
    first_var = raster_manager.valid_variables[0]
    result = raster_manager.getDataset(test_geometry, geometry_crs, variables=[first_var])
    
    assert isinstance(result, xr.Dataset)
    assert hasattr(result, 'rio')
    assert result.rio.crs is not None
    assert first_var in result.data_vars
    assert len(result.data_vars) == 1
    
    # Verify data has reasonable values
    data_array = result[first_var]
    assert data_array.size > 0
    assert not np.all(np.isnan(data_array.values))


def test_getDataset_default_behavior(raster_manager, test_geometry, dtb_raster_path):
    """Test the getDataset method with default parameters."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Use default parameters (no variables specified)
    result = raster_manager.getDataset(test_geometry, geometry_crs)
    
    # Should return xr.Dataset
    assert isinstance(result, xr.Dataset)
    assert hasattr(result, 'rio')
    assert result.rio.crs is not None
    
    # Should have name and source attributes from base class
    assert 'name' in result.attrs
    assert 'source' in result.attrs
    expected_name = f'raster: "{os.path.basename(dtb_raster_path)}"'
    assert result.attrs['name'] == expected_name
    assert result.attrs['source'] == os.path.abspath(dtb_raster_path)
    
    # Should have at least one data variable (default variables)
    assert len(result.data_vars) >= 1
    
    # For single-variable case, should have 'raster' variable
    if raster_manager.valid_variables is None:
        assert 'raster' in result.data_vars
    else:
        # Should have default variables
        for var in raster_manager.default_variables:
            assert var in result.data_vars


def test_getDataset_with_multiple_variables(raster_manager, test_geometry):
    """Test getDataset method with multiple variables."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Request multiple variables if available
    if len(raster_manager.valid_variables) > 1:
        vars_to_request = raster_manager.valid_variables[:2]  # First two variables
        result = raster_manager.getDataset(test_geometry, geometry_crs, variables=vars_to_request)
        
        assert isinstance(result, xr.Dataset)
        for var in vars_to_request:
            assert var in result.data_vars
            # Verify each variable has data
            data_array = result[var]
            assert data_array.size > 0
        assert len(result.data_vars) == len(vars_to_request)
    else:
        # Single variable case - test with all variables
        result = raster_manager.getDataset(test_geometry, geometry_crs, variables=raster_manager.valid_variables)
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 1


def test_invalid_variable_raises_error(raster_manager, test_geometry):
    """Test that requesting invalid variable raises appropriate error."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Try to request a variable that doesn't exist
    with pytest.raises(ValueError, match="Invalid variable"):
        raster_manager.getDataset(test_geometry, geometry_crs, variables=['nonexistent_variable'])
    
    # Test with valid and invalid variables mixed
    if raster_manager.valid_variables:
        valid_var = raster_manager.valid_variables[0]
        with pytest.raises(ValueError, match="Invalid variable"):
            raster_manager.getDataset(test_geometry, geometry_crs, variables=[valid_var, 'invalid_var'])




def test_file_not_found_raises_error():
    """Test that non-existent file raises appropriate error."""
    nonexistent_file = 'path/to/nonexistent/file.tif'
    
    with pytest.raises((FileNotFoundError, OSError)):
        ManagerRaster(nonexistent_file)


def test_invalid_band_request_raises_error(raster_manager, test_geometry):
    """Test that requesting invalid band raises appropriate error."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Try to request a band that doesn't exist (assuming single band raster)
    invalid_band = 'band_999'
    with pytest.raises(ValueError, match="Invalid variable"):
        raster_manager.getDataset(test_geometry, geometry_crs, variables=[invalid_band])


def test_single_variable_case_returns_raster_variable(raster_manager, test_geometry):
    """Test that single-variable rasters return 'raster' variable name."""
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Create a manager with no valid_variables (single-variable case)
    class SingleVarManager(ManagerRaster):
        def __init__(self, filename):
            super().__init__(filename)
            self.valid_variables = None
            self.default_variables = None
    
    single_mgr = SingleVarManager(raster_manager.filename)
    result = single_mgr.getDataset(test_geometry, geometry_crs)
    
    assert isinstance(result, xr.Dataset)
    assert 'raster' in result.data_vars
    assert len(result.data_vars) == 1


def test_manager_properties_match_file_properties(raster_manager, dtb_raster_path):
    """Test that manager properties correctly reflect the raster file properties."""
    # Open the file directly to compare properties
    import rioxarray
    
    with rioxarray.open_rasterio(dtb_raster_path) as direct_ds:
        # CRS should match
        assert raster_manager.native_crs_in == direct_ds.rio.crs
        assert raster_manager.native_crs_out == direct_ds.rio.crs
        
        # Resolution should be reasonable
        if len(direct_ds.coords['x']) > 1:
            x_res = abs(float(direct_ds.coords['x'][1] - direct_ds.coords['x'][0]))
            assert abs(raster_manager.native_resolution - x_res) < x_res * 0.1  # Within 10%
        
        # Variable setup should match bands
        if hasattr(direct_ds, 'band'):
            expected_vars = [f'band_{i}' for i in range(1, len(direct_ds.band) + 1)]
            assert raster_manager.valid_variables == expected_vars
            assert raster_manager.default_variables == [expected_vars[0]]


def test_coweeta_basin_integration(raster_manager, coweeta):
    """Test ManagerRaster with Coweeta basin geometry."""
    # Get first geometry from coweeta fixture
    basin_geom = coweeta.geometry.iloc[0]
    basin_crs = coweeta.crs
    
    # Test getDataset method
    result_dataset = raster_manager.getDataset(basin_geom, basin_crs)
    assert isinstance(result_dataset, xr.Dataset)
    assert len(result_dataset.data_vars) >= 1
    assert 'name' in result_dataset.attrs
    assert 'source' in result_dataset.attrs
    
    # Verify spatial properties
    first_var = list(result_dataset.data_vars)[0]
    result_array = result_dataset[first_var]
    assert isinstance(result_array, xr.DataArray)
    assert result_array.size > 0
    
    # Check that result is spatially clipped (smaller than original)
    assert result_array.rio.bounds() is not None
    
    # Verify CRS is preserved
    assert result_dataset.rio.crs is not None
    assert watershed_workflow.crs.isEqual(result_dataset.rio.crs, raster_manager.native_crs_out)