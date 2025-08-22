import pytest
import os
import shapely.geometry
import numpy as np
import datetime
import cftime
import xarray as xr
from unittest.mock import patch, MagicMock

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs
from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_dataset import ManagerDataset
from watershed_workflow.sources.manager_modis_appeears import ManagerMODISAppEEARS


@pytest.fixture
def modis_manager():
    """Create MODIS manager without authentication for testing."""
    import watershed_workflow.config
    watershed_workflow.config.setDataDirectory(os.path.join('examples', 'Coweeta', 'input_data'))
    return ManagerMODISAppEEARS()


@pytest.fixture
def coweeta_geometry():
    """Small test geometry matching the existing MODIS files."""
    # Geometry that matches the existing file bounds: 35.0133x-83.4943_35.0889x-83.408
    return shapely.geometry.box(-83.4943, 35.0133, -83.408, 35.0889)


@pytest.fixture
def coweeta_crs():
    """CRS for coweeta geometry (WGS84)."""
    return CRS.from_epsg(4269)


@pytest.fixture
def existing_lai_file():
    """Path to existing LAI file for testing."""
    return '/home/ecoon/code/watershed_workflow/repos/master/examples/Coweeta/input_data/land_cover/MODIS/modis_LAI_08-01-2010_08-01-2011_35.0889x-83.4943_35.0133x-83.408.nc'


@pytest.fixture
def existing_lulc_file():
    """Path to existing LULC file for testing."""
    return '/home/ecoon/code/watershed_workflow/repos/master/examples/Coweeta/input_data/land_cover/MODIS/modis_LULC_08-01-2010_08-01-2011_35.0889x-83.4943_35.0133x-83.408.nc'


@pytest.fixture
def coweeta_date_range():
    """Date range that matches existing files."""
    start = cftime.datetime(2010, 8, 1, calendar='standard')
    end = cftime.datetime(2011, 8, 1, calendar='standard')
    return start, end


def test_manager_properties(modis_manager):
    """Test basic manager properties and initialization."""
    assert modis_manager.name == 'MODIS'
    assert modis_manager.source == 'AppEEARS'
    assert modis_manager.native_crs_in == CRS.from_epsg(4269)
    assert modis_manager.native_crs_out == CRS.from_epsg(4269)
    assert modis_manager.native_resolution == 500.0
    assert set(modis_manager.valid_variables) == {'LAI', 'LULC'}
    assert set(modis_manager.default_variables) == {'LAI', 'LULC'}


def test_filename_generation(modis_manager):
    """Test filename generation for caching."""
    bounds = (-83.5, 35.0, -83.4, 35.1)
    filename = modis_manager._filename(bounds, '08-01-2010', '08-01-2011', 'LAI')
    
    assert 'modis_LAI_08-01-2010_08-01-2011' in filename
    assert '35.1x-83.5_35.0x-83.4.nc' in filename
    assert filename.endswith('.nc')


def test_read_existing_lai_file(modis_manager, existing_lai_file):
    """Test reading existing LAI file."""
    if not os.path.exists(existing_lai_file):
        pytest.skip("LAI test file not available")
    
    data_array = modis_manager._readFile(existing_lai_file, 'LAI')
    
    assert isinstance(data_array, xr.DataArray)
    assert data_array.name == 'LAI'
    assert data_array.rio.crs is not None
    assert 'time' in data_array.dims
    assert 'lat' in data_array.dims or 'latitude' in data_array.dims
    assert 'lon' in data_array.dims or 'longitude' in data_array.dims


def test_read_existing_lulc_file(modis_manager, existing_lulc_file):
    """Test reading existing LULC file."""
    if not os.path.exists(existing_lulc_file):
        pytest.skip("LULC test file not available")
    
    data_array = modis_manager._readFile(existing_lulc_file, 'LULC')
    
    assert isinstance(data_array, xr.DataArray)
    assert data_array.name == 'LULC'
    assert data_array.rio.crs is not None


def test_read_dataset_multiple_variables(modis_manager, existing_lai_file, existing_lulc_file):
    """Test reading dataset with multiple variables from existing files."""
    # Create a mock request with both variables
    request = MagicMock()
    request.variables = ['LAI', 'LULC']
    request.filenames = {
        'LAI': existing_lai_file,
        'LULC': existing_lulc_file
    }
    
    dataset = modis_manager._readData(request)
    
    assert isinstance(dataset, xr.Dataset)
    assert 'LAI' in dataset.data_vars
    assert 'LULC' in dataset.data_vars
    assert len(dataset.data_vars) == 2


def test_fetch_existing_dataset(modis_manager, existing_lai_file, existing_lulc_file):
    """Test _fetchDataset with existing files."""
    # Create a mock request with existing files
    request = modis_manager.Request(
        manager=modis_manager,
        is_ready=True,
        geometry=shapely.geometry.box(-83.5, 35.0, -83.4, 35.1),
        start=cftime.datetime(2010, 8, 1, calendar='standard'),
        end=cftime.datetime(2011, 8, 1, calendar='standard'),
        variables=['LAI', 'LULC'],
        task_id="",
        filenames={
            'LAI': existing_lai_file,
            'LULC': existing_lulc_file
        },
        urls={}
    )
    
    dataset = modis_manager._fetchDataset(request)
    
    assert isinstance(dataset, xr.Dataset)
    assert 'LAI' in dataset.data_vars
    assert 'LULC' in dataset.data_vars


def test_getDataset_with_existing_files(modis_manager, coweeta_geometry, coweeta_crs, coweeta_date_range, existing_lai_file, existing_lulc_file):
    """Test blocking getDataset with existing files."""
    start, end = coweeta_date_range
    
    request_in = ManagerDataset.Request(modis_manager, False, coweeta_geometry, start, end, ['LAI'])
    request_out = modis_manager._requestDataset(request_in)
    dataset = modis_manager.fetchRequest(request_out)
        
    assert isinstance(dataset, xr.Dataset)
    assert 'LAI' in dataset.data_vars
    assert dataset.attrs['name'] == 'MODIS'
    assert dataset.attrs['source'] == 'AppEEARS'


def test_request_fetch_pattern(modis_manager, coweeta_geometry, coweeta_crs, coweeta_date_range, existing_lai_file):
    """Test non-blocking request/fetch pattern."""
    if not os.path.exists(existing_lai_file):
        pytest.skip("Test file not available")
    
    start, end = coweeta_date_range
    
    # Mock filename generation
    with patch.object(modis_manager, '_filename') as mock_filename:
        mock_filename.return_value = existing_lai_file
        
        # Request dataset
        request = modis_manager.requestDataset(
            coweeta_geometry, coweeta_crs,
            start=start, end=end,
            variables=['LAI']
        )
        
        # Should be ready immediately
        assert modis_manager.isReady(request)
        
        # Fetch dataset
        dataset = modis_manager.fetchRequest(request)
        
        assert isinstance(dataset, xr.Dataset)
        assert 'LAI' in dataset.data_vars


def test_invalid_variable_error(modis_manager, coweeta_geometry, coweeta_crs, coweeta_date_range):
    """Test error handling for invalid variables."""
    start, end = coweeta_date_range
    
    with pytest.raises(ValueError, match="Invalid variable"):
        modis_manager.getDataset(
            coweeta_geometry, coweeta_crs,
            start=start, end=end,
            variables=['InvalidVariable']
        )


def test_date_validation_errors(modis_manager, coweeta_geometry, coweeta_crs):
    """Test date validation against MODIS bounds."""
    # Test start date before MODIS start (2002)
    with pytest.raises(ValueError, match="Start date .* is before dataset start"):
        modis_manager.getDataset(
            coweeta_geometry, coweeta_crs,
            start='2000-01-01', end='2002-08-01',
            variables=['LAI']
        )
    
    # Test end date after MODIS end (2021)
    with pytest.raises(ValueError, match="End date .* is after dataset end"):
        modis_manager.getDataset(
            coweeta_geometry, coweeta_crs,
            start='2020-01-01', end='2025-01-01',
            variables=['LAI']
        )


def test_default_variables(modis_manager, coweeta_geometry, coweeta_crs, coweeta_date_range, existing_lai_file, existing_lulc_file):
    """Test that default variables work when none specified."""
    start, end = coweeta_date_range
    
    # Mock filename generation
    with patch.object(modis_manager, '_filename') as mock_filename:
        mock_filename.side_effect = lambda bounds, start_str, end_str, var: {
            'LAI': existing_lai_file,
            'LULC': existing_lulc_file
        }[var]
        
        # Don't specify variables - should use defaults
        dataset = modis_manager.getDataset(
            coweeta_geometry, coweeta_crs,
            start=start, end=end
        )
        
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) == 2  # Both LAI and LULC by default
        assert 'LAI' in dataset.data_vars
        assert 'LULC' in dataset.data_vars


@patch('watershed_workflow.sources.manager_modis_appeears.requests.post')
def test_construct_request_mocked(mock_post, modis_manager):
    """Test _constructRequest with mocked API call."""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = {'task_id': 'test_task_123'}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    # Mock authentication
    modis_manager.login_token = 'fake_token'
    
    bounds = (-83.5, 35.0, -83.4, 35.1)
    task_id = modis_manager._constructRequest(bounds, '08-01-2010', '08-01-2011', ['LAI'])
    
    assert task_id == 'test_task_123'
    assert mock_post.called
    
    # Verify the request was constructed correctly
    call_args = mock_post.call_args
    assert call_args[1]['json']['task_type'] == 'area'
    assert call_args[1]['json']['params']['layers'][0]['product'] == 'MCD15A3H.061'


def test_missing_files_error(modis_manager):
    """Test behavior when expected files don't exist."""
    # Create request with non-existent files
    request = modis_manager.Request(
        manager=modis_manager,
        is_ready=True,
        geometry=shapely.geometry.box(-83.5, 35.0, -83.4, 35.1),
        start=cftime.datetime(2010, 8, 1, calendar='standard'),
        end=cftime.datetime(2011, 8, 1, calendar='standard'),
        variables=['LAI'],
        task_id="",
        filenames={'LAI': '/nonexistent/file.nc'},
        urls={}
    )
    
    with pytest.raises(Exception):  # Should raise some kind of file not found error
        modis_manager._fetchDataset(request)
