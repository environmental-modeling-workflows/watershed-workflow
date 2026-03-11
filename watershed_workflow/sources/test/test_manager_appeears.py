import pytest
import os
import shapely.geometry
import numpy as np
import datetime
import cftime
import xarray as xr
from unittest.mock import patch, MagicMock

import watershed_workflow.crs
from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_dataset import ManagerDataset
from watershed_workflow.sources.manager_modis_appeears import ManagerMODISAppEEARS


@pytest.fixture
def modis_manager():
    """Create MODIS manager without authentication for testing."""
    import watershed_workflow.utils.config
    watershed_workflow.utils.config.setDataDirectory(os.path.join('examples', 'Coweeta', 'input_data'))
    return ManagerMODISAppEEARS()


@pytest.fixture
def existing_lai_file():
    """Path to existing LAI file for testing."""
    return './examples/Coweeta/input_data/land_cover/MODIS/modis_LAI_08-01-2010_08-01-2011_35.0782x-83.4826_35.0237x-83.4178.nc'


@pytest.fixture
def existing_lulc_file():
    """Path to existing LULC file for testing."""
    return './examples/Coweeta/input_data/land_cover/MODIS/modis_LULC_08-01-2010_08-01-2011_35.0782x-83.4826_35.0237x-83.4178.nc'


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
    assert set(modis_manager.valid_variables) == {'LAI', 'LULC'}
    assert set(modis_manager.default_variables) == {'LAI', 'LULC'}


def test_filename_generation(modis_manager):
    """Test cache filename generation via the standard Manager scheme."""
    bounds = (-83.5, 35.0, -83.4, 35.1)
    filename = modis_manager._cacheFilename(bounds, var='LAI', start_year=2010, end_year=2011)

    assert 'LAI' in filename
    assert '2010-2011' in filename
    assert filename.endswith('.nc')
    assert 'MODIS' in os.path.basename(filename)


def test_read_existing_lai_file(modis_manager, existing_lai_file):
    """Test reading existing LAI file."""
    if not os.path.exists(existing_lai_file):
        pytest.skip("LAI test file not available")

    data_array = modis_manager._readFile(existing_lai_file, 'LAI')

    assert isinstance(data_array, xr.DataArray)
    assert data_array.name == 'LAI'
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


def test_read_dataset_multiple_variables(modis_manager, existing_lai_file, existing_lulc_file):
    """Test reading dataset with multiple variables from existing files."""
    if not os.path.exists(existing_lai_file) or not os.path.exists(existing_lulc_file):
        pytest.skip("Test files not available")

    # Create a mock request using the new cache_filenames attribute
    request = MagicMock()
    request.variables = ['LAI', 'LULC']
    request.cache_filenames = {
        'LAI': existing_lai_file,
        'LULC': existing_lulc_file
    }

    dataset = modis_manager._readData(request)

    assert isinstance(dataset, xr.Dataset)
    assert 'LAI' in dataset.data_vars
    assert 'LULC' in dataset.data_vars
    assert len(dataset.data_vars) == 2


def test_getDataset_with_existing_files(modis_manager, coweeta, coweeta_date_range, existing_lai_file, existing_lulc_file):
    """Test blocking getDataset with existing files.

    The existing files have old-style filenames, so the manager will not find
    them automatically by the standard cache name.  We skip this test when the
    files are absent, since the test would otherwise try to authenticate with
    AppEEARS.
    """
    bounds = (-83.493, 35.0145, -83.4075, 35.091)
    # Build the canonical cache filename that the manager would look for
    canonical_fname = modis_manager._cacheFilename(bounds, var='LAI', start_year=2010, end_year=2011)
    if not os.path.exists(canonical_fname):
        pytest.skip("Canonical MODIS cache file not available")

    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    start, end = coweeta_date_range

    request_out = modis_manager.requestDataset(coweeta_geometry, coweeta_crs, start, end, ['LAI'])
    dataset = modis_manager.fetchRequest(request_out)

    assert isinstance(dataset, xr.Dataset)
    assert 'LAI' in dataset.data_vars
    assert dataset.attrs['name'] == 'MODIS'
    assert dataset.attrs['source'] == 'AppEEARS'


def test_request_fetch_pattern(modis_manager, coweeta, coweeta_date_range, existing_lai_file):
    """Test non-blocking request/fetch pattern."""
    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]

    if not os.path.exists(existing_lai_file):
        pytest.skip("Test file not available")

    start, end = coweeta_date_range

    # Mock _cacheFilename to return the existing file path
    with patch.object(modis_manager, '_cacheFilename', return_value=existing_lai_file):
        request = modis_manager.requestDataset(
            coweeta_geometry, coweeta_crs,
            start=start, end=end,
            variables=['LAI']
        )

        assert modis_manager.isReady(request)

        dataset = modis_manager.fetchRequest(request)

        assert isinstance(dataset, xr.Dataset)
        assert 'LAI' in dataset.data_vars


def test_invalid_variable_error(modis_manager, coweeta, coweeta_date_range):
    """Test error handling for invalid variables."""
    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    start, end = coweeta_date_range

    with pytest.raises(ValueError, match="Invalid variable"):
        modis_manager.getDataset(
            coweeta_geometry, coweeta_crs,
            start=start, end=end,
            variables=['InvalidVariable']
        )


def test_date_validation_errors(modis_manager, coweeta):
    """Test date validation against MODIS bounds."""
    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    # Test start date before MODIS start (2002)
    with pytest.raises(ValueError, match="Start date .* is before dataset start"):
        modis_manager.getDataset(
            coweeta_geometry, coweeta_crs,
            start='2000-01-01', end='2002-08-01',
            variables=['LAI']
        )

    # Test end date after MODIS end (2024)
    with pytest.raises(ValueError, match="End date .* is after dataset end"):
        modis_manager.getDataset(
            coweeta,
            start='2020-01-01', end='2025-01-01',
            variables=['LAI']
        )


def test_default_variables(modis_manager, coweeta, coweeta_date_range, tmp_path):
    """Test that default variables work when none specified."""
    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    start, end = coweeta_date_range

    # Create dummy nc files in tmp_path so isfile() succeeds
    lai_file = str(tmp_path / 'modis_LAI_2010-2011.nc')
    lulc_file = str(tmp_path / 'modis_LULC_2010-2011.nc')

    # Build minimal valid xarray datasets and save them
    import numpy as np
    ds_lai = xr.Dataset({'Lai_500m': xr.DataArray(
        np.zeros((1, 2, 2)), dims=['time', 'lat', 'lon'])})
    ds_lai.to_netcdf(lai_file)
    ds_lulc = xr.Dataset({'LC_Type1': xr.DataArray(
        np.zeros((1, 2, 2), dtype='int16'), dims=['time', 'lat', 'lon'])})
    ds_lulc.to_netcdf(lulc_file)

    # Map var name → temp file; _cacheFilename is called per var
    def mock_cache_filename(bounds, var=None, start_year=None, end_year=None):
        return {'LAI': lai_file, 'LULC': lulc_file}.get(var, '/nonexistent.nc')

    with patch.object(modis_manager, '_cacheFilename', side_effect=mock_cache_filename), \
         patch.object(modis_manager, '_checkCache', return_value=None):
        # Don't specify variables - should use defaults
        request = modis_manager.requestDataset(
            coweeta_geometry, coweeta_crs,
            start=start, end=end
        )

        assert request.is_ready
        assert set(request.variables) == {'LAI', 'LULC'}


@patch('watershed_workflow.sources.manager_modis_appeears.requests.post')
def test_construct_request_mocked(mock_post, modis_manager):
    """Test _constructRequest with mocked API call."""
    mock_response = MagicMock()
    mock_response.json.return_value = {'task_id': 'test_task_123'}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    modis_manager.login_token = 'fake_token'

    bounds = (-83.5, 35.0, -83.4, 35.1)
    task_id = modis_manager._constructRequest(bounds, 2010, 2011, ['LAI'])

    assert task_id == 'test_task_123'
    assert mock_post.called

    call_args = mock_post.call_args
    assert call_args[1]['json']['task_type'] == 'area'
    assert call_args[1]['json']['params']['layers'][0]['product'] == 'MCD15A3H.061'


def test_missing_files_error(modis_manager):
    """Test behavior when expected files don't exist."""
    r = ManagerDataset.Request(manager=modis_manager,
                               is_ready=True,
                               geometry=shapely.geometry.box(-83.5, 35.0, -83.4, 35.1),
                               start=cftime.datetime(2010, 8, 1, calendar='standard'),
                               end=cftime.datetime(2011, 8, 1, calendar='standard'),
                               variables=['LAI'],
                               )

    # Use the new cache_filenames kwarg (not filenames)
    request = modis_manager.Request(r,
        task_id="",
        cache_filenames={'LAI': '/nonexistent/file.nc'},
        urls={}
    )

    with pytest.raises(Exception):  # Should raise some kind of file not found error
        modis_manager._fetchDataset(request)
