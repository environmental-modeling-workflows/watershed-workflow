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
    from watershed_workflow.sources.manager_modis_earthdata import _MODIS_SINU_CRS
    assert modis_manager.product == 'MODIS'
    assert modis_manager.source == 'NASA AppEEARS'
    assert modis_manager.native_crs_in == CRS.from_epsg(4326)
    assert watershed_workflow.crs.isEqual(modis_manager.native_crs_out, _MODIS_SINU_CRS)
    assert set(modis_manager.valid_variables) == {'LAI', 'LULC'}
    assert set(modis_manager.default_variables) == {'LAI', 'LULC'}


def test_cache_dir_generation(modis_manager, tmp_path, monkeypatch):
    """Test cache directory name generation via cacheDirname free function."""
    import watershed_workflow.utils.config
    from watershed_workflow.sources.cache_info import cacheDirname
    monkeypatch.setitem(watershed_workflow.utils.config.rcParams['DEFAULT'],
                        'data_directory', str(tmp_path))
    bounds = (-83.5, 35.0, -83.4, 35.1)
    dirpath = cacheDirname(modis_manager.attrs, bounds, start_year=2010, end_year=2011)
    dirname = os.path.basename(dirpath)
    assert '2010-2011' in dirname
    assert 'nasa_appeears' in dirname


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


def test_read_dataset_multiple_variables(modis_manager, existing_lai_file, existing_lulc_file, tmp_path):
    """Test reading dataset with multiple variables from existing files via _loadDataset."""
    if not os.path.exists(existing_lai_file) or not os.path.exists(existing_lulc_file):
        pytest.skip("Test files not available")

    # Set up a fake cache directory with the files
    import shutil
    os.makedirs(tmp_path / 'cache_dir', exist_ok=True)
    shutil.copy(existing_lai_file, tmp_path / 'cache_dir' / 'LAI.nc')
    shutil.copy(existing_lulc_file, tmp_path / 'cache_dir' / 'LULC.nc')

    request = MagicMock()
    request.variables = ['LAI', 'LULC']
    request._download_path = str(tmp_path / 'cache_dir')
    request._cache_hit = True

    dataset = modis_manager._loadDataset(request)

    assert isinstance(dataset, xr.Dataset)
    assert 'LAI' in dataset.data_vars
    assert 'LULC' in dataset.data_vars
    assert len(dataset.data_vars) == 2


def test_getDataset_with_existing_files(modis_manager, coweeta, coweeta_date_range, tmp_path,
                                         monkeypatch, existing_lai_file, existing_lulc_file):
    """Test blocking getDataset with existing files via isComplete mock."""
    if not os.path.exists(existing_lai_file) or not os.path.exists(existing_lulc_file):
        pytest.skip("Test files not available")

    import shutil, watershed_workflow.utils.config
    monkeypatch.setitem(watershed_workflow.utils.config.rcParams['DEFAULT'],
                        'data_directory', str(tmp_path))

    # Build cache dir for the request and populate it
    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    start, end = coweeta_date_range

    cache_dir = modis_manager._cache_info.cacheDirname(
        coweeta_geometry.buffer(3 * modis_manager.native_resolution).bounds,
        start_year=start.year, end_year=end.year)
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy(existing_lai_file, os.path.join(cache_dir, 'LAI.nc'))
    shutil.copy(existing_lulc_file, os.path.join(cache_dir, 'LULC.nc'))

    dataset = modis_manager.getDataset(coweeta_geometry, coweeta_crs, start, end, ['LAI', 'LULC'])

    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs['name'] == 'MODIS'
    assert dataset.attrs['source'] == 'AppEEARS'


def test_request_fetch_pattern(modis_manager, coweeta, coweeta_date_range, tmp_path,
                                monkeypatch, existing_lai_file):
    """Test non-blocking request/fetch pattern using isComplete mock."""
    if not os.path.exists(existing_lai_file):
        pytest.skip("Test file not available")

    import shutil, watershed_workflow.utils.config
    monkeypatch.setitem(watershed_workflow.utils.config.rcParams['DEFAULT'],
                        'data_directory', str(tmp_path))

    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    start, end = coweeta_date_range

    cache_dir = modis_manager._cache_info.cacheDirname(
        coweeta_geometry.buffer(3 * modis_manager.native_resolution).bounds,
        start_year=start.year, end_year=end.year)
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy(existing_lai_file, os.path.join(cache_dir, 'LAI.nc'))

    request = modis_manager.requestDataset(
        coweeta_geometry, coweeta_crs, start=start, end=end, variables=['LAI']
    )
    assert modis_manager.isReady(request)

    dataset = modis_manager.fetchDataset(request)
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


def test_default_variables(modis_manager, coweeta, coweeta_date_range):
    """Test that default variables are used when none specified."""
    coweeta_crs = coweeta.crs
    coweeta_geometry = coweeta.geometry[0]
    start, end = coweeta_date_range

    request = modis_manager._preprocessParameters(
        coweeta_geometry, coweeta_crs, start, end, None, None, None
    )

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


def test_missing_files_error(modis_manager, tmp_path):
    """Test behavior when expected files don't exist in cache directory."""
    request = MagicMock()
    request.variables = ['LAI']
    request._download_path = str(tmp_path / 'empty_dir')
    os.makedirs(request._download_path, exist_ok=True)
    request._cache_hit = True

    with pytest.raises(Exception):  # Should raise some kind of file not found error
        modis_manager._loadDataset(request)
