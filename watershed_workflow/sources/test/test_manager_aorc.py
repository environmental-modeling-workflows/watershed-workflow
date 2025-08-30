import pytest
import numpy as np
import xarray as xr
import shapely.geometry
import cftime
import datetime

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs
import watershed_workflow.warp
from watershed_workflow.crs import CRS

from watershed_workflow.sources.manager_aorc import ManagerAORC


@pytest.fixture
def small_aorc_geometry():
    """Small test geometry in AORC native CRS (WGS84) for fast testing."""
    # Small box in North Carolina in WGS84 coordinates (lat/lon)
    return shapely.geometry.box(-83.0, 35.0, -82.9, 35.1)


@pytest.fixture  
def aorc_crs():
    """CRS for AORC data (WGS84)."""
    return CRS.from_epsg(4326)


@pytest.fixture 
def aorc_manager():
    """Standard AORC manager for testing."""
    return ManagerAORC()


@pytest.fixture
def short_time_range():
    """Short time range for fast testing."""
    start = cftime.datetime(2020, 6, 1, calendar='standard')
    end = cftime.datetime(2020, 6, 3, calendar='standard')  # 3 days
    return start, end


def test_getDataset_single_meteorological_variable(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test getDataset with single meteorological variable."""
    start, end = short_time_range
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['APCP_surface'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 1
    assert 'APCP_surface' in result.data_vars
    assert 'time' in result.dims
    assert result.rio.crs == aorc_manager.native_crs_out
    assert result.attrs['name'] == 'AORC v1.1'
    assert result.attrs['source'] == 'NOAA AWS S3 Zarr'
    
    # Base class clips to requested time range
    assert result.time.min().values <= start
    assert result.time.max().values <= end
    assert 'DatetimeGregorian' in str(type(result.time.values[0]))


def test_getDataset_multiple_meteorological_variables(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test getDataset with multiple meteorological variables."""
    start, end = short_time_range
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, 
                                     variables=['APCP_surface', 'TMP_2maboveground', 'PRES_surface'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 3
    assert 'APCP_surface' in result.data_vars
    assert 'TMP_2maboveground' in result.data_vars  
    assert 'PRES_surface' in result.data_vars
    assert 'time' in result.dims
    assert result.attrs['name'] == 'AORC v1.1'
    assert result.attrs['source'] == 'NOAA AWS S3 Zarr'


def test_getDataset_all_variables(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test getDataset with all AORC variables."""
    start, end = short_time_range
    
    all_vars = ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface', 
                'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=all_vars)
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 8
    for var in all_vars:
        assert var in result.data_vars


def test_getDataset_multi_year_request(small_aorc_geometry, aorc_crs, aorc_manager):
    """Test getDataset spanning multiple years."""
    start = cftime.datetime(2019, 12, 30, calendar='standard')
    end = cftime.datetime(2020, 1, 2, calendar='standard')  # Spans 2019-2020
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['APCP_surface'])
    
    assert isinstance(result, xr.Dataset)
    assert 'APCP_surface' in result.data_vars
    
    # Base class clips to requested time range
    assert result.time.min().values <= start
    assert result.time.max().values <= end
    
    # Check years are present (may include more than just the boundary years)
    years = set(result.time.dt.year.values)
    assert 2019 in years or 2020 in years  # At least one should be present


def test_standard_calendar_handling(small_aorc_geometry, aorc_crs, aorc_manager):
    """Test that AORC properly handles standard calendar."""
    start = cftime.datetime(2020, 2, 28, calendar='standard')
    end = cftime.datetime(2020, 3, 1, calendar='standard')  # Should include Feb 29 in leap year
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['APCP_surface'])
    
    # Check that Feb 29 exists in 2020 (leap year)
    assert isinstance(result, xr.Dataset)
    assert 'time' in result.dims
    
    # Check that Feb 29 is present in the dataset (2020 is a leap year)
    feb29_times = [t for t in result.time.values if (t.month == 2 and t.day == 29)]
    assert len(feb29_times) > 0, "Feb 29, 2020 should be present in leap year data"


def test_leap_year_handling(small_aorc_geometry, aorc_crs, aorc_manager):
    """Test specific leap year handling for Feb 29."""
    # Test leap year 2020
    start = cftime.datetime(2020, 2, 28, calendar='standard')
    end = cftime.datetime(2020, 3, 1, calendar='standard')  # Include Feb 29
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['TMP_2maboveground'])
    
    assert isinstance(result, xr.Dataset)
    assert 'TMP_2maboveground' in result.data_vars
    
    # Verify Feb 29 exists in the dataset
    feb29_times = [t for t in result.time.values if (t.month == 2 and t.day == 29)]
    assert len(feb29_times) > 0, "Should have Feb 29, 2020 data in leap year"


def test_native_crs_properties(aorc_manager):
    """Test AORC WGS84 CRS properties."""
    # Test CRS is WGS84 (EPSG:4326)
    assert aorc_manager.native_crs_in == CRS.from_epsg(4326)
    assert aorc_manager.native_crs_out == CRS.from_epsg(4326)
    assert aorc_manager.native_resolution == 0.00833333  # ~1km in degrees


def test_coordinate_system(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test that coordinates are in degrees (lat/lon), not meters."""
    start, end = short_time_range
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['APCP_surface'])
    
    # Check that coordinates are in degrees (small values for lat/lon)
    assert abs(result.longitude.values[0]) < 180  # Longitude should be < 180 degrees
    assert abs(result.latitude.values[0]) < 90    # Latitude should be < 90 degrees
    
    # Check coordinate spacing is small (degrees, not meters)
    lon_spacing = abs(result.longitude[1] - result.longitude[0]).values
    lat_spacing = abs(result.latitude[1] - result.latitude[0]).values
    assert lon_spacing < 1.0  # Should be fraction of a degree
    assert lat_spacing < 1.0  # Should be fraction of a degree


def test_default_variables_property(aorc_manager):
    """Test that all AORC variables are both valid and default."""
    expected_vars = ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface',
                     'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    
    assert aorc_manager.default_variables == expected_vars
    assert aorc_manager.valid_variables == expected_vars
    assert len(aorc_manager.valid_variables) == 8  # All 8 AORC variables


def test_invalid_year_range_error(small_aorc_geometry, aorc_crs, aorc_manager):
    """Test error when start year > end year."""
    start = cftime.datetime(2021, 1, 1, calendar='standard')
    end = cftime.datetime(2020, 1, 1, calendar='standard')  # Invalid: start > end
    
    with pytest.raises(RuntimeError, match="start year .* is after .* end year"):
        aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['APCP_surface'])


def test_getDataset_coweeta_compatibility(coweeta):
    """Test AORC works with existing coweeta fixture."""
    aorc_manager = ManagerAORC()
    
    start = cftime.datetime(2020, 7, 1, calendar='standard')
    end = cftime.datetime(2020, 7, 2, calendar='standard')  # 2 days
    
    # Use coweeta GeoDataFrame directly (CRS will be detected automatically)
    result = aorc_manager.getDataset(coweeta, start=start, end=end, variables=['APCP_surface', 'TMP_2maboveground'])
    
    assert isinstance(result, xr.Dataset)
    assert 'APCP_surface' in result.data_vars
    assert 'TMP_2maboveground' in result.data_vars
    assert result.attrs['name'] == 'AORC v1.1'
    assert result.attrs['source'] == 'NOAA AWS S3 Zarr'
    
    # Check it covers requested range
    assert result.time.min().values <= start
    assert result.time.max().values <= end
    
    # Check reasonable values for July in North Carolina
    temp_mean = result['TMP_2maboveground'].values.mean()
    assert 280 < temp_mean < 310  # Reasonable July temperature in Kelvin (~7-37°C)


def test_variable_filtering(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test that dataset returns only requested variables."""
    start, end = short_time_range
    
    # Request only 2 variables out of 8
    requested_vars = ['APCP_surface', 'TMP_2maboveground']
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=requested_vars)
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 2
    assert set(result.data_vars.keys()) == set(requested_vars)
    
    # Should not contain other AORC variables
    other_vars = ['DLWRF_surface', 'DSWRF_surface', 'PRES_surface', 
                  'SPFH_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    for var in other_vars:
        assert var not in result.data_vars


def test_hourly_temporal_resolution(small_aorc_geometry, aorc_crs, aorc_manager):
    """Test that data has hourly timestamps."""
    start = cftime.datetime(2020, 6, 1, 0, calendar='standard')  # Start at midnight
    end = cftime.datetime(2020, 6, 1, 23, calendar='standard')   # End at 11 PM (same day)
    
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, variables=['APCP_surface'])
    
    # Check hourly resolution
    assert isinstance(result, xr.Dataset)
    assert 'time' in result.dims
    
    # Check that timestamps are hourly by examining consecutive differences
    time_diffs = np.diff(result.time.values[:min(100, len(result.time))])  # Check available timestamps
    expected_hour = np.timedelta64(1, 'h')
    
    # Most time differences should be 1 hour (allowing for some variation)
    hourly_diffs = [diff for diff in time_diffs if diff == expected_hour]
    assert len(hourly_diffs) > len(time_diffs) * 0.8, "Most time differences should be 1 hour"
    
    # Check that June 1 data exists if in range
    june1_times = [t for t in result.time.values if (t.month == 6 and t.day == 1)]
    if len(june1_times) > 0:  # If June 1 is in the dataset
        assert len(june1_times) <= 24, "Should have at most 24 hours for June 1"


def test_requestDataset_pattern(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test the non-blocking request/fetch pattern."""
    start, end = short_time_range
    
    # Request the dataset
    request = aorc_manager.requestDataset(small_aorc_geometry, aorc_crs, 
                                         start=start, end=end, variables=['APCP_surface'])
    
    # Should be ready immediately after download completes
    assert aorc_manager.isReady(request)
    
    # Fetch the data
    result = aorc_manager.fetchRequest(request)
    
    assert isinstance(result, xr.Dataset)
    assert 'APCP_surface' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(4326)
    assert result.attrs['name'] == 'AORC v1.1'
    assert result.attrs['source'] == 'NOAA AWS S3 Zarr'


def test_waitForDataset_pattern(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test the request/wait pattern."""
    start, end = short_time_range
    
    # Request the dataset
    request = aorc_manager.requestDataset(small_aorc_geometry, aorc_crs, 
                                         start=start, end=end, variables=['TMP_2maboveground'])
    
    # Wait for completion (should be immediate since data gets downloaded during request)
    result = aorc_manager.waitForDataset(request, interval=1, tries=3)
    
    assert isinstance(result, xr.Dataset)
    assert 'TMP_2maboveground' in result.data_vars
    assert result.rio.crs == CRS.from_epsg(4326)
    assert result.attrs['name'] == 'AORC v1.1'
    assert result.attrs['source'] == 'NOAA AWS S3 Zarr'


def test_default_variables(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test that default variables work when none specified."""
    start, end = short_time_range
    
    # Don't specify variables - should use defaults (all 8 variables)
    result = aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end)
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 8  # All AORC variables by default
    expected_vars = ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface',
                     'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    for var in expected_vars:
        assert var in result.data_vars


def test_invalid_variable_error(small_aorc_geometry, aorc_crs, aorc_manager, short_time_range):
    """Test that invalid variables raise appropriate errors."""
    start, end = short_time_range
    
    with pytest.raises(ValueError, match="Invalid variable"):
        aorc_manager.getDataset(small_aorc_geometry, aorc_crs, start=start, end=end, 
                               variables=['InvalidVariable'])


def test_date_validation_errors(small_aorc_geometry, aorc_crs, aorc_manager):
    """Test date validation against AORC bounds."""
    # Test start date before AORC start (2007)
    with pytest.raises(ValueError, match="Start date .* is before dataset start"):
        aorc_manager.getDataset(small_aorc_geometry, aorc_crs, 
                               start='2005-01-01', end='2007-01-02', variables=['APCP_surface'])
    
    # Test end date after AORC end (2024)
    with pytest.raises(ValueError, match="End date .* is after dataset end"):
        aorc_manager.getDataset(small_aorc_geometry, aorc_crs, 
                               start='2024-01-01', end='2025-01-01', variables=['APCP_surface'])
