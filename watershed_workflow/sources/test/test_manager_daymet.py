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

from watershed_workflow.sources.manager_daymet import ManagerDaymet


@pytest.fixture
def small_daymet_geometry():
    """Small test geometry in WGS84 for fast testing."""
    # Small box in Colorado (WGS84 coordinates)
    return shapely.geometry.box(-107.0, 39.0, -106.9, 39.1)


@pytest.fixture 
def daymet_manager():
    """Standard DayMet manager for testing."""
    return ManagerDaymet()


@pytest.fixture
def short_time_range():
    """Short time range for fast testing."""
    start = cftime.datetime(2020, 6, 1, calendar='noleap')
    end = cftime.datetime(2020, 6, 7, calendar='noleap')  # One week
    return start, end


def test_getDataset_single_meteorological_variable(small_daymet_geometry, daymet_manager, short_time_range):
    """Test getDataset with single meteorological variable."""
    start, end = short_time_range
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    result = daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['tmin'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 1
    assert 'tmin' in result.data_vars
    assert 'time' in result.dims
    assert result.rio.crs == daymet_manager.native_crs_out
    
    # Check temporal properties
    assert len(result.time) == 7  # One week of daily data
    assert result.time.values[0].calendar == 'noleap'
    
    # Check metadata added by base class
    assert 'name' in result.attrs
    assert 'source' in result.attrs


def test_getDataset_multiple_meteorological_variables(small_daymet_geometry, daymet_manager, short_time_range):
    """Test getDataset with multiple meteorological variables."""
    start, end = short_time_range
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    result = daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end,
                                       ['tmin', 'tmax', 'prcp'])
    
    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) == 3
    assert 'tmin' in result.data_vars
    assert 'tmax' in result.data_vars  
    assert 'prcp' in result.data_vars
    assert 'time' in result.dims


def test_getDataset_multi_year_request(small_daymet_geometry, daymet_manager):
    """Test getDataset spanning multiple years."""
    start = cftime.datetime(2019, 12, 30, calendar='noleap')
    end = cftime.datetime(2020, 1, 3, calendar='noleap')  # Spans 2019-2020
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    result = daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['tmin'])
    
    assert isinstance(result, xr.Dataset)
    assert 'tmin' in result.data_vars
    
    # Should have 5 days total (Dec 30-31, Jan 1-3)
    assert len(result.time) == 5
    
    # Check years are present
    years = set(result.time.dt.year.values)
    assert years == {2019, 2020}


def test_noleap_calendar_handling(small_daymet_geometry, daymet_manager):
    """Test that DayMet properly handles noleap calendar."""
    start = cftime.datetime(2020, 2, 28, calendar='noleap')
    end = cftime.datetime(2020, 3, 1, calendar='noleap')  # No Feb 29 in noleap
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    result = daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['tmin'])
    
    # Should have Feb 28 and Mar 1, no Feb 29
    assert len(result.time) == 2
    dates = [t.strftime('%m-%d') for t in result.time.values]
    assert '02-28' in dates
    assert '03-01' in dates
    assert '02-29' not in dates


def test_native_crs_properties(daymet_manager):
    """Test DayMet CRS properties."""
    # Test that output CRS is Lambert Conformal Conic with DayMet parameters
    proj4_str = daymet_manager.native_crs_out.to_proj4()
    assert '+proj=lcc' in proj4_str
    assert '+lat_1=25' in proj4_str
    assert '+lat_2=60' in proj4_str
    assert '+lat_0=42.5' in proj4_str
    assert '+lon_0=-100' in proj4_str
    assert '+units=m' in proj4_str
    
    assert daymet_manager.native_crs_in == CRS.from_epsg(4326)  # WGS84 input
    assert daymet_manager.native_resolution == 1000.0  # 1km


def test_coordinate_conversion_from_km_to_meters(small_daymet_geometry, daymet_manager, short_time_range):
    """Test that coordinates are properly converted from km to meters."""
    start, end = short_time_range
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    result = daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['tmin'])
    
    # Check that coordinates are in meters (large values)
    assert abs(result.x.values[0]) > 1000000  # Should be > 1000km in meters
    assert abs(result.y.values[0]) > 100000   # Should be > 100km in meters
    
    # Check coordinate spacing is ~1000m (1km resolution)
    x_spacing = abs(result.x[1] - result.x[0]).values
    assert abs(x_spacing - 1000.0) < 50.0  # Within 50m tolerance


def test_default_variables_property(daymet_manager):
    """Test that default variables excludes 'swe' (snow water equivalent)."""
    expected_default = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']
    expected_valid = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    
    assert daymet_manager.default_variables == expected_default
    assert daymet_manager.valid_variables == expected_valid
    assert 'swe' not in daymet_manager.default_variables  # Snow not in default


def test_invalid_year_range_error(small_daymet_geometry, daymet_manager):
    """Test error when start year > end year."""
    start = cftime.datetime(2021, 1, 1, calendar='noleap')
    end = cftime.datetime(2020, 1, 1, calendar='noleap')  # Invalid: start > end
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    with pytest.raises(RuntimeError, match="Start year .* is after end year"):
        daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['tmin'])


def test_getDataset_coweeta_integration(coweeta):
    """Test DayMet works with existing coweeta fixture."""
    mgr = ManagerDaymet()
    
    # Use first geometry from coweeta with proper CRS
    geom = coweeta.geometry.iloc[0]
    geom_crs = coweeta.crs
    
    start = cftime.datetime(2020, 7, 1, calendar='noleap')
    end = cftime.datetime(2020, 7, 3, calendar='noleap')  # 3 days
    
    result = mgr.getDataset(geom, geom_crs, start, end, ['tmin', 'tmax'])
    
    assert isinstance(result, xr.Dataset)
    assert 'tmin' in result.data_vars
    assert 'tmax' in result.data_vars
    assert len(result.time) == 3
    
    # Check reasonable temperature values for July in North Carolina
    tmin_mean = result['tmin'].values.mean()
    tmax_mean = result['tmax'].values.mean()
    assert 15 < tmin_mean < 30  # Reasonable July tmin (°C)
    assert 25 < tmax_mean < 40  # Reasonable July tmax (°C)


def test_daymet_async_interface(small_daymet_geometry, daymet_manager, short_time_range):
    """Test the async interface methods."""
    start, end = short_time_range
    geometry_crs = watershed_workflow.crs.latlon_crs
    variables = ['tmin']
    
    # Test requestDataset
    request = daymet_manager.requestDataset(small_daymet_geometry, geometry_crs, start, end, variables)
    assert isinstance(request, daymet_manager.Request)
    assert request.manager == daymet_manager
    assert request.geometry is not None
    assert request.variables == variables
    # Check DayMet-specific attributes
    assert hasattr(request, 'bounds')
    assert hasattr(request, 'start_year')
    assert hasattr(request, 'end_year')
    assert hasattr(request, 'filenames')
    assert request.start_year == 2020
    assert request.end_year == 2020
    
    # Test isReady - might be ready if files exist, or not if they need downloading
    ready_status = daymet_manager.isReady(request)
    assert isinstance(ready_status, bool)
    
    # Test fetchRequest
    result = daymet_manager.fetchRequest(request)
    assert isinstance(result, xr.Dataset)
    assert 'tmin' in result.data_vars
    assert result.rio.crs == daymet_manager.native_crs_out


def test_daymet_default_variables(small_daymet_geometry, daymet_manager, short_time_range):
    """Test that default variables work correctly."""
    start, end = short_time_range
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test with no variables specified (should use defaults)
    result = daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end)
    
    assert isinstance(result, xr.Dataset)
    # Should have default variables (excludes 'swe')
    expected_defaults = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']
    assert len(result.data_vars) == len(expected_defaults)
    for var in expected_defaults:
        assert var in result.data_vars
    assert 'swe' not in result.data_vars


def test_daymet_invalid_variable_error(small_daymet_geometry, daymet_manager, short_time_range):
    """Test that invalid variables raise appropriate errors."""
    start, end = short_time_range
    geometry_crs = watershed_workflow.crs.latlon_crs
    
    # Test invalid variable
    with pytest.raises(ValueError, match="Invalid variable"):
        daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['invalid_var'])
    
    # Test mix of valid and invalid variables  
    with pytest.raises(ValueError, match="Invalid variable"):
        daymet_manager.getDataset(small_daymet_geometry, geometry_crs, start, end, ['tmin', 'invalid_var'])
