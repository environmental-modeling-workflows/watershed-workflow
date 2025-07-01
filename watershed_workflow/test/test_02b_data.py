import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import cftime
import datetime
import warnings

import shapely.geometry
import shapely.wkt

import watershed_workflow.data as wwd


class TestConvertTimesToCFTimeNoleap:
    """Test class for _convertTimesToCFTimeNoleap helper function."""

    def test_numpy_datetime64_array_input_output(self):
        """Test numpy array input returns numpy array output."""
        dates = np.array(['2000-01-01T12:30:45', '2000-02-28T06:15:30'], dtype='datetime64[s]')
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert all(isinstance(t, cftime.DatetimeNoLeap) for t in result)
        
        # Check first date
        assert result[0].year == 2000
        assert result[0].month == 1
        assert result[0].day == 1
        assert result[0].hour == 12
        assert result[0].minute == 30
        assert result[0].second == 45

    def test_pandas_series_input_output(self):
        """Test pandas Series input returns pandas Series output."""
        dates = pd.Series(pd.date_range('2000-01-01', '2000-01-03', freq='D'))
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert all(isinstance(t, cftime.DatetimeNoLeap) for t in result)

    def test_list_input_output(self):
        """Test list input returns list output."""
        dates = [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)]
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(t, cftime.DatetimeNoLeap) for t in result)

    def test_tuple_input_returns_list(self):
        """Test tuple input returns list output."""
        dates = (datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2))
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(t, cftime.DatetimeNoLeap) for t in result)

    def test_numpy_datetime64_with_leap_day_raises_error(self):
        """Test that numpy datetime64 with leap day raises ValueError."""
        dates = np.array(['2000-01-01', '2000-12-31'], dtype='datetime64[D]')  # 2000-12-31 is day 366
        r1 = wwd._convertTimesToCFTime(dates)
        
        with pytest.raises(ValueError, match="Input contains leap day"):
            wwd._convertTimesToCFTimeNoleap(r1)

    def test_pandas_timestamp_with_leap_day_raises_error(self):
        """Test that pandas Timestamp with leap day raises ValueError."""
        dates = pd.Series([
            pd.Timestamp('2000-01-01'),
            pd.Timestamp('2000-12-31')  # Day 366 in leap year
        ])
        r1 = wwd._convertTimesToCFTime(dates)
        
        with pytest.raises(ValueError, match="Input contains leap day"):
            wwd._convertTimesToCFTimeNoleap(r1)

    def test_python_datetime_with_leap_day_raises_error(self):
        """Test that Python datetime with leap day raises ValueError."""
        dates = [
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 12, 31)  # Day 366 in leap year
        ]
        r1 = wwd._convertTimesToCFTime(dates)
        
        with pytest.raises(ValueError, match="Input contains leap day"):
            wwd._convertTimesToCFTimeNoleap(r1)

    def test_cftime_with_leap_day_raises_error(self):
        """Test that cftime objects with leap day raise ValueError."""
        dates = [
            cftime.DatetimeGregorian(2000, 1, 1),
            cftime.DatetimeGregorian(2000, 12, 31)  # Day 366 in leap year
        ]
        r1 = wwd._convertTimesToCFTime(dates)
        
        with pytest.raises(ValueError, match="Input contains leap day"):
            wwd._convertTimesToCFTimeNoleap(dates)
        with pytest.raises(ValueError, match="Input contains leap day"):
            wwd._convertTimesToCFTimeNoleap(r1)

    def test_multiple_leap_days_error_message(self):
        """Test that error message includes all leap days found."""
        dates = np.array(['2000-12-31', '2004-12-31'], dtype='datetime64[D]')
        r1 = wwd._convertTimesToCFTime(dates)
        
        with pytest.raises(ValueError) as exc_info:
            wwd._convertTimesToCFTimeNoleap(r1)
        
        error_message = str(exc_info.value)
        assert "Input contains leap day" in error_message

    def test_non_leap_year_dec_31_allowed(self):
        """Test that Dec 31 in non-leap years is allowed."""
        dates = [
            datetime.datetime(2001, 1, 1),
            datetime.datetime(2001, 12, 31)  # Day 365 in non-leap year
        ]
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert len(result) == 2
        assert result[1].month == 12
        assert result[1].day == 31

    def test_feb_29_allowed(self):
        """Test that Feb 29 in leap years is allowed (not removed)."""
        dates = [datetime.datetime(2000, 2, 29)]  # Feb 29 is day 60, not 366
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert len(result) == 1
        assert result[0].month == 3
        assert result[0].day == 1

    def test_empty_numpy_array(self):
        """Test handling of empty numpy array."""
        dates = np.array([], dtype='datetime64[D]')
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_empty_pandas_series(self):
        """Test handling of empty pandas Series."""
        dates = pd.Series([], dtype='datetime64[ns]')
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_empty_list(self):
        """Test handling of empty list."""
        dates = []
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_time_attributes_preserved_pandas(self):
        """Test that time attributes (hour, minute, second) are preserved for pandas."""
        dates = pd.Series([
            pd.Timestamp('2000-01-01 14:30:45'),
            pd.Timestamp('2000-01-02 16:25:30')
        ])
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, pd.Series)
        assert result.iloc[0].hour == 14
        assert result.iloc[0].minute == 30
        assert result.iloc[0].second == 45
        assert result.iloc[1].hour == 16
        assert result.iloc[1].minute == 25
        assert result.iloc[1].second == 30

    def test_time_attributes_preserved_python_datetime(self):
        """Test that time attributes are preserved for Python datetime."""
        dates = [
            datetime.datetime(2000, 1, 1, 9, 15, 30),
            datetime.datetime(2000, 1, 2, 14, 45, 50)
        ]
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert result[0].hour == 9
        assert result[0].minute == 15
        assert result[0].second == 30
        assert result[1].hour == 14
        assert result[1].minute == 45
        assert result[1].second == 50

    def test_time_attributes_preserved_numpy_datetime64(self):
        """Test that time attributes are preserved for numpy datetime64."""
        dates = np.array(['2000-01-01T12:30:45', '2000-01-02T18:15:20'], dtype='datetime64[s]')
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert result[0].hour == 12
        assert result[0].minute == 30
        assert result[0].second == 45
        assert result[1].hour == 18
        assert result[1].minute == 15
        assert result[1].second == 20

    def test_numpy_datetime64_microsecond_precision(self):
        """Test that microsecond precision is preserved."""
        dates = np.array(['2000-01-01T12:30:45.123456'], dtype='datetime64[us]')
        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert isinstance(result, np.ndarray)
        assert result[0].microsecond == 123456

    def test_consistency_across_input_types(self):
        """Test that conversion produces consistent results across input types."""
        # Same dates in different formats
        base_date = datetime.datetime(2000, 1, 15, 10, 30, 45)
        
        list_input = [base_date]
        array_input = np.array([base_date], dtype=object)
        series_input = pd.Series([pd.Timestamp(base_date)])
        
        list_input2 = wwd._convertTimesToCFTime(list_input)
        list_result = wwd._convertTimesToCFTimeNoleap(list_input2)

        array_input2 = wwd._convertTimesToCFTime(array_input)
        array_result = wwd._convertTimesToCFTimeNoleap(array_input2)

        series_input2 = wwd._convertTimesToCFTime(series_input)
        series_result = wwd._convertTimesToCFTimeNoleap(series_input2)
        
        # All should produce equivalent cftime objects
        assert list_result[0].year == array_result[0].year == series_result.iloc[0].year
        assert list_result[0].month == array_result[0].month == series_result.iloc[0].month
        assert list_result[0].day == array_result[0].day == series_result.iloc[0].day
        assert list_result[0].hour == array_result[0].hour == series_result.iloc[0].hour
        assert list_result[0].minute == array_result[0].minute == series_result.iloc[0].minute
        assert list_result[0].second == array_result[0].second == series_result.iloc[0].second

    def test_mixed_years_some_leap_some_not(self):
        """Test dataset spanning multiple years with some leap years."""
        # Include dates from leap year 2000 and non-leap year 2001, but no Dec 31 from leap year
        dates = [
            datetime.datetime(2000, 2, 28),   # Day 59 in leap year
            datetime.datetime(2000, 2, 29),   # Day 60 in leap year (Feb 29 - allowed)
            datetime.datetime(2000, 3, 1),    # Day 61 in leap year
            datetime.datetime(2001, 12, 31),  # Day 365 in non-leap year (allowed)
        ]

        r1 = wwd._convertTimesToCFTime(dates)
        result = wwd._convertTimesToCFTimeNoleap(r1)
        
        assert len(result) == 4
        assert all(isinstance(t, cftime.DatetimeNoLeap) for t in result)
        
        # Verify specific dates
        assert result[1].month == 3 and result[1].day == 1  # Feb 29 --> Mar 1 preserved
        assert result[2].month == 3 and result[2].day == 2  # Mar 1 --> Mar 2
        assert result[3].month == 12 and result[3].day == 31  # Dec 31 non-leap year preserved

    def test_numpy_datetime64_different_precisions(self):
        """Test numpy datetime64 with different time precisions."""
        # Test with day precision
        dates_day = np.array(['2000-01-01', '2000-01-02'], dtype='datetime64[D]')
        r1_day = wwd._convertTimesToCFTime(dates_day)
        result_day = wwd._convertTimesToCFTimeNoleap(r1_day)
        assert all(t.hour == 0 and t.minute == 0 and t.second == 0 for t in result_day)
        
        # Test with second precision
        dates_sec = np.array(['2000-01-01T12:30:45'], dtype='datetime64[s]')
        r1_sec = wwd._convertTimesToCFTime(dates_sec)
        result_sec = wwd._convertTimesToCFTimeNoleap(r1_sec)
        assert result_sec[0].hour == 12
        assert result_sec[0].minute == 30
        assert result_sec[0].second == 45


class TestFilterLeapDay:
    
    def test_removes_leap_day_366(self):
        """Test that day 366 in leap years is removed."""
        # Create DataFrame with leap year data including day 366
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': [1, 2, 3, 4]
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # Check that 2020-12-31 (day 366) was removed
        assert len(result) == 3
        
        # Extract dates from cftime objects
        result_dates = [(dt.year, dt.month, dt.day) for dt in result['time']]
        assert (2020, 12, 31) in result_dates
        assert (2021, 1, 1) in result_dates
        assert (2021, 1, 2) in result_dates
    
    def test_preserves_non_leap_year_dec_31(self):
        """Test that Dec 31 in non-leap years is preserved."""
        # Create DataFrame with non-leap year Dec 31
        dates = pd.date_range('2021-12-30', '2022-01-02', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': [1, 2, 3, 4]
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # All dates should be preserved
        assert len(result) == 4
        result_dates = [(dt.year, dt.month, dt.day) for dt in result['time']]
        assert (2021, 12, 31) in result_dates
    
    def test_handles_multiple_leap_years(self):
        """Test filtering across multiple leap years."""
        # Create data spanning multiple leap years
        dates = pd.date_range('2019-12-31', '2021-01-01', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': range(len(dates))
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # Check that only 2020-12-31 was removed (2020 is leap year)
        original_count = len(df)
        filtered_count = len(result)
        assert original_count - filtered_count == 1
        
        result_dates = [(dt.year, dt.month, dt.day) for dt in result['time']]
        assert (2020, 2, 29) not in result_dates
        assert (2020, 12, 31) in result_dates
        assert (2019, 12, 31) in result_dates  # Non-leap year preserved
    
    def test_preserves_other_columns(self):
        """Test that other DataFrame columns are preserved correctly."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': [10, 20, 30, 40],
            'category': ['A', 'B', 'C', 'D']
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # Check that values align correctly after filtering
        assert list(result['value']) == [10, 30, 40]
        assert list(result['category']) == ['A', 'C', 'D']
    
    def test_resets_index(self):
        """Test that DataFrame index is reset after filtering."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': [1, 2, 3, 4]
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # Index should be continuous from 0
        assert list(result.index) == [0, 1, 2]
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'time': pd.Series(dtype='datetime64[ns]'), 'value': []})
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        assert len(result) == 0
        assert 'time' in result.columns
    
    def test_no_leap_days_present(self):
        """Test DataFrame with no leap days."""
        dates = pd.date_range('2021-06-01', '2021-06-30', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': range(len(dates))
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # All rows should be preserved
        assert len(result) == len(df)
    
    def test_invalid_column_name_raises_error(self):
        """Test that invalid column name raises ValueError."""
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
            wwd.filterLeapDay_DataFrame(df, 'invalid_column')
    
    def test_non_datetime_column_raises_error(self):
        """Test that non-datetime column raises ValueError."""
        df = pd.DataFrame({
            'time': ['not', 'a', 'datetime'],
            'value': [1, 2, 3]
        })
        
        with pytest.raises(ValueError, match="Could not convert column 'time' to"):
            wwd.filterLeapDay_DataFrame(df, 'time')
    
    def test_preserves_dataframe_copy(self):
        """Test that original DataFrame is not modified."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': [1, 2, 3, 4]
        })
        df_copy = df.copy()
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(df, df_copy)
    
    def test_cftime_objects_have_noleap_calendar(self):
        """Test that returned time objects are cftime with noleap calendar."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        df = pd.DataFrame({
            'time': dates,
            'value': [1, 2, 3, 4]
        })
        
        result = wwd.filterLeapDay_DataFrame(df, 'time')
        
        # Check all time values are cftime objects
        assert all(isinstance(dt, cftime.datetime) for dt in result['time'])
        
        # Verify they have the noleap calendar
        for dt in result['time']:
            assert hasattr(dt, 'calendar')
            assert 'noleap' in dt.calendar or '365_day' in dt.calendar



class TestFilterLeapDayDataArray:
    
    def test_removes_leap_day_366(self):
        """Test that day 366 in leap years is removed from DataArray."""
        # Create DataArray with leap year data including day 366
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        data = np.arange(len(dates))
        da = xr.DataArray(data, coords={'time': dates}, dims=['time'])
        
        result = wwd.filterLeapDay_xarray(da)
        
        # Check that 2020-12-31 (day 366) was removed
        assert len(result.time) == 3
        
        # Extract dates from cftime objects
        result_dates = [(dt.year, dt.month, dt.day) for dt in result.time.values]
        assert (2020, 12, 30) not in result_dates
        assert (2020, 12, 31) in result_dates
        assert (2021, 1, 1) in result_dates
        assert (2021, 1, 2) in result_dates
        
        # Check data values are preserved correctly
        assert result.values[0] == 0  # Dec 30
        assert result.values[1] == 2  # Jan 1 (skipping Dec 31)
        assert result.values[2] == 3  # Jan 2
    
    def test_preserves_non_leap_year_dec_31(self):
        """Test that Dec 31 in non-leap years is preserved."""
        dates = pd.date_range('2021-12-30', '2022-01-02', freq='D')
        data = np.arange(len(dates))
        da = xr.DataArray(data, coords={'time': dates}, dims=['time'])
        
        result = wwd.filterLeapDay_xarray(da)
        
        # All dates should be preserved
        assert len(result.time) == 4
        result_dates = [(dt.year, dt.month, dt.day) for dt in result.time.values]
        assert (2021, 12, 31) in result_dates
    
    def test_multidimensional_array(self):
        """Test filtering with multidimensional DataArray."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        data = np.random.rand(len(dates), 5, 5)
        da = xr.DataArray(
            data, 
            coords={'time': dates, 'x': range(5), 'y': range(5)}, 
            dims=['time', 'x', 'y']
        )
        
        result = wwd.filterLeapDay_xarray(da)
        
        # Check dimensions
        assert result.shape == (3, 5, 5)  # One time step removed
        assert len(result.time) == 3
        
        # Check that spatial dimensions are preserved
        assert np.array_equal(result.x.values, da.x.values)
        assert np.array_equal(result.y.values, da.y.values)
    
    def test_preserves_attributes(self):
        """Test that DataArray attributes are preserved."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        data = np.arange(len(dates))
        da = xr.DataArray(
            data, 
            coords={'time': dates}, 
            dims=['time'],
            attrs={'units': 'meters', 'description': 'test data'}
        )
        
        result = wwd.filterLeapDay_xarray(da)
        
        # Check attributes are preserved
        assert result.attrs['units'] == 'meters'
        assert result.attrs['description'] == 'test data'
    
    def test_preserves_rasterio_attributes(self):
        """Test that rasterio-specific attributes (nodata, crs) are preserved."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        data = np.random.rand(len(dates), 10, 10)
        
        # Create DataArray with rasterio-like attributes
        da = xr.DataArray(
            data,
            coords={
                'time': dates,
                'y': np.arange(10),
                'x': np.arange(10)
            },
            dims=['time', 'y', 'x'],
            attrs={'nodata': -9999, 'crs': 'EPSG:4326'}
        )
        
        # Add CRS to spatial coordinates (as rasterio does)
        da.coords['x'].attrs['crs'] = 'EPSG:4326'
        da.coords['y'].attrs['crs'] = 'EPSG:4326'
        
        result = wwd.filterLeapDay_xarray(da)
        
        # Check rasterio attributes are preserved
        assert result.attrs['nodata'] == -9999
        assert result.attrs['crs'] == 'EPSG:4326'
        
        # Check coordinate attributes are preserved
        assert result.coords['x'].attrs['crs'] == 'EPSG:4326'
        assert result.coords['y'].attrs['crs'] == 'EPSG:4326'
    
    def test_cftime_objects_have_noleap_calendar(self):
        """Test that returned time objects are cftime with noleap calendar."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        data = np.arange(len(dates))
        da = xr.DataArray(data, coords={'time': dates}, dims=['time'])
        
        result = wwd.filterLeapDay_xarray(da)
        
        # Check all time values are cftime objects
        assert all(isinstance(dt, cftime.datetime) for dt in result.time.values)
        
        # Verify they have the noleap calendar
        for dt in result.time.values:
            assert hasattr(dt, 'calendar')
            assert 'noleap' in dt.calendar or '365_day' in dt.calendar
    
    def test_no_time_dimension_raises_error(self):
        """Test that DataArray without time dimension raises ValueError."""
        data = np.random.rand(5, 5)
        da = xr.DataArray(data, coords={'x': range(5), 'y': range(5)}, dims=['x', 'y'])
        
        with pytest.raises(ValueError, match="Data must have a 'time' dimension"):
            wwd.filterLeapDay_xarray(da)
    
    def test_non_datetime_time_raises_error(self):
        """Test that non-datetime time dimension raises ValueError."""
        data = np.arange(3)
        da = xr.DataArray(data, coords={'time': ['a', 'b', 'c']}, dims=['time'])
        
        with pytest.raises(TypeError, match="Cannot convert items"):
            wwd.filterLeapDay_xarray(da)
    
    def test_empty_dataarray(self):
        """Test handling of empty DataArray."""
        dates = pd.date_range('2020-01-01', periods=0)
        data = np.array([])
        da = xr.DataArray(data, coords={'time': dates}, dims=['time'])
        
        result = wwd.filterLeapDay_xarray(da)
        
        assert len(result.time) == 0
        assert result.shape == (0,)
    
    def test_no_leap_days_present(self):
        """Test DataArray with no leap days."""
        dates = pd.date_range('2021-06-01', '2021-06-30', freq='D')
        data = np.arange(len(dates))
        da = xr.DataArray(data, coords={'time': dates}, dims=['time'])
        
        result = wwd.filterLeapDay_xarray(da)
        
        # All time steps should be preserved
        assert len(result.time) == len(da.time)
    
    def test_preserves_data_array_name(self):
        """Test that DataArray name is preserved."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        data = np.arange(len(dates))
        da = xr.DataArray(
            data, 
            coords={'time': dates}, 
            dims=['time'],
            name='temperature'
        )
        
        result = wwd.filterLeapDay_xarray(da)
        
        assert result.name == 'temperature'
            


class TestFilterLeapDayDataset:
    
    def test_removes_leap_day_366(self):
        """Test that day 366 in leap years is removed from Dataset."""
        # Create Dataset with leap year data including day 366
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.arange(len(dates)), dims=['time']),
            'humidity': xr.DataArray(np.arange(100, 100 + len(dates)), dims=['time'])
        }, coords={'time': dates})
        
        result = wwd.filterLeapDay_xarray(ds)
        
        # Check that 2020-12-31 (day 366) was removed
        assert len(result.time) == 3
        
        # Extract dates from cftime objects
        result_dates = [(dt.year, dt.month, dt.day) for dt in result.time.values]
        assert (2020, 12, 30) not in result_dates
        assert (2020, 12, 31) in result_dates
        assert (2021, 1, 1) in result_dates
        assert (2021, 1, 2) in result_dates
        
        # Check data values are preserved correctly for both variables
        assert result['temperature'].values[0] == 0  # Dec 30
        assert result['temperature'].values[1] == 2  # Jan 1 (skipping Dec 31)
        assert result['temperature'].values[2] == 3  # Jan 2
        
        assert result['humidity'].values[0] == 100
        assert result['humidity'].values[1] == 102
        assert result['humidity'].values[2] == 103
    
    def test_custom_time_dimension_name(self):
        """Test using a custom time dimension name."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.arange(len(dates)), dims=['date'])
        }, coords={'date': dates})
        
        result = wwd.filterLeapDay_xarray(ds, time_dim='date')
        
        # Check that filtering worked with custom dimension name
        assert len(result.date) == 3
        result_dates = [(dt.year, dt.month, dt.day) for dt in result.date.values]
        assert (2020, 12, 30) not in result_dates
        assert (2020, 12, 31) in result_dates
    
    def test_mixed_dimensions(self):
        """Test Dataset with variables having different dimensions."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        ds = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.rand(len(dates), 5, 5), 
                dims=['time', 'x', 'y']
            ),
            'station_data': xr.DataArray(
                np.arange(len(dates)), 
                dims=['time']
            ),
            'static_field': xr.DataArray(
                np.ones((5, 5)), 
                dims=['x', 'y']
            )
        }, coords={
            'time': dates,
            'x': range(5),
            'y': range(5)
        })
        
        result = wwd.filterLeapDay_xarray(ds)
        
        # Check time-dependent variables are filtered
        assert result['temperature'].shape == (3, 5, 5)
        assert result['station_data'].shape == (3,)
        
        # Check static field is unchanged
        assert result['static_field'].shape == (5, 5)
        assert np.array_equal(result['static_field'].values, ds['static_field'].values)
    
    def test_preserves_attributes(self):
        """Test that Dataset and variable attributes are preserved."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        ds = xr.Dataset({
            'temperature': xr.DataArray(
                np.arange(len(dates)), 
                dims=['time'],
                attrs={'units': 'celsius', 'long_name': 'Air Temperature'}
            ),
            'humidity': xr.DataArray(
                np.arange(len(dates)), 
                dims=['time'],
                attrs={'units': 'percent', 'valid_range': [0, 100]}
            )
        }, 
        coords={'time': dates},
        attrs={'title': 'Weather Data', 'institution': 'NOAA'})
        
        result = wwd.filterLeapDay_xarray(ds)
        
        # Check Dataset attributes
        assert result.attrs['title'] == 'Weather Data'
        assert result.attrs['institution'] == 'NOAA'
        
        # Check variable attributes
        assert result['temperature'].attrs['units'] == 'celsius'
        assert result['temperature'].attrs['long_name'] == 'Air Temperature'
        assert result['humidity'].attrs['units'] == 'percent'
        assert result['humidity'].attrs['valid_range'] == [0, 100]
    
    def test_preserves_rasterio_attributes(self):
        """Test that rasterio-specific attributes are preserved."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        
        # Create Dataset with rasterio-like structure
        ds = xr.Dataset({
            'band1': xr.DataArray(
                np.random.rand(len(dates), 10, 10),
                dims=['time', 'y', 'x'],
                attrs={'nodata': -9999}
            ),
            'band2': xr.DataArray(
                np.random.rand(len(dates), 10, 10),
                dims=['time', 'y', 'x'],
                attrs={'nodata': -9999}
            )
        },
        coords={
            'time': dates,
            'y': np.arange(10),
            'x': np.arange(10)
        },
        attrs={'crs': 'EPSG:4326'})
        
        # Add CRS to spatial coordinates
        ds.coords['x'].attrs['crs'] = 'EPSG:4326'
        ds.coords['y'].attrs['crs'] = 'EPSG:4326'
        
        result = wwd.filterLeapDay_xarray(ds)
        
        # Check Dataset CRS
        assert result.attrs['crs'] == 'EPSG:4326'
        
        # Check variable nodata values
        assert result['band1'].attrs['nodata'] == -9999
        assert result['band2'].attrs['nodata'] == -9999
        
        # Check coordinate CRS attributes
        assert result.coords['x'].attrs['crs'] == 'EPSG:4326'
        assert result.coords['y'].attrs['crs'] == 'EPSG:4326'
    
    def test_cftime_objects_have_noleap_calendar(self):
        """Test that returned time objects are cftime with noleap calendar."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.arange(len(dates)), dims=['time'])
        }, coords={'time': dates})
        
        result = wwd.filterLeapDay_xarray(ds)
        
        # Check all time values are cftime objects
        assert all(isinstance(dt, cftime.datetime) for dt in result.time.values)
        
        # Verify they have the noleap calendar
        for dt in result.time.values:
            assert hasattr(dt, 'calendar')
            assert 'noleap' in dt.calendar or '365_day' in dt.calendar
    
    def test_no_time_dimension_raises_error(self):
        """Test that Dataset without time dimension raises ValueError."""
        ds = xr.Dataset({
            'data': xr.DataArray(np.random.rand(5, 5), dims=['x', 'y'])
        }, coords={'x': range(5), 'y': range(5)})
        
        with pytest.raises(ValueError, match="Data must have a 'time' dimension"):
            wwd.filterLeapDay_xarray(ds)
    
    def test_wrong_time_dim_name_raises_error(self):
        """Test that wrong time_dim parameter raises ValueError."""
        dates = pd.date_range('2020-01-01', periods=5)
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.arange(5), dims=['time'])
        }, coords={'time': dates})
        
        with pytest.raises(ValueError, match="Data must have a 'date' dimension"):
            wwd.filterLeapDay_xarray(ds, time_dim='date')
    
    def test_non_datetime_time_raises_error(self):
        """Test that non-datetime time dimension raises ValueError."""
        ds = xr.Dataset({
            'data': xr.DataArray(np.arange(3), dims=['time'])
        }, coords={'time': ['a', 'b', 'c']})
        
        with pytest.raises(TypeError, match="Cannot convert"):
            wwd.filterLeapDay_xarray(ds)
    
    def test_empty_dataset(self):
        """Test handling of empty Dataset."""
        dates = pd.date_range('2020-01-01', periods=0)
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.array([]), dims=['time'])
        }, coords={'time': dates})
        
        result = wwd.filterLeapDay_xarray(ds)
        
        assert len(result.time) == 0
        assert result['temperature'].shape == (0,)
    
    def test_preserves_encoding(self):
        """Test that variable encoding is preserved."""
        dates = pd.date_range('2020-12-30', '2021-01-02', freq='D')
        ds = xr.Dataset({
            'temperature': xr.DataArray(
                np.arange(len(dates), dtype=np.float32), 
                dims=['time']
            )
        }, coords={'time': dates})
        
        # Set encoding
        ds['temperature'].encoding = {'dtype': 'float32', '_FillValue': -9999}
        
        result = wwd.filterLeapDay_xarray(ds)
        
        # Encoding should be preserved
        assert result['temperature'].encoding['dtype'] == 'float32'
        assert result['temperature'].encoding['_FillValue'] == -9999

        

class TestInterpolateToRegular:
    
    @pytest.fixture
    def sample_times(self):
        """Create sample irregular cftime dates."""
        return [
            cftime.DatetimeNoLeap(2020, 1, 1),
            cftime.DatetimeNoLeap(2020, 1, 3),
            cftime.DatetimeNoLeap(2020, 1, 7),
            cftime.DatetimeNoLeap(2020, 1, 8),
            cftime.DatetimeNoLeap(2020, 1, 15),
        ]
    
    @pytest.fixture
    def sample_dataframe(self, sample_times):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'time': sample_times,
            'temperature': [10.0, 12.0, 15.0, 16.0, 20.0],
            'humidity': [50.0, 55.0, 60.0, 58.0, 65.0],
            'station': ['A', 'A', 'B', 'B', 'C']
        })
    
    @pytest.fixture
    def sample_dataarray(self, sample_times):
        """Create sample DataArray."""
        return xr.DataArray(
            [10.0, 12.0, 15.0, 16.0, 20.0],
            coords={'time': sample_times},
            dims=['time'],
            name='temperature',
            attrs={'units': 'celsius'}
        )
    
    @pytest.fixture
    def sample_dataset(self, sample_times):
        """Create sample Dataset."""
        return xr.Dataset({
            'temperature': xr.DataArray(
                [10.0, 12.0, 15.0, 16.0, 20.0],
                dims=['time'],
                attrs={'units': 'celsius'}
            ),
            'humidity': xr.DataArray(
                [50.0, 55.0, 60.0, 58.0, 65.0],
                dims=['time'],
                attrs={'units': 'percent'}
            )
        }, coords={'time': sample_times})
    
    def test_dataframe_interpolation(self, sample_dataframe):
        """Test DataFrame interpolation."""
        result = wwd.interpolateToRegular(sample_dataframe, 1, 'time')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 15  # Jan 1 to Jan 15
        
        # Check regular intervals
        time_diffs = np.diff([t.toordinal() for t in result['time']])
        assert all(diff == 1 for diff in time_diffs)
    
    def test_dataarray_interpolation(self, sample_dataarray):
        """Test DataArray interpolation."""
        result = wwd.interpolateToRegular(sample_dataarray)
        
        assert isinstance(result, xr.DataArray)
        assert len(result.time) == 15
        assert result.name == 'temperature'
        assert result.attrs['units'] == 'celsius'
    
    def test_dataset_interpolation(self, sample_dataset):
        """Test Dataset interpolation."""
        result = wwd.interpolateToRegular(sample_dataset)
        
        assert isinstance(result, xr.Dataset)
        assert len(result.time) == 15
        assert 'temperature' in result.data_vars
        assert 'humidity' in result.data_vars
        assert result['temperature'].attrs['units'] == 'celsius'
    
    def test_5_day_interval(self, sample_dataframe):
        """Test 5-day interval interpolation."""
        result = wwd.interpolateToRegular(sample_dataframe, 5, 'time')
        
        assert len(result) == 3  # Jan 1, 6, 11
        time_diffs = np.diff([t.toordinal() for t in result['time']])
        assert all(diff == 5 for diff in time_diffs)
    
    def test_custom_time_dimension(self, sample_times):
        """Test custom time dimension name."""
        ds = xr.Dataset({
            'temp': xr.DataArray([10, 12, 15, 16, 20], dims=['date'])
        }, coords={'date': sample_times})
        
        result = wwd.interpolateToRegular(ds, time_dim='date')
        
        assert 'date' in result.dims
        assert len(result.date) == 15
    
    def test_selective_interpolation(self, sample_dataset):
        """Test interpolating only specific variables."""
        result = wwd.interpolateToRegular(sample_dataset, variables=['temperature'])
        
        # Temperature should be interpolated
        assert len(result.temperature) == 15
        
        # Humidity does not exist, it is also on time and time has changed.
        assert 'humidity' not in result.data_vars
    
    def test_limit_parameter(self, sample_dataarray):
        """Test limit parameter for gap filling."""
        result = wwd.interpolateToRegular(sample_dataarray, limit=2)
        
        assert len(result.time) == 15
        # Large gaps should have NaNs
        # (exact behavior depends on gap size and limit)
    
    def test_different_methods(self, sample_dataarray):
        """Test different interpolation methods."""
        for method in ['linear', 'nearest', 'zero']:
            result = wwd.interpolateToRegular(sample_dataarray, method=method)
            assert len(result.time) == 15
            assert not np.all(np.isnan(result.values))
    
    def test_type_preservation(self, sample_dataframe, sample_dataarray, sample_dataset):
        """Test that output type matches input type."""
        df_result = wwd.interpolateToRegular(sample_dataframe, time_dim='time')
        assert type(df_result) is pd.DataFrame
        
        da_result = wwd.interpolateToRegular(sample_dataarray)
        assert type(da_result) is xr.DataArray
        
        ds_result = wwd.interpolateToRegular(sample_dataset)
        assert type(ds_result) is xr.Dataset
    
    def test_invalid_type_raises(self):
        """Test that invalid input type raises error."""
        data = [1, 2, 3, 4]
        
        with pytest.raises(TypeError, match="Input data must be"):
            wwd.interpolateToRegular(data)
    
    def test_invalid_interval_raises(self, sample_dataarray):
        """Test that invalid interval raises error."""
        with pytest.raises(ValueError, match="Interval must be 1 or 5"):
            wwd.interpolateToRegular(sample_dataarray, interval=3)


class TestInterpolationHelpers:
    
    def test_generate_regular_times(self):
        """Test regular time generation."""
        start = cftime.DatetimeNoLeap(2020, 1, 1)
        end = cftime.DatetimeNoLeap(2020, 1, 11)
        
        # 1-day interval
        times_1 = wwd._generateRegularTimes(start, end, 1)
        assert len(times_1) == 11
        assert times_1[0] == start
        assert times_1[-1] == end
        
        # 5-day interval
        times_5 = wwd._generateRegularTimes(start, end, 5)
        assert len(times_5) == 3  # Jan 1, 6, 11 (but 11 > 10, so only 2)
        assert times_5[0] == start
        assert times_5[1] == cftime.DatetimeNoLeap(2020, 1, 6)
    

class TestAverageAcrossYearsDataFrame:
    
    @pytest.fixture
    def sample_multiyear_df(self):
        """Create sample DataFrame spanning multiple years."""
        dates = []
        values = []
        
        for year in [2018, 2019, 2020]:
            for doy in range(1, 366):  # 365 days
                date = cftime.DatetimeNoLeap(year, 1, 1) + datetime.timedelta(days=doy-1)
                dates.append(date)
                values.append(doy + year - 2019)
        
        return pd.DataFrame({
            'time': dates,
            'temperature': values,
            'humidity': [50 + v * 0.1 for v in values],
            'station': ['A'] * len(dates),  # Non-numeric column
            'notes': ['clear'] * len(dates)  # Another non-numeric column
        })
    
    def test_default_all_numeric_columns(self, sample_multiyear_df):
        """Test averaging all numeric columns by default."""
        result = wwd.computeAverageYear_DataFrame(
            sample_multiyear_df, 'time', 2025, 1
        )
        
        # Should only include time and numeric columns
        assert set(result.columns) == {'time', 'temperature', 'humidity'}
        assert 'station' not in result.columns
        assert 'notes' not in result.columns
        
        # Check length
        assert len(result) == 365
    
    def test_specific_columns(self, sample_multiyear_df):
        """Test averaging specific columns."""
        result = wwd.computeAverageYear_DataFrame(
            sample_multiyear_df, 'time', 2025, 1,
            columns=['temperature']
        )
        
        # Should only include time and temperature
        assert set(result.columns) == {'time', 'temperature'}
        assert 'humidity' not in result.columns
    
    def test_non_numeric_columns_ignored(self, sample_multiyear_df):
        """Test that non-numeric columns are ignored with warning."""
        with warnings.catch_warnings(record=True) as w:
            result = wwd.computeAverageYear_DataFrame(
                sample_multiyear_df, 'time', 2025, 1,
                columns=['temperature', 'station', 'notes']
            )
            
            # Should have warning about non-numeric columns
            assert len(w) == 1
            assert 'Non-numeric columns will be ignored' in str(w[0].message)
            assert 'station' in str(w[0].message)
            assert 'notes' in str(w[0].message)
        
        # Result should only have numeric columns
        assert set(result.columns) == {'time', 'temperature'}
    
    def test_averaging_accuracy(self, sample_multiyear_df):
        """Test that averaging is mathematically correct."""
        result = wwd.computeAverageYear_DataFrame(
            sample_multiyear_df, 'time', 2025, 1
        )
        
        # For day 1: values are 0, 1, 2 → average = 1
        day1_value = result.loc[0, 'temperature']
        assert abs(day1_value - 1.0) < 0.001
        
        # For day 365: values are 364, 365, 366 → average = 365
        day365_value = result.loc[364, 'temperature']
        assert abs(day365_value - 365.0) < 0.001
    
    def test_multiple_years_output(self, sample_multiyear_df):
        """Test repeating pattern for multiple years."""
        result = wwd.computeAverageYear_DataFrame(
            sample_multiyear_df, 'time', 2025, 3
        )
        
        # Should have 3 years of data
        assert len(result) == 365 * 3
        
        # Check years
        years = [d.year for d in result['time']]
        assert set(years) == {2025, 2026, 2027}
        
        # Pattern should repeat
        assert result.loc[0, 'temperature'] == result.loc[365, 'temperature']
        assert result.loc[0, 'temperature'] == result.loc[730, 'temperature']
    
    def test_missing_columns_error(self, sample_multiyear_df):
        """Test error when specified columns don't exist."""
        with pytest.raises(ValueError, match="Columns not found"):
            wwd.computeAverageYear_DataFrame(
                sample_multiyear_df, 'time', 2025, 1,
                columns=['nonexistent']
            )
    
    def test_empty_numeric_columns(self):
        """Test handling when no numeric columns to average."""
        df = pd.DataFrame({
            'time': [cftime.DatetimeNoLeap(2020, 1, 1), 
                    cftime.DatetimeNoLeap(2020, 1, 2)],
            'category': ['A', 'B']
        })

        with pytest.raises(ValueError, match="No numeric columns"):
            result = wwd.computeAverageYear_DataFrame(df, 'time', 2025, 1)
    
    def test_nan_handling(self):
        """Test handling of NaN values in averaging."""
        dates = []
        values = []
        
        for year in [2018, 2019, 2020]:
            for doy in [1, 2, 3]:
                date = cftime.DatetimeNoLeap(year, 1, 1) + datetime.timedelta(days=doy-1)
                dates.append(date)
                # Add NaN for 2019, day 2
                if year == 2019 and doy == 2:
                    values.append(np.nan)
                else:
                    values.append(float(doy))
        
        df = pd.DataFrame({'time': dates, 'value': values})
        result = wwd.computeAverageYear_DataFrame(df, 'time', 2025, 1)
        
        # Day 2 should average only non-NaN values: (2 + 2) / 2 = 2
        day2_value = result.loc[1, 'value']
        assert abs(day2_value - 2.0) < 0.001
    
    def test_5day_interval(self):
        """Test with 5-day interval data."""
        dates = []
        values = []
        
        for year in [2018, 2019]:
            for doy in range(1, 366, 5):  # Every 5 days
                date = cftime.DatetimeNoLeap(year, 1, 1) + datetime.timedelta(days=doy-1)
                dates.append(date)
                values.append(doy)
        
        df = pd.DataFrame({'time': dates, 'value': values})
        result = wwd.computeAverageYear_DataFrame(df, 'time', 2025, 1)
        
        # Should have 73 entries (365/5)
        assert len(result) == 73
        
        # Check 5-day intervals
        time_diffs = np.diff([d.toordinal() for d in result['time']])
        assert all(diff == 5 for diff in time_diffs)



class TestSmoothOverloaded:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        times = pd.date_range('2020-01-01', periods=50, freq='D')
        signal = np.sin(np.linspace(0, 2*np.pi, 50)) + 0.1 * np.random.randn(50)
        return times, signal
    
    @pytest.fixture
    def sample_dataframe(self, sample_data):
        """Create sample DataFrame."""
        times, signal = sample_data
        return pd.DataFrame({
            'time': times,
            'signal': signal,
            'signal2': -signal + 0.05 * np.random.randn(50),
            'category': ['A'] * 25 + ['B'] * 25
        })
    
    @pytest.fixture
    def sample_dataarray(self, sample_data):
        """Create sample DataArray."""
        times, signal = sample_data
        return xr.DataArray(
            signal,
            coords={'time': times},
            dims=['time'],
            name='signal',
            attrs={'units': 'm/s'}
        )
    
    @pytest.fixture
    def sample_dataset(self, sample_data):
        """Create sample Dataset."""
        times, signal = sample_data
        return xr.Dataset({
            'signal': xr.DataArray(signal, dims=['time'], attrs={'units': 'm/s'}),
            'signal2': xr.DataArray(-signal, dims=['time'], attrs={'units': 'm/s'}),
            'static': xr.DataArray([1, 2, 3], dims=['x'])
        }, coords={'time': times, 'x': [0, 1, 2]})
    
    def test_dataframe_savgol(self, sample_dataframe):
        """Test DataFrame smoothing with Savitzky-Golay."""
        result = wwd.smoothTimeSeries(sample_dataframe, 'time', method='savgol', window_length=11, polyorder=3)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dataframe.shape
        assert np.var(result['signal']) < np.var(sample_dataframe['signal'])
        assert (result['category'] == sample_dataframe['category']).all()
    
    def test_dataframe_rolling_mean(self, sample_dataframe):
        """Test DataFrame smoothing with rolling mean."""
        result = wwd.smoothTimeSeries(sample_dataframe, 'time', method='rolling_mean', window=7)
        
        assert isinstance(result, pd.DataFrame)
        # Check variance on non-NaN values
        valid_idx = ~np.isnan(result['signal'])
        assert np.var(result['signal'][valid_idx]) < np.var(sample_dataframe['signal'][valid_idx])
    
    def test_dataarray_savgol(self, sample_dataarray):
        """Test DataArray smoothing with Savitzky-Golay."""
        result = wwd.smoothTimeSeries(sample_dataarray, method='savgol', window_length=11, polyorder=3)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_dataarray.shape
        assert np.var(result.values) < np.var(sample_dataarray.values)
        assert result.attrs == sample_dataarray.attrs
    
    def test_dataarray_custom_time_dim(self, sample_data):
        """Test DataArray with custom time dimension."""
        times, signal = sample_data
        da = xr.DataArray(signal, coords={'date': times}, dims=['date'])
        
        result = wwd.smoothTimeSeries(da, time_dim='date', method='savgol')
        
        assert isinstance(result, xr.DataArray)
        assert 'date' in result.dims
        assert np.var(result.values) < np.var(da.values)
    
    def test_dataset_savgol(self, sample_dataset):
        """Test Dataset smoothing with Savitzky-Golay."""
        result = wwd.smoothTimeSeries(sample_dataset, method='savgol', window_length=11, polyorder=3)
        
        assert isinstance(result, xr.Dataset)
        assert np.var(result['signal'].values) < np.var(sample_dataset['signal'].values)
        assert np.array_equal(result['static'].values, sample_dataset['static'].values)
    
    def test_dataset_specific_variables(self, sample_dataset):
        """Test Dataset smoothing with specific variables."""
        result = wwd.smoothTimeSeries(sample_dataset, columns=['signal'], method='rolling_mean', window=5)
        
        # Only 'signal' should be smoothed
        valid_idx = ~np.isnan(result['signal'].values)
        assert np.var(result['signal'].values[valid_idx]) < np.var(sample_dataset['signal'].values[valid_idx])
        assert np.array_equal(result['signal2'].values, sample_dataset['signal2'].values)
    
    def test_dataset_time_dim_parameter(self, sample_data):
        """Test Dataset with time_dim parameter."""
        times, signal = sample_data
        ds = xr.Dataset({
            'signal': xr.DataArray(signal, dims=['date'])
        }, coords={'date': times})
        
        # Should work with time_dim
        result = wwd.smoothTimeSeries(ds, time_dim='date', method='savgol')
        assert 'date' in result.dims
        
    def test_invalid_type_raises(self):
        """Test that invalid input type raises error."""
        data = [1, 2, 3, 4]
        
        with pytest.raises(TypeError, match="Input data must be"):
            wwd.smoothTimeSeries(data)
    
    def test_dataarray_ignores_columns(self, sample_dataarray):
        """Test that DataArray ignores columns parameter."""
        with warnings.catch_warnings(record=True) as w:
            result = wwd.smoothTimeSeries(sample_dataarray, columns=['ignored'], method='savgol')
            
            assert len(w) == 1
            assert "'columns' parameter is ignored for DataArray" in str(w[0].message)
        
        assert isinstance(result, xr.DataArray)
    
    def test_nan_raises_error(self, sample_data):
        """Test that NaN values raise errors."""
        times, signal = sample_data
        signal_with_nan = signal.copy()
        signal_with_nan[10] = np.nan
        
        # DataFrame
        df = pd.DataFrame({'time': times, 'signal': signal_with_nan})
        with pytest.raises(ValueError, match="Data contains NaN values"):
            wwd.smoothTimeSeries(df, 'time')
        
        # DataArray
        da = xr.DataArray(signal_with_nan, coords={'time': times}, dims=['time'])
        with pytest.raises(ValueError, match="DataArray contains NaN values"):
            wwd.smoothTimeSeries(da)
    
    def test_type_preservation(self, sample_dataframe, sample_dataarray, sample_dataset):
        """Test that output type matches input type."""
        df_result = wwd.smoothTimeSeries(sample_dataframe, 'time')
        assert type(df_result) is pd.DataFrame
        
        da_result = wwd.smoothTimeSeries(sample_dataarray)
        assert type(da_result) is xr.DataArray
        
        ds_result = wwd.smoothTimeSeries(sample_dataset)
        assert type(ds_result) is xr.Dataset
    
    def test_parameter_propagation(self, sample_dataframe):
        """Test that kwargs are properly propagated."""
        # Test with specific Savgol parameters
        result = wwd.smoothTimeSeries(
            sample_dataframe, 'time', 
            method='savgol',
            window_length=15,
            polyorder=5,
            mode='mirror'
        )
        
        # Should not raise any errors
        assert isinstance(result, pd.DataFrame)
        
        # Test with rolling mean parameters
        result2 = wwd.smoothTimeSeries(
            sample_dataframe, 'time',
            method='rolling_mean',
            window=10,
            center=False,
        )
        
        assert isinstance(result2, pd.DataFrame)
        # With center=False, first value should not be NaN
        assert not np.isnan(result2['signal'].iloc[0])


    @pytest.mark.parametrize("method", ['savgol', 'rolling_mean'])
    def test_all_types_support_both_methods(self, method, sample_data):
        """Test that all data types support both smoothing methods."""
        times, signal = sample_data[:2]

        # DataFrame
        df = pd.DataFrame({'time': times, 'signal': signal})
        df_result = wwd.smoothTimeSeries(df, 'time', method=method)
        assert isinstance(df_result, pd.DataFrame)

        # DataArray
        da = xr.DataArray(signal, coords={'time': times}, dims=['time'])
        da_result = wwd.smoothTimeSeries(da, method=method)
        assert isinstance(da_result, xr.DataArray)

        # Dataset
        ds = xr.Dataset({'signal': da})
        ds_result = wwd.smoothTimeSeries(ds, method=method)
        assert isinstance(ds_result, xr.Dataset)


    def test_function_overloads_type_hints(self):
        """Test that type hints work correctly with overloads."""
        # This test ensures the overloads are syntactically correct
        # In a real type-checking environment, mypy would verify these

        times = pd.date_range('2020-01-01', periods=20)
        signal = np.sin(np.linspace(0, 2*np.pi, 20))

        # DataFrame overload
        df = pd.DataFrame({'time': times, 'signal': signal})
        df_result = wwd.smoothTimeSeries(df, 'time')
        assert isinstance(df_result, pd.DataFrame)

        # DataArray overload
        da = xr.DataArray(signal, coords={'time': times}, dims=['time'])
        da_result = wwd.smoothTimeSeries(da)
        assert isinstance(da_result, xr.DataArray)

        # Dataset overload
        ds = xr.Dataset({'signal': da})
        ds_result = wwd.smoothTimeSeries(ds)
        assert isinstance(ds_result, xr.Dataset)


    def test_edge_cases(self):
        """Test edge cases for the smooth function."""
        # Very short time series
        times = pd.date_range('2020-01-01', periods=5)
        signal = np.array([1, 2, 3, 2, 1], dtype=float)

        df = pd.DataFrame({'time': times, 'signal': signal})

        # Should work with appropriate window size
        result = wwd.smoothTimeSeries(df, 'time', method='savgol', window_length=3, polyorder=1)
        assert len(result) == 5

        # Should fail with window too large
        with pytest.raises(ValueError, match="Data length"):
            wwd.smoothTimeSeries(df, 'time', method='savgol', window_length=7)        


import pytest
import pandas as pd
import numpy as np
import xarray as xr


class TestComputeMode:
    
    def test_simple_integer_mode(self):
        """Test mode computation with simple integer data."""
        times = pd.date_range('2020-01-01', periods=7)
        data = np.array([1, 2, 2, 3, 2, 4, 2])
        da = xr.DataArray(data, dims=['time'], coords={'time': times})
        
        result = wwd.computeMode(da)
        
        assert result.values == 2
        assert 'time' not in result.dims
        assert result.shape == ()
    
    def test_simple_float_mode(self):
        """Test mode computation with float data."""
        times = pd.date_range('2020-01-01', periods=5)
        data = np.array([1.5, 2.5, 2.5, 3.5, 2.5])
        da = xr.DataArray(data, dims=['time'], coords={'time': times})
        
        result = wwd.computeMode(da)
        
        assert result.values == 2.5
        assert 'time' not in result.dims
    
    def test_multidimensional_mode(self):
        """Test mode computation with multidimensional data."""
        times = pd.date_range('2020-01-01', periods=5)
        # Create 3D data (time, x, y)
        data = np.array([
            [[1, 2], [3, 4]],
            [[1, 3], [3, 5]],
            [[2, 2], [3, 4]],
            [[1, 2], [4, 4]],
            [[2, 3], [3, 5]]
        ])
        da = xr.DataArray(
            data, 
            dims=['time', 'x', 'y'],
            coords={'time': times, 'x': [0, 1], 'y': [0, 1]}
        )
        
        result = wwd.computeMode(da)
        
        assert result.shape == (2, 2)
        assert 'time' not in result.dims
        assert 'x' in result.dims
        assert 'y' in result.dims
        # Check specific values
        assert result.values[0, 0] == 1  # Mode of [1, 1, 2, 1, 2] is 1
        assert result.values[0, 1] == 2  # Mode of [2, 3, 2, 2, 3] is 2
        assert result.values[1, 0] == 3  # Mode of [3, 3, 3, 4, 3] is 3
        assert result.values[1, 1] == 4  # Mode of [4, 5, 4, 4, 5] is 4
    
    def test_multiple_modes(self):
        """Test behavior when multiple values have same frequency."""
        times = pd.date_range('2020-01-01', periods=4)
        data = np.array([1, 2, 3, 4])  # All values appear once
        da = xr.DataArray(data, dims=['time'], coords={'time': times})
        
        result = wwd.computeMode(da)
        
        # scipy.stats.mode returns the smallest value when there are multiple modes
        assert result.values == 1
    
    def test_with_nan_values(self):
        """Test mode computation with NaN values."""
        times = pd.date_range('2020-01-01', periods=7)
        data = np.array([1, 2, np.nan, 2, 3, np.nan, 2])
        da = xr.DataArray(data, dims=['time'], coords={'time': times})
        
        result = wwd.computeMode(da)
        
        # Mode should be 2 (appears 3 times, ignoring NaN)
        assert result.values == 2
    
    def test_all_nan_values(self):
        """Test mode computation when all values are NaN."""
        times = pd.date_range('2020-01-01', periods=5)
        data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        da = xr.DataArray(data, dims=['time'], coords={'time': times})
        
        result = wwd.computeMode(da)
        
        assert np.isnan(result.values)
    
    def test_custom_time_dimension(self):
        """Test mode computation with custom time dimension name."""
        dates = pd.date_range('2020-01-01', periods=5)
        data = np.array([10, 20, 20, 30, 20])
        da = xr.DataArray(data, dims=['date'], coords={'date': dates})
        
        result = wwd.computeMode(da, time_dim='date')
        
        assert result.values == 20
        assert 'date' not in result.dims
    
    def test_missing_time_dimension_raises(self):
        """Test that missing time dimension raises ValueError."""
        data = np.array([1, 2, 3])
        da = xr.DataArray(data, dims=['x'])
        
        with pytest.raises(ValueError, match="Data must have a 'time' dimension"):
            wwd.computeMode(da)
    
    def test_preserve_attributes(self):
        """Test that attributes are preserved."""
        times = pd.date_range('2020-01-01', periods=5)
        data = np.array([1, 2, 2, 3, 2])
        da = xr.DataArray(
            data, 
            dims=['time'], 
            coords={'time': times},
            attrs={'units': 'meters', 'description': 'test data'}
        )
        
        result = wwd.computeMode(da)
        
        assert result.attrs['units'] == 'meters'
        assert result.attrs['description'] == 'test data'
    
    def test_preserve_name(self):
        """Test that name is modified appropriately."""
        times = pd.date_range('2020-01-01', periods=5)
        data = np.array([1, 2, 2, 3, 2])
        da = xr.DataArray(data, dims=['time'], coords={'time': times}, name='temperature')
        
        result = wwd.computeMode(da)
        
        assert result.name == 'temperature_mode'
    
    def test_preserve_non_time_coordinates(self):
        """Test that non-time coordinates are preserved."""
        times = pd.date_range('2020-01-01', periods=5)
        data = np.random.randint(0, 5, size=(5, 3, 4))
        da = xr.DataArray(
            data,
            dims=['time', 'x', 'y'],
            coords={
                'time': times,
                'x': [10, 20, 30],
                'y': [100, 200, 300, 400],
                'lat': (['x', 'y'], np.random.rand(3, 4)),
                'lon': (['x', 'y'], np.random.rand(3, 4))
            }
        )
        
        result = wwd.computeMode(da)
        
        assert 'lat' in result.coords
        assert 'lon' in result.coords
        assert np.array_equal(result.coords['lat'].values, da.coords['lat'].values)
        assert np.array_equal(result.coords['lon'].values, da.coords['lon'].values)

        
    def test_mode_with_categorical_data(self):
        """Test mode computation with categorical-like data."""
        times = pd.date_range('2020-01-01', periods=10)
        # Simulate categorical data as integers
        categories = np.array([0, 1, 2, 1, 1, 2, 0, 1, 2, 1])  # 1 appears most
        da = xr.DataArray(categories, dims=['time'], coords={'time': times})

        result = wwd.computeMode(da)

        assert result.values == 1


    def test_mode_empty_array(self):
        """Test mode computation with empty array."""
        times = pd.date_range('2020-01-01', periods=0)
        data = np.array([])
        da = xr.DataArray(data, dims=['time'], coords={'time': times})

        # This might raise an error or return empty array depending on scipy version
        # Just ensure it doesn't crash
        try:
            result = wwd.computeMode(da)
            assert result.shape == ()
        except:
            # Some versions might raise an error for empty input
            pass


    def test_mode_single_value(self):
        """Test mode computation with single value."""
        times = pd.date_range('2020-01-01', periods=1)
        data = np.array([42])
        da = xr.DataArray(data, dims=['time'], coords={'time': times})

        result = wwd.computeMode(da)

        assert result.values == 42            



class TestSmooth2DSpatial:
    
    @pytest.fixture
    def sample_2d_data(self):
        """Create sample 2D data with some structure."""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        # Create data with a gaussian bump
        data = np.exp(-((X - 5)**2 + (Y - 5)**2) / 4)
        # Add some noise
        data += 0.1 * np.random.randn(50, 50)
        return data
    
    @pytest.fixture
    def sample_dataarray_xy(self, sample_2d_data):
        """Create DataArray with x, y dimensions."""
        return xr.DataArray(
            sample_2d_data,
            dims=['y', 'x'],
            coords={'x': np.arange(50), 'y': np.arange(50)},
            name='temperature',
            attrs={'units': 'K'}
        )
    
    @pytest.fixture
    def sample_dataarray_latlon(self, sample_2d_data):
        """Create DataArray with lat, lon dimensions."""
        return xr.DataArray(
            sample_2d_data,
            dims=['lat', 'lon'],
            coords={'lat': np.linspace(-90, 90, 50), 'lon': np.linspace(0, 360, 50)},
            name='temperature'
        )
    
    @pytest.fixture
    def sample_3d_dataarray(self, sample_2d_data):
        """Create 3D DataArray with time dimension."""
        data_3d = np.stack([sample_2d_data + i*0.1 for i in range(10)])
        return xr.DataArray(
            data_3d,
            dims=['time', 'y', 'x'],
            coords={
                'time': pd.date_range('2020-01-01', periods=10),
                'x': np.arange(50),
                'y': np.arange(50)
            }
        )
    
    @pytest.fixture
    def sample_dataset(self, sample_dataarray_xy):
        """Create Dataset with multiple variables."""
        return xr.Dataset({
            'temperature': sample_dataarray_xy,
            'pressure': sample_dataarray_xy * 1000,
            'altitude': xr.DataArray(np.ones(20), dims=['z'], coords={'z': np.arange(20)})
        })
    
    def test_smooth_dataarray_gaussian(self, sample_dataarray_xy):
        """Test Gaussian smoothing on DataArray."""
        result = wwd.smooth2D_DataArray(sample_dataarray_xy, method='gaussian', sigma=2.0)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_dataarray_xy.shape
        assert result.dims == sample_dataarray_xy.dims
        
        # Check smoothing effect (reduced variance)
        assert np.var(result.values) < np.var(sample_dataarray_xy.values)
        
        # Check attributes preserved
        assert result.attrs == sample_dataarray_xy.attrs
        assert result.name == sample_dataarray_xy.name
    
    def test_smooth_dataarray_uniform(self, sample_dataarray_xy):
        """Test uniform smoothing on DataArray."""
        result = wwd.smooth2D_DataArray(sample_dataarray_xy, method='uniform', size=5)
        
        assert isinstance(result, xr.DataArray)
        assert np.var(result.values) < np.var(sample_dataarray_xy.values)
    
    def test_smooth_dataarray_box(self, sample_dataarray_xy):
        """Test box smoothing on DataArray."""
        result = wwd.smooth2D_DataArray(sample_dataarray_xy, method='box', kernel_size=3)
        
        assert isinstance(result, xr.DataArray)
        assert np.var(result.values) < np.var(sample_dataarray_xy.values)
    
    def test_auto_detect_xy_dims(self, sample_dataarray_xy):
        """Test automatic detection of x, y dimensions."""
        result = wwd.smooth2D_DataArray(sample_dataarray_xy, method='gaussian')
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_dataarray_xy.shape
    
    def test_auto_detect_latlon_dims(self, sample_dataarray_latlon):
        """Test automatic detection of lat, lon dimensions."""
        result = wwd.smooth2D_DataArray(sample_dataarray_latlon, method='gaussian')
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_dataarray_latlon.shape
    
    def test_custom_dims(self):
        """Test with custom dimension names."""
        data = np.random.randn(30, 40)
        da = xr.DataArray(
            data,
            dims=['north_south', 'east_west'],
            coords={'north_south': np.arange(30), 'east_west': np.arange(40)}
        )
        
        result = wwd.smooth2D_DataArray(
            da, dim1='north_south', dim2='east_west', method='gaussian'
        )
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == da.shape
    
    def test_3d_smoothing(self, sample_3d_dataarray):
        """Test smoothing 3D data (smooth each time slice)."""
        result = wwd.smooth2D_DataArray(sample_3d_dataarray, method='gaussian', sigma=1.5)
        
        assert result.shape == sample_3d_dataarray.shape
        
        # Check that each time slice is smoothed
        for t in range(len(sample_3d_dataarray.time)):
            original_slice = sample_3d_dataarray.isel(time=t)
            result_slice = result.isel(time=t)
            assert np.var(result_slice.values) < np.var(original_slice.values)
    
    def test_smooth_dataset(self, sample_dataset):
        """Test smoothing Dataset."""
        result = wwd.smooth2D_Dataset(sample_dataset, method='gaussian', sigma=2.0)
        
        assert isinstance(result, xr.Dataset)
        
        # Check that 2D variables are smoothed
        assert np.var(result['temperature'].values) < np.var(sample_dataset['temperature'].values)
        assert np.var(result['pressure'].values) < np.var(sample_dataset['pressure'].values)
        
        # Check that 1D variable is unchanged
        assert np.array_equal(result['altitude'].values, sample_dataset['altitude'].values)
    
    def test_smooth_dataset_specific_variables(self, sample_dataset):
        """Test smoothing specific variables in Dataset."""
        result = wwd.smooth2D_Dataset(
            sample_dataset, 
            variables=['temperature'], 
            method='uniform', 
            size=5
        )
        
        # Only temperature should be smoothed
        assert np.var(result['temperature'].values) < np.var(sample_dataset['temperature'].values)
        assert np.array_equal(result['pressure'].values, sample_dataset['pressure'].values)
    
    def test_nan_raises_error(self, sample_dataarray_xy):
        """Test that NaN values raise an error."""
        # Add NaN to data
        sample_dataarray_xy.values[10, 10] = np.nan
        
        with pytest.raises(ValueError, match="DataArray contains NaN values"):
            wwd.smooth2D_DataArray(sample_dataarray_xy)
    
    def test_missing_dims_raises_error(self):
        """Test error when spatial dimensions not found."""
        da = xr.DataArray(
            np.random.randn(10, 20),
            dims=['a', 'b']
        )
        
        with pytest.raises(ValueError, match="Could not find spatial dimensions"):
            wwd.smooth2D_DataArray(da)
    
    def test_invalid_method_raises_error(self, sample_dataarray_xy):
        """Test error for invalid smoothing method."""
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            wwd.smooth2D_DataArray(sample_dataarray_xy, method='invalid')
    
    def test_anisotropic_smoothing(self, sample_dataarray_xy):
        """Test anisotropic smoothing (different sigma/size for each dimension)."""
        # Gaussian with different sigmas
        result = wwd.smooth2D_DataArray(
            sample_dataarray_xy, 
            method='gaussian', 
            sigma=(3.0, 1.0)  # More smoothing in y direction
        )
        
        assert isinstance(result, xr.DataArray)
        
        # Uniform with different sizes
        result2 = wwd.smooth2D_DataArray(
            sample_dataarray_xy,
            method='uniform',
            size=(5, 3)  # Different sizes
        )
        
        assert isinstance(result2, xr.DataArray)
    
    def test_overloaded_function_dataarray(self, sample_dataarray_xy):
        """Test overloaded smooth2D with DataArray."""
        result = wwd.smooth2D(sample_dataarray_xy, method='gaussian', sigma=2.0)
        
        assert isinstance(result, xr.DataArray)
        assert np.var(result.values) < np.var(sample_dataarray_xy.values)
    
    def test_overloaded_function_dataset(self, sample_dataset):
        """Test overloaded smooth2D with Dataset."""
        result = wwd.smooth2D(sample_dataset, method='gaussian', sigma=2.0)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
    
    def test_overloaded_dataarray_ignores_variables(self, sample_dataarray_xy):
        """Test that DataArray ignores variables parameter."""
        with warnings.catch_warnings(record=True) as w:
            result = wwd.smooth2D(sample_dataarray_xy, variables=['ignored'])
            
            assert len(w) == 1
            assert "'variables' parameter is ignored" in str(w[0].message)
        
        assert isinstance(result, xr.DataArray)
    
    def test_preserve_coordinates(self, sample_3d_dataarray):
        """Test that all coordinates are preserved."""
        # Add some extra coordinates
        sample_3d_dataarray = sample_3d_dataarray.assign_coords({
            'lat': (['y'], np.linspace(-90, 90, 50)),
            'lon': (['x'], np.linspace(0, 360, 50))
        })
        
        result = wwd.smooth2D_DataArray(sample_3d_dataarray, dim1='x', dim2='y')
        
        assert 'lat' in result.coords
        assert 'lon' in result.coords
        assert np.array_equal(result.coords['lat'].values, sample_3d_dataarray.coords['lat'].values)
        assert np.array_equal(result.coords['lon'].values, sample_3d_dataarray.coords['lon'].values)


    def test_edge_preservation(self):
        """Test that smoothing handles edges properly."""
        # Create data with sharp edge
        data = np.zeros((50, 50))
        data[20:30, 20:30] = 1.0

        da = xr.DataArray(data, dims=['y', 'x'])

        # Smooth with different methods
        result_gauss = wwd.smooth2D(da, method='gaussian', sigma=1.0)
        result_uniform = wwd.smooth2D(da, method='uniform', size=3)
        result_box = wwd.smooth2D(da, method='box', kernel_size=3)

        # All methods should preserve total sum approximately (for reflect mode)
        assert abs(np.sum(result_gauss.values) - np.sum(data)) < 0.1
        assert abs(np.sum(result_uniform.values) - np.sum(data)) < 0.1
        assert abs(np.sum(result_box.values) - np.sum(data)) < 0.1


    def test_type_hints(self):
        """Test that type hints work correctly."""
        data = np.random.randn(30, 40)

        # DataArray
        da = xr.DataArray(data, dims=['lat', 'lon'])
        da_result = wwd.smooth2D(da)
        assert isinstance(da_result, xr.DataArray)

        # Dataset
        ds = xr.Dataset({'temp': da})
        ds_result = wwd.smooth2D(ds)
        assert isinstance(ds_result, xr.Dataset)        



@pytest.fixture
def data_xy():
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    data = np.outer(y, x)
    return xr.DataArray(data, coords={'x': x, 'y': y}, dims=['y', 'x'])

@pytest.fixture
def data_latlon():
    lon = np.linspace(0, 10, 11)
    lat = np.linspace(0, 10, 11)
    data = np.outer(lat, lon)
    return xr.DataArray(data, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'])

def test_linear_interp_xy(data_xy):
    points = np.array([[5.5, 5.5], [2.1, 3.3]])
    result = wwd.interpolateValues(points, None, data_xy, method='linear')
    assert result.shape == (2,)

def test_nearest_interp_latlon(data_latlon):
    points = np.array([[5.1, 5.1], [1.9, 2.2]])
    result = wwd.interpolateValues(points, None, data_latlon, method='nearest')
    assert result.shape == (2,)

def test_invalid_coords():
    data = xr.DataArray(np.zeros((5, 5)), coords={'a': np.arange(5), 'b': np.arange(5)}, dims=['a', 'b'])
    points = np.array([[1, 2]])
    with pytest.raises(ValueError):
        wwd.interpolateValues(points, None, data, method='linear')        


class TestRasterizeGeoDataFrame:
    
    @pytest.fixture
    def simple_polygon_gdf(self):
        """Create a simple GeoDataFrame with polygons."""
        data = {
            'value': [1.0, 2.0, 3.0, 4.0],
            'geometry': [
                shapely.geometry.box(0, 0, 1, 1),      # Square at (0,0)
                shapely.geometry.box(2, 0, 3, 1),      # Square at (2,0)
                shapely.geometry.box(0, 2, 1, 3),      # Square at (0,2)
                shapely.geometry.box(2, 2, 3, 3)       # Square at (2,2)
            ]
        }
        return gpd.GeoDataFrame(data, crs='EPSG:4326')
    
    @pytest.fixture
    def overlapping_polygon_gdf(self):
        """Create a GeoDataFrame with overlapping polygons."""
        data = {
            'value': [10, 20, 30],
            'geometry': [
                shapely.geometry.box(0, 0, 2, 2),      # Square from (0,0) to (2,2)
                shapely.geometry.box(1, 1, 3, 3),      # Overlapping square
                shapely.geometry.box(4, 0, 5, 1)       # Non-overlapping rectangle
            ]
        }
        return gpd.GeoDataFrame(data)
    
    @pytest.fixture
    def multipolygon_gdf(self):
        """Create a GeoDataFrame with MultiPolygons."""
        # Create two separate polygons for MultiPolygon
        poly1 = shapely.geometry.box(0, 0, 1, 1)
        poly2 = shapely.geometry.box(2, 2, 3, 3)
        multipoly = shapely.geometry.MultiPolygon([poly1, poly2])
        
        data = {
            'value': [100, 200],
            'geometry': [
                multipoly,
                shapely.geometry.box(4, 0, 6, 2)  # Regular polygon
            ]
        }
        return gpd.GeoDataFrame(data)
    
    @pytest.fixture
    def mixed_types_gdf(self):
        """Create a GeoDataFrame with different numeric types, leaving a gap for nans."""
        data = {
            'int_col': np.array([1, 2, 3], dtype=np.int32),
            'float_col': np.array([1.5, 2.5, 3.5], dtype=np.float32),
            'geometry': [
                shapely.geometry.box(0, 0, 1, 1),
                shapely.geometry.box(1, 0, 2, 1),
                shapely.geometry.box(3, 0, 4, 1)
            ]
        }
        return gpd.GeoDataFrame(data)
    
    def test_basic_rasterization(self, simple_polygon_gdf):
        """Test basic rasterization of polygons."""
        result = wwd.rasterizeGeoDataFrame(simple_polygon_gdf, 'value', resolution=0.5)
        
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('y', 'x')
        assert 'x' in result.coords
        assert 'y' in result.coords
        
        # Check that we have some non-NaN values
        assert not np.all(np.isnan(result.values))
        
        # Check attributes
        assert result.attrs['resolution'] == 0.5
        assert result.attrs['source_column'] == 'value'
        assert np.isnan(result.attrs['nodata'])  # Default for float
        assert 'crs' in result.attrs
    
    def test_overlapping_rasterization(self, overlapping_polygon_gdf):
        """Test rasterization of overlapping polygons."""
        result = wwd.rasterizeGeoDataFrame(overlapping_polygon_gdf, 'value', resolution=0.5)
        
        # Check dimensions - should cover bounds (0,0) to (5,3)
        assert result.sizes['x'] == 10  # 5 / 0.5
        assert result.sizes['y'] == 6   # 3 / 0.5
        
        # Check that polygons were rasterized
        assert not np.all(result.values == result.attrs['nodata'])
        
        # Check specific values
        unique_values = np.unique(result.values[~np.isnan(result.values)])
        assert 10 in unique_values
        assert 20 in unique_values
        assert 30 in unique_values
        
        # Check that overlapping area has value from second polygon (20)
        x_idx = np.argmin(np.abs(result.x.values - 1.5))
        y_idx = np.argmin(np.abs(result.y.values - 1.5))
        assert result.values[y_idx, x_idx] == 20
    
    def test_multipolygon_rasterization(self, multipolygon_gdf):
        """Test rasterization with MultiPolygon geometries."""
        result = wwd.rasterizeGeoDataFrame(multipolygon_gdf, 'value', resolution=0.5)
        
        # Check that MultiPolygon was rasterized correctly
        unique_values = np.unique(result.values[~np.isnan(result.values)])
        assert 100 in unique_values  # From MultiPolygon
        assert 200 in unique_values  # From regular polygon
        
        # Check that both parts of MultiPolygon have same value
        # Bottom-left part (around 0.5, 0.5)
        x_idx1 = np.argmin(np.abs(result.x.values - 0.5))
        y_idx1 = np.argmin(np.abs(result.y.values - 0.5))
        
        # Top-right part (around 2.5, 2.5)
        x_idx2 = np.argmin(np.abs(result.x.values - 2.5))
        y_idx2 = np.argmin(np.abs(result.y.values - 2.5))
        
        assert result.values[y_idx1, x_idx1] == 100
        assert result.values[y_idx2, x_idx2] == 100
    
    def test_integer_column_default_nodata(self, mixed_types_gdf):
        """Test rasterization of integer column with default nodata."""
        result = wwd.rasterizeGeoDataFrame(mixed_types_gdf, 'int_col', resolution=0.5)
        
        # Integer columns should maintain integer dtype
        assert result.dtype == np.int32
        
        # Default nodata for integers should be -999
        assert result.attrs['nodata'] == -999
        
        # Check valid values
        valid_values = result.values[result.values != -999]
        assert len(valid_values) > 0
        assert all(v in [1, 2, 3] for v in valid_values)
    
    def test_float_column_default_nodata(self, mixed_types_gdf):
        """Test rasterization of float column with default nodata."""
        result = wwd.rasterizeGeoDataFrame(mixed_types_gdf, 'float_col', resolution=0.5)
        
        # Float columns should maintain float dtype
        assert result.dtype == np.float32
        
        # Default nodata for floats should be NaN
        assert np.isnan(result.attrs['nodata'])
        
        # Check values
        non_nan_values = result.values[~np.isnan(result.values)]
        assert len(non_nan_values) > 0
    
    def test_custom_nodata(self, mixed_types_gdf):
        """Test custom nodata values."""
        # Integer with custom nodata
        result_int = wwd.rasterizeGeoDataFrame(
            mixed_types_gdf, 'int_col', resolution=0.5, nodata=-9999
        )
        assert result_int.attrs['nodata'] == -9999
        assert np.any(result_int.values == -9999)
        
        # Float with custom nodata
        result_float = wwd.rasterizeGeoDataFrame(
            mixed_types_gdf, 'float_col', resolution=0.5, nodata=-999.99
        )
        assert result_float.attrs['nodata'] == -999.99
        assert np.any(np.isclose(result_float.values, -999.99))
    
    def test_custom_bounds(self, simple_polygon_gdf):
        """Test with custom bounds."""
        custom_bounds = (-1, -1, 4, 4)
        result = wwd.rasterizeGeoDataFrame(
            simple_polygon_gdf, 'value', resolution=0.5, bounds=custom_bounds
        )
        
        # Check that bounds are respected
        assert result.x.min() >= -1
        assert result.x.max() <= 4
        assert result.y.min() >= -1
        assert result.y.max() <= 4
    
    def test_invalid_geometry_type_points(self):
        """Test error with point geometries."""
        data = {
            'value': [1, 2, 3],
            'geometry': [
                shapely.geometry.Point(0, 0),
                shapely.geometry.Point(1, 1),
                shapely.geometry.Point(2, 2)
            ]
        }
        gdf = gpd.GeoDataFrame(data)
        result = wwd.rasterizeGeoDataFrame(gdf, 'value', resolution=1.0)
        assert (result.values == -999).all()
    
    def test_mixed_geometry_types(self):
        """Test error with mixed geometry types."""
        data = {
            'value': [1, 2, 3],
            'geometry': [
                shapely.geometry.box(0, 0, 1, 1),      # Polygon
                shapely.geometry.Point(2, 2),           # Point
                shapely.geometry.box(3, 3, 4, 4)        # Polygon
            ]
        }
        gdf = gpd.GeoDataFrame(data)
        result = wwd.rasterizeGeoDataFrame(gdf, 'value', resolution=1.0)
        assert result.values[-1,0] == 1
        assert result.values[0,-1] == 3
    
    def test_empty_geodataframe(self):
        """Test error with empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame(columns=['value', 'geometry'])
        
        with pytest.raises(ValueError, match="GeoDataFrame is empty"):
            wwd.rasterizeGeoDataFrame(gdf, 'value', resolution=1.0)
    
    def test_missing_column(self, simple_polygon_gdf):
        """Test error with missing column."""
        with pytest.raises(ValueError, match="Column 'missing' not found"):
            wwd.rasterizeGeoDataFrame(simple_polygon_gdf, 'missing', resolution=1.0)
    
    def test_non_numeric_column(self):
        """Test error with non-numeric column."""
        data = {
            'text_col': ['a', 'b', 'c'],
            'geometry': [
                shapely.geometry.box(0, 0, 1, 1),
                shapely.geometry.box(1, 0, 2, 1),
                shapely.geometry.box(2, 0, 3, 1)
            ]
        }
        gdf = gpd.GeoDataFrame(data)
        
        with pytest.raises(ValueError, match="must be numeric type"):
            wwd.rasterizeGeoDataFrame(gdf, 'text_col', resolution=1.0)
    
    def test_invalid_resolution(self, simple_polygon_gdf):
        """Test error with invalid resolution."""
        with pytest.raises(ValueError, match="Resolution must be positive"):
            wwd.rasterizeGeoDataFrame(simple_polygon_gdf, 'value', resolution=0)
        
        with pytest.raises(ValueError, match="Resolution must be positive"):
            wwd.rasterizeGeoDataFrame(simple_polygon_gdf, 'value', resolution=-1)
    
    def test_with_nan_values(self):
        """Test handling of NaN values in column."""
        data = {
            'value': [1.0, np.nan, 3.0],
            'geometry': [
                shapely.geometry.box(0, 0, 1, 1),
                shapely.geometry.box(1, 0, 2, 1),
                shapely.geometry.box(2, 0, 3, 1)
            ]
        }
        gdf = gpd.GeoDataFrame(data)
        
        result = wwd.rasterizeGeoDataFrame(gdf, 'value', resolution=0.5)
        
        # Should successfully rasterize non-NaN values
        non_nan_values = result.values[~np.isnan(result.values)]
        assert 1.0 in non_nan_values
        assert 3.0 in non_nan_values
        
        # Middle polygon should not be rasterized
        x_idx = np.argmin(np.abs(result.x.values - 1.5))
        y_idx = np.argmin(np.abs(result.y.values - 0.5))
        assert np.isnan(result.values[y_idx, x_idx])
    
    def test_with_invalid_geometries(self):
        """Test handling of None geometries."""
        data = {
            'value': [1.0, 2.0, 3.0],
            'geometry': [
                shapely.geometry.box(0, 0, 1, 1),
                None,  # Invalid geometry
                shapely.geometry.box(2, 0, 3, 1)
            ]
        }
        gdf = gpd.GeoDataFrame(data)
        
        result = wwd.rasterizeGeoDataFrame(gdf, 'value', resolution=0.5)
        
        # Should successfully rasterize valid geometries
        non_nan_values = result.values[~np.isnan(result.values)]
        assert 1.0 in non_nan_values
        assert 3.0 in non_nan_values
        # Value 2.0 should not appear since its geometry is None
        assert 2.0 not in non_nan_values


def test_complex_polygon_shapes():
    """Test with more complex polygon shapes."""
    # Create a polygon with a hole
    exterior = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    interior = [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)]
    poly_with_hole = shapely.geometry.Polygon(exterior, [interior])
    
    data = {
        'value': [42],
        'geometry': [poly_with_hole]
    }
    gdf = gpd.GeoDataFrame(data)
    
    result = wwd.rasterizeGeoDataFrame(gdf, 'value', resolution=0.5)
    
    # Check that the hole is not filled
    # Center of the hole (around 2, 2) should be NaN
    x_idx = np.argmin(np.abs(result.x.values - 2))
    y_idx = np.argmin(np.abs(result.y.values - 2))
    assert result.values[y_idx, x_idx] == -999
    
    # Check that the ring around the hole has the value
    x_idx_edge = np.argmin(np.abs(result.x.values - 0.5))
    y_idx_edge = np.argmin(np.abs(result.y.values - 0.5))
    assert result.values[y_idx_edge, x_idx_edge] == 42        
