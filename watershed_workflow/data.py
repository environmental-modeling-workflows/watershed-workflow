"""Functions used in interacting with data, either pandas DataFrames or xarray DataArrays."""
from typing import Union, List, Iterable, Tuple, Any, Optional, Literal, overload, Sequence
from xarray.core.types import InterpOptions
import numpy.typing as npt

import warnings
import cftime
import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rasterio.transform
import rasterio.features
import shapely.geometry
import scipy.signal
import scipy.stats
import scipy.ndimage
import scipy.interpolate

import watershed_workflow.crs
from watershed_workflow.crs import CRS

ValidTime = Union[cftime._cftime.datetime, pd.Timestamp, datetime.datetime, np.datetime64]

#
# Helper functions
#
def convertTimesToCFTime(time_values: Sequence[ValidTime]) -> npt.NDArray[cftime._cftime.datetime]:
    """Convert an iterable of datetime objects to cftime object.
    
    This function accepts various datetime types and converts them to
    cftime Gregorian calendar.
    
    Parameters
    ----------
    time_values
        Iterable of datetime objects (numpy datetime64, pandas Timestamp, 
        Python datetime, or cftime objects). All elements must be the same type.
        
    Returns
    -------
    List[cftime._cftime.datetime]
        List of cftime objects, likely in the Gregorian calendar.

    """
    # Handle empty input
    if len(time_values) == 0:
        return np.array([], dtype=cftime.DatetimeGregorian)

    # Get a sample time and conditional on type
    sample_time = time_values[0]
    if isinstance(sample_time, cftime._cftime.datetime):
        return np.array(time_values)

    if isinstance(sample_time, (np.datetime64, datetime.datetime)):
        time_values = pd.to_datetime(time_values).tolist()

    if not isinstance(time_values[0], pd.Timestamp):
        raise TypeError(f'Cannot convert items of type {type(time_values[0])} to cftime.')

    # convert pd.Timestamp to cftime
    res = [cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond) for t in time_values]
    return np.array(res)


def convertTimesToCFTimeNoleap(time_values: Sequence[cftime._cftime.datetime]) -> npt.NDArray[cftime.DatetimeNoLeap]:
    """Convert an iterable of cftime objects on any calendar to cftime DatetimeNoLeap calendar.
    
    This function accepts various datetime types and converts them to cftime NoLeap
    calendar while preserving the input container type. Raises an error if any
    input date represents the 366th day of a year (leap day in DayMet convention).
    
    Parameters
    ----------
    time_values
        Sequence of datetime objects (numpy datetime64, pandas Timestamp, 
        Python datetime, or cftime objects). All elements must be the same type.
        
    Returns
    -------
    Container of cftime.DatetimeNoLeap objects.
        
    Raises
    ------
    ValueError
        If any date in the input represents the 366th day of a year (leap day).
    """
    if len(time_values) == 0:
        return np.array([], dtype=cftime.DatetimeNoLeap)

    dayofyr = [t.dayofyr for t in time_values]

    if max(dayofyr) == 366:
        raise ValueError(f"Input contains leap day(s) (366th day of year)")

    return np.array([
        cftime.DatetimeNoLeap(t.year, 1, 1, t.hour, t.minute, t.second, t.microsecond)
        + datetime.timedelta(days=(t.dayofyr - 1)) for t in time_values
    ])


def createNoleapMask(time_values : Sequence[ValidTime]
                     ) -> Tuple[npt.NDArray[cftime.DatetimeNoLeap], npt.NDArray[bool]]:
    """Create a mask that is true for any non-leap-day (day 366).
    
    Parameters
    ----------
    time_values : Sequence[ValidTime]
        Time values to filter for leap days.
        
    Returns
    -------
    Sequence[cftime.DatetimeNoLeap]
        Time values converted to cftime format with leap days filtered.
    List[bool]
        Boolean mask where True indicates non-leap days.
    """
    # no times --> no leap days
    if len(time_values) == 0:
        return list(), list()

    # cftime.DataNoleap --> no leap days
    if isinstance(time_values[0], cftime.DatetimeNoLeap):
        return time_values, [True, ] * len(time_values)

    # Get the time column is in cftime format
    time_in_cftime = convertTimesToCFTime(time_values)
    mask = np.array([(t.dayofyr != 366) for t in time_in_cftime])
    return time_in_cftime[mask], mask


def _convertDataFrameToDataset(df: pd.DataFrame, time_column: str) -> xr.Dataset:
    """Convert DataFrame to Dataset for shared processing.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time series data.
    time_column : str
        Name of the time column.
        
    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the converted data.
        
    Raises
    ------
    ValueError
        If time_column not found or no numeric columns available.
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    # Sort by time
    df_sorted = df.sort_values(time_column).copy()

    # Get all numeric columns except time column
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
    columns_out = [col for col in numeric_cols if col != time_column]

    # Create Dataset from numeric columns
    data_vars = {}
    for col in columns_out:
        data_vars[col] = xr.DataArray(df_sorted[col].values,
                                      coords={ time_column: df_sorted[time_column].values },
                                      dims=[time_column])

    if len(data_vars) == 0:
        raise ValueError(f"No numeric columns provided or found.")

    return xr.Dataset(data_vars)


def _convertDatasetToDataFrame(ds: xr.Dataset, time_column: str, output_times: pd.Series) -> pd.DataFrame:
    """Convert Dataset back to DataFrame format.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset to convert.
    time_column : str
        Name of the time column to use.
    output_times : pandas.Series
        Time values to use for the output DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        Converted DataFrame with time column first, then other columns.
    """
    result_df = ds.to_dataframe().reset_index()
    result_df[time_column] = output_times
    return result_df


def computeMode(da: xr.DataArray, time_dim: str = 'time') -> xr.DataArray:
    """Compute the mode along the time dimension of a DataArray.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray. Can contain any data type that scipy.stats.mode can handle.
    time_dim : str, optional
        Name of the time dimension along which to compute the mode. Default is 'time'.
        
    Returns
    -------
    xr.DataArray
        DataArray with the mode computed along the time dimension. The time dimension
        is removed from the output. All other dimensions, coordinates, and attributes
        are preserved. In case of multiple modes, returns the smallest value.
        
    Raises
    ------
    ValueError
        If the specified time dimension is not found in the DataArray.
        
    Notes
    -----
    For continuous data, the mode may not be meaningful. This function is most
    useful for discrete or categorical data.
    
    When multiple values have the same highest frequency (multiple modes), 
    scipy.stats.mode returns the smallest of these values.
    
    NaN values are ignored in the mode calculation.

    """
    if time_dim not in da.dims:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    # Get the axis number for time dimension
    time_axis = da.dims.index(time_dim)

    if np.issubdtype(da.dtype, np.integer) or np.issubdtype(da.dtype, np.floating):
        # For numeric data, use scipy.stats.mode
        mode_result = scipy.stats.mode(da.values, axis=time_axis, nan_policy='omit', keepdims=False)
        mode_values = mode_result.mode
    else:
        raise ValueError('Non-numeric data modes are not implemented.')

    # Create coordinates for the result (all except time dimension)
    new_coords = { k: v for k, v in da.coords.items() if k != time_dim }

    # Create new dimensions list (all except time dimension)
    new_dims = [d for d in da.dims if d != time_dim]

    # Create result DataArray
    result = xr.DataArray(mode_values, dims=new_dims, coords=new_coords, attrs=da.attrs.copy())

    # transfer the crs
    try:
        crs = da.rio.crs
    except AttributeError:
        pass
    else:
        if crs is not None:
            result.rio.write_crs(crs, inplace=True)

    # Preserve the name if it exists
    if da.name is not None:
        result.name = f"{da.name}_mode"

    return result


#
# filter leap day
#
def filterLeapDay_DataFrame(df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
    """Remove day 366 (Dec 31) from leap years and convert time column to CFTime noleap calendar.
    
    Parameters
    ----------
    df
        Input DataFrame containing time series data.
    time_column
        Name of the column containing datetime data. Must be convertible to pandas datetime.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with day 366 of leap years removed and time column converted to 
        cftime noleap calendar format.
        
    Raises
    ------
    ValueError
        If `time_column` is not found in the DataFrame.
        If the time column cannot be converted to datetime format.
        
    Notes
    -----
    Day 366 only occurs on December 31st in leap years. The function assumes that
    the input time column is not already in cftime noleap format, as noleap 
    calendars by definition do not have day 366.
    
    The DataFrame index is reset after filtering to ensure continuous indexing.
    """
    # Validate inputs
    try:
        time_series = df[time_column]
    except KeyError:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    # no times --> no leap days
    if len(time_series) == 0:
        return df

    # cftime.DataNoleap --> no leap days
    if isinstance(time_series[0], cftime.DatetimeNoLeap):
        return df

    # Get the time column is in cftime format, and a mask that is True for any non-leap-days
    try:
        time_series_cftime, mask = createNoleapMask(time_series)
    except Exception as e:
        raise ValueError(f"Could not convert column '{time_column}' to cftime: {e}")

    # Apply the filter
    df_filtered = df[mask].reset_index(drop=True)

    # Convert the time column to CFTime noleap calendar
    df_filtered[time_column] = convertTimesToCFTimeNoleap(time_series_cftime)

    return df_filtered


def filterLeapDay_xarray(da: xr.DataArray | xr.Dataset,
                         time_dim: str = 'time') -> xr.DataArray | xr.Dataset:
    """Remove day 366 (Dec 31) from leap years and convert time dimension to CFTime noleap calendar.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with a time dimension. The time dimension must contain
        datetime-like values that can be converted to pandas datetime.
        
    Returns
    -------
    xr.DataArray
        DataArray with day 366 of leap years removed and time dimension converted to 
        cftime noleap calendar format. All attributes, including rasterio-specific
        attributes like 'nodata' and 'crs', are preserved.
        
    Raises
    ------
    ValueError
        If the DataArray does not have a 'time' dimension.
        If the time dimension cannot be converted to datetime format.
        
    Notes
    -----
    Day 366 only occurs on December 31st in leap years. The function assumes that
    the input time dimension is not already in cftime noleap format, as noleap 
    calendars by definition do not have day 366.
    
    For rasterio-based DataArrays, this function preserves the coordinate reference
    system (CRS) and nodata value attributes. All other attributes are also preserved.
    
    The time dimension name is preserved, but its values are replaced with cftime
    noleap calendar objects.
    """
    # deal with missing time_dim
    try:
        time_series = da[time_dim]
    except KeyError:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    # deal with empty time_dim
    if len(time_series) == 0:
        return da

    # Create mask for values to keep (exclude day 366 in leap years)
    time_array_cftime, mask = createNoleapMask(time_series.values)

    # Apply the filter to the DataArray
    da_filtered = da.isel({ time_dim: mask })

    # Convert the filtered time values to cftime noleap
    time_array_noleap = convertTimesToCFTimeNoleap(time_array_cftime)

    # Replace the time coordinate with cftime noleap
    da_filtered = da_filtered.assign_coords({ time_dim: time_array_noleap })

    # Preserve all attributes from the original DataArray
    try:
        crs = da.rio.crs
    except AttributeError:
        pass
    else:
        if crs is not None:
            da_filtered.rio.write_crs(crs, inplace=True)
    da_filtered.attrs = da.attrs.copy()

    # Preserve coordinate attributes (including CRS if present)
    for coord in da_filtered.coords:
        if coord in da.coords and hasattr(da.coords[coord], 'attrs'):
            da_filtered.coords[coord].attrs = da.coords[coord].attrs.copy()

    # Preserve attributes for all variables in a Dataset
    if hasattr(da_filtered, 'data_vars'):
        for var in da_filtered.data_vars:
            if var in da.data_vars and hasattr(da[var], 'attrs'):
                da_filtered[var].attrs = da[var].attrs.copy()

    return da_filtered


@overload
def filterLeapDay(data: xr.Dataset, time_column: str = 'time') -> xr.Dataset:
    ...


@overload
def filterLeapDay(data: xr.DataArray, time_column: str = 'time') -> xr.DataArray:
    ...


@overload
def filterLeapDay(data: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
    ...


def filterLeapDay(data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
                  time_column: str = 'time') -> Union[pd.DataFrame, xr.DataArray, xr.Dataset]:
    """Remove day 366 (Dec 31) from leap years and convert time to CFTime noleap calendar.
    
    This function automatically selects the appropriate implementation based on the
    input data type: DataFrame, DataArray, or Dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame, xr.DataArray, or xr.Dataset
        Input data containing time series information to filter.
    time_column : str, optional
        For DataFrame: Name of the column containing datetime data (required).
        For Dataset: Name of the time dimension (defaults to 'time').
        For DataArray: Ignored (always uses 'time' dimension).
        
    Returns
    -------
    pandas.DataFrame, xr.DataArray, or xr.Dataset
        Same type as input with day 366 of leap years removed and time 
        converted to cftime noleap calendar format.
        
    Raises
    ------
    TypeError
        If data is not a DataFrame, DataArray, or Dataset.
        If DataFrame is provided without time_column.
    ValueError
        If time column/dimension is not found or cannot be converted to datetime.
        
    Notes
    -----
    Day 366 only occurs on December 31st in leap years. The function assumes that
    the input time data is not already in cftime noleap format, as noleap 
    calendars by definition do not have day 366.
    
    For DataFrames:
        - time_column parameter is required
        - DataFrame index is reset after filtering
        
    For DataArrays & Datasets:
        - time_column parameter specifies the time dimension (default: 'time')
        - All attributes including rasterio-specific ones are preserved
        
    See Also
    --------
    filterLeapDay_DataFrame : DataFrame-specific implementation
    filterLeapDay_DataArray : DataArray-specific implementation  
    filterLeapDay_Dataset : Dataset-specific implementation
    """
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        # Dataset can use custom time dimension name
        return filterLeapDay_xarray(data, time_dim=time_column)

    elif isinstance(data, pd.DataFrame):
        if time_column is None:
            raise TypeError("time_column parameter is required for DataFrame input")
        return filterLeapDay_DataFrame(data, time_column)

    else:
        raise TypeError(f"Input data must be a pandas DataFrame, xr DataArray, or xr Dataset. "
                        f"Got {type(data).__name__}")


def interpolate_Dataset(ds: xr.Dataset | xr.DataArray,
                        time_values: Sequence[ValidTime],
                        time_dim: str = 'time',
                        method: InterpOptions = 'linear') -> xr.Dataset:
    """Interpolate Dataset to arbitrary times.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with a time dimension containing cftime objects.
    time_values : Sequence[ValidTime]
        Time values to interpolate to.
    time_dim : str
        Name of the time dimension. Default is 'time'.
    method : str, optional
        Interpolation method. Default is 'linear'.
        
    Returns
    -------
    xr.Dataset
        Dataset with regular time intervals and interpolated values.
        
    Raises
    ------
    ValueError
        If time dimension is not found or interval is not 1 or 5.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    result = ds.interp({ time_dim: time_values },
                       method=method,
                       kwargs={ 'fill_value': 'extrapolate'} if method in ['linear', 'nearest'] else {})
    
    # Preserve Dataset attributes
    result.attrs = ds.attrs.copy()
    return result


def interpolate_DataFrame(df: pd.DataFrame,
                          time_values: Sequence[ValidTime],
                          time_column: str = 'time',
                          method: InterpOptions = 'linear') -> pd.DataFrame:
    """Interpolate DataFrame to arbitrary times.

    NOTE: this is not the same as pandas.interpolate(), but more like
    pandas.reindex(time_values).interpolate() with scipy-based
    interpolation options.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a time column containing cftime objects.
    time_values : Sequence[ValidTime]
        Time values to interpolate to.
    time_column : str
        Name of the column containing cftime datetime objects.
    method : str, optional
        Interpolation method. Default is 'linear'.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with regular time intervals and interpolated values.
        
    Raises
    ------
    ValueError
        If time_column is not found or interval is not 1 or 5.

    """
    # get a dataset
    ds = _convertDataFrameToDataset(df, time_column)

    # interpolate using dataset function
    interp_ds = interpolate_Dataset(ds, time_values, time_column, method)

    # Convert back to DataFrame
    return _convertDatasetToDataFrame(interp_ds, time_column, interp_ds[time_column])



@overload
def interpolate(data: xr.Dataset,
                time_values: Sequence[ValidTime] = ...,
                time_dim: str = ...,
                method: InterpOptions = ...,
                ) -> xr.Dataset:
    ...


@overload
def interpolate(data: xr.DataArray,
                time_values: Sequence[ValidTime] = ...,
                time_dim: str = ...,
                method: InterpOptions = ...,
                ) -> xr.DataArray:
    ...


@overload
def interpolate(data: pd.DataFrame,
                time_values: Sequence[ValidTime] = ...,
                time_dim: str = ...,
                method: InterpOptions = ...,
                ) -> pd.DataFrame:
    ...


def interpolate(data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
                time_values: Sequence[ValidTime] = ...,
                time_dim: str = 'time',
                method: InterpOptions = 'linear'
                ) -> Union[pd.DataFrame, xr.DataArray, xr.Dataset]:
    """
    Interpolate data to new times.
    
    This function automatically selects the appropriate implementation based on the
    input data type: DataFrame, DataArray, or Dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame, xr.DataArray, or xr.Dataset
        Input data containing time series with cftime calendar.
    time_values : Sequence[ValidTime]
        Time values to interpolate to.
    time_dim : str, optional
        For DataFrame: Name of the time column (required).
        For Dataset/DataArray: Name of the time dimension (default: 'time').
    method : str, optional
        Interpolation method. Default is 'linear'.
        
    Returns
    -------
    pandas.DataFrame, xr.DataArray, or xr.Dataset
        Same type as input with regular time intervals and interpolated values.
        
    Raises
    ------
    TypeError
        If data is not a DataFrame, DataArray, or Dataset.
    ValueError
        If required parameters are missing or invalid.
        
    See Also
    --------
    interpolate_DataFrame : DataFrame-specific implementation
    interpolate_Dataset : Dataset-specific implementation
    """
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        return interpolate_Dataset(data, time_values, time_dim, method)
    elif isinstance(data, pd.DataFrame):
        return interpolate_DataFrame(data, time_values, time_dim, method)
    else:
        raise TypeError(f"Input data must be a pandas DataFrame, xr DataArray, or xr Dataset. "
                        f"Got {type(data).__name__}")


def interpolateToRegular(data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
                         interval : int = 1,
                         time_dim: str = 'time',
                         method: InterpOptions = 'linear'
                         ) -> Union[pd.DataFrame, xr.DataArray, xr.Dataset]:
    """
    Interpolate data to new times.
    
    This function automatically selects the appropriate implementation based on the
    input data type: DataFrame, DataArray, or Dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame, xr.DataArray, or xr.Dataset
        Input data containing time series with cftime calendar.
    time_values : Sequence[ValidTime]
        Time values to interpolate to.
    time_dim : str, optional
        For DataFrame: Name of the time column (required).
        For Dataset/DataArray: Name of the time dimension (default: 'time').
    method : str, optional
        Interpolation method. Default is 'linear'.
        
    Returns
    -------
    pandas.DataFrame, xr.DataArray, or xr.Dataset
        Same type as input with regular time intervals and interpolated values.
        
    Raises
    ------
    TypeError
        If data is not a DataFrame, DataArray, or Dataset.
    ValueError
        If required parameters are missing or invalid.
        
    See Also
    --------
    interpolate_DataFrame : DataFrame-specific implementation
    interpolate_Dataset : Dataset-specific implementation
    """
    start_year = data[time_dim].values[0].year
    end_year = data[time_dim].values[-1].year

    calendar = None
    if isinstance(data[time_dim].values[0], cftime.DatetimeNoLeap):
        calendar = 'noleap'
        
    new_times = xr.date_range(data[time_dim].values[0], data[time_dim].values[-1], freq=f'{interval}D', calendar=calendar)
    return interpolate(data, new_times, time_dim, method)
    

#
# Compute an annual average
#
def _computeAverageYear(ds: xr.Dataset, time_dim: str, start_date: cftime.datetime,
                        output_nyears: int) -> Tuple[xr.Dataset, List[cftime.datetime]]:
    """
    Compute annual average for a Dataset and generate output times.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with cftime noleap calendar dates.
    time_dim : str
        Name of the time dimension.
    start_date : cftime.datetime
        Start date for the output time series.
    output_nyears : int
        Number of years to repeat the averaged pattern.
        
    Returns
    -------
    averaged_ds : xr.Dataset
        Dataset with averaged values indexed by day of year.
    output_times : list of cftime.datetime
        List of output times for the repeated pattern.
    """
    # Calculate day of year for each time point
    time_values = ds[time_dim].values
    doys = np.array([(date - cftime.DatetimeNoLeap(date.year, 1, 1)).days + 1
                     for date in time_values])

    # Get unique days of year
    unique_doys = np.unique(doys)

    # Add day of year coordinate temporarily
    ds_with_doy = ds.assign_coords(day_of_year=(time_dim, doys))

    # Group by day of year and compute mean
    averaged_ds = ds_with_doy.groupby('day_of_year').mean(dim=time_dim)

    # Create output times
    output_times = []
    output_doys = []

    for year_offset in range(output_nyears):
        current_year = start_date.year + year_offset

        for doy in unique_doys:
            date = cftime.DatetimeNoLeap(current_year, 1, 1) + datetime.timedelta(days=int(doy - 1))
            output_times.append(date)
            output_doys.append(doy)

    # Create mapping from output indices to day of year
    output_doys_array = xr.DataArray(output_doys, dims=['new_time'])

    # Create result by selecting appropriate days
    result_ds = averaged_ds.sel(day_of_year=output_doys_array)
    result_ds = result_ds.rename({ 'new_time': time_dim })

    # Drop day_of_year coordinate
    if 'day_of_year' in result_ds.coords:
        result_ds = result_ds.drop_vars('day_of_year')

    return result_ds, output_times


def _parseStartDate(
        start_date: Union[str, int, datetime.datetime, cftime.datetime]) -> cftime.datetime:
    """
    Parse start_date into cftime.DatetimeNoLeap.
    
    Parameters
    ----------
    start_date : str, datetime, or cftime.datetime
        Start date in various formats.
        
    Returns
    -------
    cftime.DatetimeNoLeap
        Parsed start date.
    """
    if isinstance(start_date, int):
        start_date = str(start_date)

    if isinstance(start_date, str):
        parts = start_date.split('-')
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return cftime.DatetimeNoLeap(year, month, day)

    elif isinstance(start_date, datetime.datetime):
        return cftime.DatetimeNoLeap(start_date.year, start_date.month, start_date.day)
    else:
        return start_date


def computeAverageYear_DataFrame(df: pd.DataFrame,
                                 time_column: str = 'time',
                                 start_date: Union[str, datetime.datetime,
                                                   cftime.datetime] = '2020-1-1',
                                 output_nyears: int = 2) -> pd.DataFrame:
    """
    Average DataFrame values across years and repeat for specified number of years.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with cftime noleap calendar dates at 1- or 5-day intervals.
    time_column : str
        Name of the column containing cftime datetime objects.
    start_date : str, datetime, or cftime.datetime
        Start date for the output time series. If string, should be 'YYYY-MM-DD' format.
    output_nyears : int
        Number of years to repeat the averaged pattern.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with averaged values repeated for the specified number of years,
        starting from start_date. Only includes the time column and averaged numeric columns.
        
    Raises
    ------
    ValueError
        If time_column is not found or contains invalid data.
        
    Notes
    -----
    The function computes the average value for each day of year (1-365) across all
    years in the input data. For 5-day intervals, it averages values at days 1, 6,
    11, etc. The resulting pattern is then repeated for output_nyears starting from
    start_date.
    
    Missing values (NaN) are ignored in the averaging process.
    Non-numeric columns are excluded from the output.
    """
    # Parse start date
    start_cftime = _parseStartDate(start_date)

    # get a dataset
    ds = _convertDataFrameToDataset(df, time_column)

    # Compute averages using shared function
    averaged_ds, output_times = _computeAverageYear(ds, time_column, start_cftime, output_nyears)

    # Convert back to DataFrame
    return _convertDatasetToDataFrame(averaged_ds, time_column, output_times)


def computeAverageYear_DataArray(da: xr.DataArray,
                                 time_dim: str = 'time',
                                 start_date: Union[str, datetime.datetime,
                                                   cftime.datetime] = '2020-1-1',
                                 output_nyears: int = 2,
                                 ) -> xr.DataArray:
    """
    Average DataArray values across years and repeat for specified number of years.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with cftime noleap calendar dates at 1- or 5-day intervals.
    start_date : str, datetime, or cftime.datetime
        Start date for the output time series. If string, should be 'YYYY-MM-DD' format.
    output_nyears : int
        Number of years to repeat the averaged pattern.
    time_dim : str, optional
        Name of the time dimension. Default is 'time'.
        
    Returns
    -------
    xr.DataArray
        DataArray with averaged values repeated for the specified number of years,
        starting from start_date. All attributes are preserved.
        
    Raises
    ------
    ValueError
        If time dimension is not found.
        
    Notes
    -----
    The function computes the average value for each day of year (1-365) across all
    years in the input data. The resulting 365-day pattern is then repeated for
    output_nyears starting from start_date.
    
    This is particularly useful for creating climatological datasets or for
    generating synthetic time series based on historical patterns.
    """
    if time_dim not in da.dims:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    # Parse start date
    start_cftime = _parseStartDate(start_date)

    # Convert to Dataset for processing
    temp_ds = xr.Dataset({da.name or 'data': da})

    # Compute averages using shared function
    averaged_ds, output_times = _computeAverageYear(temp_ds, time_dim, start_cftime, output_nyears)

    # Extract the DataArray
    result = averaged_ds[da.name or 'data']

    # Assign the output times
    result = result.assign_coords({ time_dim: output_times })

    # Preserve attributes
    result.attrs = da.attrs.copy()
    result.name = da.name

    return result


def computeAverageYear_Dataset(ds: xr.Dataset,
                               time_dim: str = 'time',
                               start_date: Union[str, datetime.datetime,
                                                 cftime.datetime] = '2020-1-1',
                               output_nyears: int = 2,
                               variables: Optional[List[str]] = None) -> xr.Dataset:
    """
    Average Dataset values across years and repeat for specified number of years.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with cftime noleap calendar dates at 1- or 5-day intervals.
    start_date : str, datetime, or cftime.datetime
        Start date for the output time series. If string, should be 'YYYY-MM-DD' format.
    output_nyears : int
        Number of years to repeat the averaged pattern.
    time_dim : str, optional
        Name of the time dimension. Default is 'time'.
    variables : list of str, optional
        List of variables to average. If None, averages all variables with the
        time dimension.
        
    Returns
    -------
    xr.Dataset
        Dataset with averaged values repeated for the specified number of years,
        starting from start_date. All attributes are preserved.
        
    Raises
    ------
    ValueError
        If time dimension is not found or if specified variables don't exist.
        
    Notes
    -----
    Variables without the time dimension are preserved unchanged in the output.
    
    For each variable with the time dimension, the function computes the average
    value for each day of year across all years in the input data.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    # Parse start date
    start_cftime = _parseStartDate(start_date)

    # Select variables to process
    if variables is None:
        vars_with_time = [var for var in ds.data_vars if time_dim in ds[var].dims]
        vars_without_time = [var for var in ds.data_vars if time_dim not in ds[var].dims]
    else:
        # Validate variables
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise ValueError(f"Variables not found in Dataset: {missing}")
        vars_with_time = [v for v in variables if time_dim in ds[v].dims]
        vars_without_time = [var for var in ds.data_vars if var not in variables]

    # Create Dataset with only time-dependent variables
    ds_to_avg = ds[vars_with_time] if vars_with_time else xr.Dataset()

    if vars_with_time:
        # Compute averages using shared function
        averaged_ds, output_times = _computeAverageYear(ds_to_avg, time_dim, start_cftime,
                                                        output_nyears)

        # Assign the output times
        result = averaged_ds.assign_coords({ time_dim: output_times })
    else:
        # No time-dependent variables to average
        result = xr.Dataset()

    # Add variables without time dimension
    for var in vars_without_time:
        result[var] = ds[var]

    # Copy over other coordinates
    for coord in ds.coords:
        if coord != time_dim and coord not in result.coords:
            result.coords[coord] = ds.coords[coord]

    # Preserve attributes
    result.attrs = ds.attrs.copy()
    for var in result.data_vars:
        if var in ds.data_vars:
            result[var].attrs = ds[var].attrs.copy()

    return result


@overload
def computeAverageYear(data: xr.Dataset,
                       time_column: str,
                       start_year: int,
                       output_nyears: int) -> xr.Dataset:
    ...


@overload
def computeAverageYear(data: xr.DataArray,
                       time_column: str,
                       start_year: int,
                       output_nyears: int) -> xr.DataArray:
    ...


@overload
def computeAverageYear(data: pd.DataFrame,
                       time_column: str,
                       start_year: int,
                       output_nyears: int) -> pd.DataFrame:
    ...


def computeAverageYear(
        data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
        time_column: str,
        start_year: int,
        output_nyears: int) -> Union[pd.DataFrame, xr.DataArray, xr.Dataset]:
    """
    Average data values across years and repeat for specified number of years.
    
    This function automatically selects the appropriate implementation based on the
    input data type: DataFrame, DataArray, or Dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame, xr.DataArray, or xr.Dataset
        Input data with cftime noleap calendar dates at 1- or 5-day intervals.
    time_column : str, optional
        For DataFrame: Name of the time column (required).
        For Dataset: Name of the time dimension (default: 'time').
        For DataArray: Ignored (always uses 'time' dimension).
    start_year : int
        Start year for the output time series.
    output_nyears : int, optional
        Number of years to repeat the averaged pattern. Default is 1.
        
    Returns
    -------
    pandas.DataFrame, xr.DataArray, or xr.Dataset
        Same type as input with averaged values repeated for the specified number
        of years, starting from start_date.
        
    Raises
    ------
    TypeError
        If data is not a DataFrame, DataArray, or Dataset.
        If DataFrame is provided without time_column.
    ValueError
        If time column/dimension is not found or contains invalid data.
        
    Notes
    -----
    The function computes the average value for each day of year (1-365) across all
    years in the input data. For 5-day intervals, it averages values at days 1, 6,
    11, etc. The resulting pattern is then repeated for output_nyears starting from
    start_date.
    
    Missing values (NaN) are ignored in the averaging process.
    
    For DataFrames, only numeric columns are included in the output
        
    For DataArrays all attributes are preserved
        
    For Datasets
        - Variables without the time dimension are preserved unchanged
        - All attributes and encodings are preserved
    """
    start_date = cftime.DatetimeNoLeap(start_year, 1, 1)
    if isinstance(data, pd.DataFrame):
        if time_column is None:
            raise TypeError("time_column parameter is required for DataFrame input")
        return computeAverageYear_DataFrame(data, time_column, start_date, output_nyears)

    elif isinstance(data, xr.DataArray):
        return computeAverageYear_DataArray(data, time_column, start_date, output_nyears)

    elif isinstance(data, xr.Dataset):
        # Dataset can use custom time dimension name
        time_dim = time_column or time_column
        return computeAverageYear_Dataset(data, time_dim, start_date, output_nyears)

    else:
        raise TypeError(f"Input data must be a pandas DataFrame, xr DataArray, or xr Dataset. "
                        f"Got {type(data).__name__}")


#
# Smooth data temporally
#
def smoothTimeSeries_Array(data: np.ndarray,
                           method: Literal['savgol', 'rolling_mean'] = 'savgol',
                           axis: int = -1,
                           **kwargs) -> np.ndarray:
    """
    Smooth time series data using specified method.
    
    Parameters
    ----------
    data : numpy.ndarray
        Array of data to smooth. Must not contain NaN values.
    method : {'savgol', 'rolling_mean'}
        Smoothing method to use.
    axis : int, optional
        Axis along which to smooth. Default is -1 (last axis).
    **kwargs : dict
        Method-specific parameters.
        
        For 'savgol':
            - window_length : int, odd number (default: 7)
            - polyorder : int (default: 3)
            - mode : str (default: 'interp')
            
        For 'rolling_mean':
            - window : int (default: 5)
            - center : bool (default: True)
        
    Returns
    -------
    numpy.ndarray
        Smoothed data array.
        
    Raises
    ------
    ValueError
        If method is not recognized, parameters are invalid, or data contains NaN.
    """
    # Check for NaN values
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    if method == 'savgol':
        # Extract savgol parameters
        window_length = kwargs.get('window_length', 7)
        polyorder = kwargs.get('polyorder', 3)
        mode = kwargs.get('mode', 'interp')

        # Validate parameters
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd for Savitzky-Golay filter")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")
        if data.shape[axis] < window_length:
            raise ValueError(
                f"Data length along axis {axis} ({data.shape[axis]}) must be >= window_length ({window_length})"
            )

        # Apply Savitzky-Golay filter along specified axis
        return scipy.signal.savgol_filter(data, window_length, polyorder, axis=axis, mode=mode)

    elif method == 'rolling_mean':
        # Extract rolling mean parameters
        window = kwargs.get('window', 5)
        center = kwargs.get('center', True)

        # Calculate origin for centering
        if center:
            origin = 0
        else:
            origin = -(window // 2)

        # Apply uniform filter (equivalent to rolling mean)
        return scipy.ndimage.uniform_filter1d(data,
                                              size=window,
                                              axis=axis,
                                              mode='reflect',
                                              origin=origin)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def smoothTimeSeries_DataFrame(df: pd.DataFrame,
                               time_column: str = 'time',
                               method: Literal['savgol', 'rolling_mean'] = 'savgol',
                               **kwargs) -> pd.DataFrame:
    """
    Smooth time series data in a DataFrame along the time dimension.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with time series data. Must not contain NaN values
        in columns to be smoothed.
    time_column : str
        Name of the time column.
    method : {'savgol', 'rolling_mean'}, optional
        Smoothing method. Default is 'savgol'.
    **kwargs : dict
        Method-specific parameters passed to smoothing function.
        
        For 'savgol':
            - window_length : int, odd number (default: 7)
            - polyorder : int (default: 3)
            - mode : {'mirror', 'constant', 'nearest', 'wrap', 'interp'} (default: 'interp')
            
        For 'rolling_mean':
            - window : int (default: 5)
            - center : bool (default: True)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with smoothed data.
        
    Raises
    ------
    ValueError
        If any column to be smoothed contains NaN values.
        
    Notes
    -----
    The Savitzky-Golay filter is useful for smoothing noisy data while preserving
    important features like peaks. The rolling mean provides simple moving average
    smoothing.
    
    Data is sorted by time before smoothing to ensure correct temporal ordering.
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    # Make a copy and sort by time
    df_sorted = df.sort_values(time_column).copy()

    # Identify columns to smooth
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_smooth = [col for col in numeric_cols if col != time_column]

    # Apply smoothing to each column
    result = df_sorted.copy()
    for col in columns_to_smooth:
        data = df_sorted[col].values
        result[col] = smoothTimeSeries_Array(data, method, axis=0, **kwargs)

    return result


def smoothTimeSeries_DataArray(da: xr.DataArray,
                               time_dim: str = 'time',
                               method: Literal['savgol', 'rolling_mean'] = 'savgol',
                               **kwargs) -> xr.DataArray:
    """
    Smooth time series data in a DataArray along the time dimension.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with time series data. Must not contain NaN values.
    time_dim : str, optional
        Name of the time dimension. Default is 'time'.
    method : {'savgol', 'rolling_mean'}, optional
        Smoothing method. Default is 'savgol'.
    **kwargs : dict
        Method-specific parameters passed to smoothing function.
        
    Returns
    -------
    xr.DataArray
        DataArray with smoothed data. All attributes and coordinates are preserved.
        
    Raises
    ------
    ValueError
        If time dimension is not found or data contains NaN values.
        
    Notes
    -----
    For multidimensional arrays, smoothing is applied along the time dimension
    for each combination of other dimensions (e.g., each spatial point).
    """
    if time_dim not in da.dims:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    # Check for NaN values
    if np.any(np.isnan(da.values)):
        raise ValueError("DataArray contains NaN values")

    # Get the axis number for time dimension
    time_axis = da.dims.index(time_dim)

    # Apply smoothing along time dimension in one pass
    smoothed_data = smoothTimeSeries_Array(da.values, method, axis=time_axis, **kwargs)

    # Create result DataArray
    result = da.copy()
    result.values = smoothed_data

    return result


def smoothTimeSeries_Dataset(ds: xr.Dataset,
                             time_dim: str = 'time',
                             method: Literal['savgol', 'rolling_mean'] = 'savgol',
                             variables: Optional[List[str]] = None,
                             **kwargs) -> xr.Dataset:
    """
    Smooth time series data in a Dataset along the time dimension.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with time series data. Variables to be smoothed must
        not contain NaN values.
    time_dim : str, optional
        Name of the time dimension. Default is 'time'.
    method : {'savgol', 'rolling_mean'}, optional
        Smoothing method. Default is 'savgol'.
    variables : list of str, optional
        Variables to smooth. If None, smooths all variables with the time dimension.
    **kwargs : dict
        Method-specific parameters passed to smoothing function.
        
    Returns
    -------
    xr.Dataset
        Dataset with smoothed data. Variables without the time dimension are
        preserved unchanged. All attributes are preserved.
        
    Raises
    ------
    ValueError
        If time dimension is not found, specified variables don't exist,
        or any variable to be smoothed contains NaN values.
        
    Notes
    -----
    Variables without the time dimension are copied unchanged to the output.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Data must have a '{time_dim}' dimension")

    # Select variables to smooth
    if variables is None:
        vars_to_smooth = [var for var in ds.data_vars if time_dim in ds[var].dims]
    else:
        # Validate variables
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise ValueError(f"Variables not found in Dataset: {missing}")

        # Only smooth variables that have time dimension
        vars_to_smooth = [v for v in variables if time_dim in ds[v].dims]

        # Warn about variables without time dimension
        no_time = [v for v in variables if time_dim not in ds[v].dims]
        if no_time:
            warnings.warn(
                f"Variables without '{time_dim}' dimension will not be smoothed: {no_time}")

    # Create result dataset
    result = ds.copy()

    # Smooth each variable
    for var in vars_to_smooth:
        result[var] = smoothTimeSeries_DataArray(ds[var], time_dim, method, **kwargs)

    return result


@overload
def smoothTimeSeries(data: xr.DataArray,
                     time_dim: str = ...,
                     method: Literal['savgol', 'rolling_mean'] = ...,
                     **kwargs) -> xr.DataArray:
    ...


@overload
def smoothTimeSeries(data: xr.Dataset,
                     time_dim: str = ...,
                     method: Literal['savgol', 'rolling_mean'] = ...,
                     **kwargs) -> xr.Dataset:
    ...


@overload
def smoothTimeSeries(data: pd.DataFrame,
                     time_dim: str = ...,
                     method: Literal['savgol', 'rolling_mean'] = ...,
                     **kwargs) -> pd.DataFrame:
    ...


def smoothTimeSeries(data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
                     time_dim: str = 'time',
                     method: Literal['savgol', 'rolling_mean'] = 'savgol',
                     **kwargs) -> Union[pd.DataFrame, xr.DataArray, xr.Dataset]:
    """
    Smooth time series data using specified method.
    
    This function automatically selects the appropriate implementation based on the
    input data type: DataFrame, DataArray, or Dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame, xr.DataArray, or xr.Dataset
        Input data with time series. Must not contain NaN values in data to be smoothed.
    time_dim : str, optional
        For DataFrame: Name of the time column (required).
        For DataArray: Ignored.
        For Dataset: Ignored (use time_dim instead).
    method : {'savgol', 'rolling_mean'}, optional
        Smoothing method. Default is 'savgol'.
    time_dim : str, optional
        For DataArray/Dataset: Name of time dimension (default: 'time').
        For DataFrame: Ignored (use time_dim instead).
    **kwargs : dict
        Method-specific parameters:
        
        For 'savgol':
            - window_length : int, odd number (default: 7)
            - polyorder : int (default: 3)
            - mode : {'mirror', 'constant', 'nearest', 'wrap', 'interp'} (default: 'interp')
            
        For 'rolling_mean':
            - window : int (default: 5)
            - center : bool (default: True)
        
    Returns
    -------
    pandas.DataFrame, xr.DataArray, or xr.Dataset
        Same type as input with smoothed data.
        
    Raises
    ------
    TypeError
        If data is not a DataFrame, DataArray, or Dataset.
        If DataFrame is provided without time_dim.
    ValueError
        If time column/dimension is not found.
        If data contains NaN values.
        If smoothing parameters are invalid.
    """
    if isinstance(data, pd.DataFrame):
        return smoothTimeSeries_DataFrame(data, time_dim, method, **kwargs)

    elif isinstance(data, xr.DataArray):
        # DataArray uses time_dim parameter
        return smoothTimeSeries_DataArray(data, time_dim, method, **kwargs)

    elif isinstance(data, xr.Dataset):
        return smoothTimeSeries_Dataset(data, time_dim, method, **kwargs)

    else:
        raise TypeError(f"Input data must be a pandas DataFrame, xr DataArray, or xr Dataset. "
                        f"Got {type(data).__name__}")


#
# 2D smoothing of dataset
#
def _smooth2D_Array(data: np.ndarray,
                    method: Literal['uniform', 'gaussian', 'box'] = 'gaussian',
                    **kwargs) -> np.ndarray:
    """
    Apply 2D spatial smoothing to data.
    
    Parameters
    ----------
    data : numpy.ndarray
        3D or higher array where the last two dimensions are spatial.
        Must not contain NaN values.
    method : {'uniform', 'gaussian', 'box'}
        Smoothing method to use.
    **kwargs : dict
        Method-specific parameters:
        
        For 'uniform':
            - size : int or tuple of int (default: 3)
              Filter size in pixels. If int, same size for both dimensions.
              
        For 'gaussian':
            - sigma : float or tuple of float (default: 1.0)
              Standard deviation of Gaussian kernel. If float, same for both dimensions.
            - truncate : float (default: 4.0)
              Truncate filter at this many standard deviations.
              
        For 'box':
            - kernel_size : int or tuple of int (default: 3)
              Size of box filter. If int, same size for both dimensions.
    
    Returns
    -------
    numpy.ndarray
        Smoothed data array with same shape as input.
        
    Raises
    ------
    ValueError
        If method is not recognized or data contains NaN.
    """
    # Check for NaN values
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    # Ensure we have at least 2D data
    if data.ndim < 2:
        raise ValueError("Data must be at least 2D for spatial smoothing")

    if method == 'uniform':
        # Uniform filter (moving average)
        size = kwargs.get('size', 3)
        if isinstance(size, int):
            size = (size, size)

        # Apply uniform filter to last two dimensions
        axes = (-2, -1)
        return scipy.ndimage.uniform_filter(data, size=size, axes=axes, mode='reflect')

    elif method == 'gaussian':
        # Gaussian filter
        sigma = kwargs.get('sigma', 1.0)
        truncate = kwargs.get('truncate', 4.0)

        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)

        # Apply Gaussian filter to last two dimensions
        axes = (-2, -1)
        return scipy.ndimage.gaussian_filter(data,
                                             sigma=sigma,
                                             axes=axes,
                                             mode='reflect',
                                             truncate=truncate)

    elif method == 'box':
        # Box filter (simple averaging)
        kernel_size = kwargs.get('kernel_size', 3)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Create box kernel
        kernel = np.ones(kernel_size) / np.prod(kernel_size)

        # For higher dimensional data, we need to apply 2D convolution along last two axes
        if data.ndim == 2:
            return scipy.signal.convolve2d(data, kernel, mode='same', boundary='symm')
        else:
            # Process along last two dimensions for each slice
            result = np.empty_like(data)
            # Iterate over all but the last two dimensions
            for idx in np.ndindex(data.shape[:-2]):
                result[idx] = scipy.signal.convolve2d(data[idx],
                                                      kernel,
                                                      mode='same',
                                                      boundary='symm')
            return result

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def _findSpatialDims(dims: List[str]) -> Tuple[str, str]:
    """
    Find spatial dimensions in a list of dimension names.
    
    Parameters
    ----------
    dims : tuple of str
        Dimension names to search.
        
    Returns
    -------
    dim1, dim2 : str
        Names of the two spatial dimensions.
        
    Raises
    ------
    ValueError
        If suitable spatial dimensions cannot be found.
    """
    # First try x and y
    if 'x' in dims and 'y' in dims:
        return 'x', 'y'

    # Then try lon and lat
    if 'lon' in dims and 'lat' in dims:
        return 'lon', 'lat'

    # Also try longitude and latitude
    if 'longitude' in dims and 'latitude' in dims:
        return 'longitude', 'latitude'

    # No standard spatial dimensions found
    raise ValueError("Could not find spatial dimensions. Expected 'x' and 'y' or 'lon' and 'lat'. "
                     f"Available dimensions: {dims}")


def smooth2D_DataArray(da: xr.DataArray,
                       dim1: Optional[str] = None,
                       dim2: Optional[str] = None,
                       method: Literal['uniform', 'gaussian', 'box'] = 'gaussian',
                       **kwargs) -> xr.DataArray:
    """
    Apply 2D spatial smoothing to a DataArray.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with at least 2 spatial dimensions. Must not contain NaN values.
    dim1 : str, optional
        First spatial dimension. If None, will try to find 'x' or 'lon'.
    dim2 : str, optional
        Second spatial dimension. If None, will try to find 'y' or 'lat'.
    method : {'uniform', 'gaussian', 'box'}, optional
        Smoothing method. Default is 'gaussian'.
    **kwargs : dict
        Method-specific parameters passed to smoothing function.
        
        For 'uniform':
            - size : int or tuple of int (default: 3)
            
        For 'gaussian':
            - sigma : float or tuple of float (default: 1.0)
            - truncate : float (default: 4.0)
            
        For 'box':
            - kernel_size : int or tuple of int (default: 3)
    
    Returns
    -------
    xr.DataArray
        DataArray with smoothed spatial data. All attributes and coordinates
        are preserved.
        
    Raises
    ------
    ValueError
        If spatial dimensions are not found or data contains NaN values.
        
    Notes
    -----
    The smoothing is applied in 2D to each slice along non-spatial dimensions.
    For example, if the data has dimensions (time, lat, lon), smoothing is
    applied to each time slice independently.
    """
    # Find spatial dimensions if not provided
    if dim1 is None or dim2 is None:
        found_dim1, found_dim2 = _findSpatialDims(
            [da.dims, ] if isinstance(da.dims, str) else [str(d) for d in da.dims])
        dim1 = dim1 or found_dim1
        dim2 = dim2 or found_dim2

    # Validate dimensions exist
    if dim1 not in da.dims:
        raise ValueError(f"Dimension '{dim1}' not found in DataArray")
    if dim2 not in da.dims:
        raise ValueError(f"Dimension '{dim2}' not found in DataArray")

    # Check for NaN values
    if np.any(np.isnan(da.values)):
        raise ValueError("DataArray contains NaN values")

    # Get indices of spatial dimensions
    dim1_idx = da.dims.index(dim1)
    dim2_idx = da.dims.index(dim2)

    # Transpose data so spatial dimensions are last
    dims_order = [d for d in da.dims if d not in [dim1, dim2]] + [dim1, dim2]
    da_transposed = da.transpose(*dims_order)

    # Apply smoothing
    smoothed_data = _smooth2D_Array(da_transposed.values, method, **kwargs)

    # Create result with same dimension order as input
    result = da.copy()
    # Transpose back to original order
    result_transposed = da_transposed.copy()
    result_transposed.values = smoothed_data
    result = result_transposed.transpose(*da.dims)

    return result


def smooth2D_Dataset(ds: xr.Dataset,
                     dim1: Optional[str] = None,
                     dim2: Optional[str] = None,
                     method: Literal['uniform', 'gaussian', 'box'] = 'gaussian',
                     variables: Optional[list[str]] = None,
                     **kwargs) -> xr.Dataset:
    """
    Apply 2D spatial smoothing to variables in a Dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with at least 2 spatial dimensions. Variables to be smoothed
        must not contain NaN values.
    dim1 : str, optional
        First spatial dimension. If None, will try to find 'x' or 'lon'.
    dim2 : str, optional
        Second spatial dimension. If None, will try to find 'y' or 'lat'.
    method : {'uniform', 'gaussian', 'box'}, optional
        Smoothing method. Default is 'gaussian'.
    variables : list of str, optional
        Variables to smooth. If None, smooths all variables that have both
        spatial dimensions.
    **kwargs : dict
        Method-specific parameters passed to smoothing function.
        
    Returns
    -------
    xr.Dataset
        Dataset with smoothed spatial data. Variables without both spatial
        dimensions are preserved unchanged. All attributes are preserved.
        
    Raises
    ------
    ValueError
        If spatial dimensions are not found, specified variables don't exist,
        or any variable to be smoothed contains NaN values.
        
    Notes
    -----
    Only variables that contain both spatial dimensions are smoothed. Other
    variables are copied unchanged to the output.
    """
    # If no dimensions specified, let the first variable determine them
    # This ensures consistent dimension finding across all variables
    if dim1 is None or dim2 is None:
        # Try to find dimensions from the Dataset's dims
        # This is just for early validation and consistency
        try:
            found_dim1, found_dim2 = _findSpatialDims(
                [ds.dims, ] if isinstance(ds.dims, str) else [str(d) for d in ds.dims])
            dim1 = dim1 or found_dim1
            dim2 = dim2 or found_dim2
        except ValueError:
            # If we can't find them at Dataset level, the DataArray function
            # will try to find them or raise an appropriate error
            pass

    # Select variables to smooth
    if variables is None:
        # Get all data variables - let smooth2DDataArray handle dimension checking
        vars_to_process = list(ds.data_vars)
    else:
        # Validate specified variables exist
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise ValueError(f"Variables not found in Dataset: {missing}")
        vars_to_process = variables

    # Create result dataset
    result = ds.copy()

    # Process each variable
    for var in vars_to_process:
        try:
            # Try to smooth the variable
            result[var] = smooth2D_DataArray(ds[var], dim1, dim2, method, **kwargs)
        except ValueError as e:
            # If it fails because of missing dimensions, just skip this variable
            if "not found in DataArray" in str(e) or "Could not find spatial dimensions" in str(e):
                # Variable doesn't have the required dimensions, keep original
                if variables is not None and var in variables:
                    # User specifically requested this variable, so warn them
                    warnings.warn(
                        f"Variable '{var}' does not have required spatial dimensions, skipping.")
            else:
                # Some other error (like NaN values), re-raise
                raise

    return result


@overload
def smooth2D(data: xr.DataArray,
             dim1: Optional[str] = ...,
             dim2: Optional[str] = ...,
             method: Literal['uniform', 'gaussian', 'box'] = ...,
             variables: None = ...,
             **kwargs) -> xr.DataArray:
    ...


@overload
def smooth2D(data: xr.Dataset,
             dim1: Optional[str] = ...,
             dim2: Optional[str] = ...,
             method: Literal['uniform', 'gaussian', 'box'] = ...,
             variables: Optional[list[str]] = ...,
             **kwargs) -> xr.Dataset:
    ...


def smooth2D(data: Union[xr.DataArray, xr.Dataset],
             dim1: Optional[str] = None,
             dim2: Optional[str] = None,
             method: Literal['uniform', 'gaussian', 'box'] = 'gaussian',
             variables: Optional[list[str]] = None,
             **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    """
    Apply 2D spatial smoothing to data.
    
    This function automatically selects the appropriate implementation based on the
    input data type: DataArray or Dataset.
    
    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data with at least 2 spatial dimensions. Must not contain NaN values
        in data to be smoothed.
    dim1 : str, optional
        First spatial dimension. If None, will try to find 'x' or 'lon'.
    dim2 : str, optional
        Second spatial dimension. If None, will try to find 'y' or 'lat'.
    method : {'uniform', 'gaussian', 'box'}, optional
        Smoothing method. Default is 'gaussian'.
    variables : list of str, optional
        For Dataset: Variables to smooth (default: all with both spatial dims).
        For DataArray: Ignored.
    **kwargs : dict
        Method-specific parameters:
        
        For 'uniform':
            - size : int or tuple of int (default: 3)
              Filter size in pixels.
              
        For 'gaussian':
            - sigma : float or tuple of float (default: 1.0)
              Standard deviation of Gaussian kernel.
            - truncate : float (default: 4.0)
              Truncate filter at this many standard deviations.
              
        For 'box':
            - kernel_size : int or tuple of int (default: 3)
              Size of box filter.
    
    Returns
    -------
    xr.DataArray or xr.Dataset
        Same type as input with spatially smoothed data.
        
    Raises
    ------
    TypeError
        If data is not a DataArray or Dataset.
    ValueError
        If spatial dimensions are not found or data contains NaN values.
        
    Examples
    --------
    Smooth a DataArray with Gaussian filter:
    
    >>> da = xr.DataArray(data, dims=['time', 'lat', 'lon'])
    >>> smoothed = smooth2D(da, method='gaussian', sigma=2.0)
    
    Smooth specific variables in a Dataset:
    
    >>> ds = xr.Dataset({'temp': da1, 'pressure': da2})
    >>> smoothed = smooth2D(ds, variables=['temp'], method='uniform', size=5)
    
    Use custom dimension names:
    
    >>> smoothed = smooth2D(data, dim1='x_coord', dim2='y_coord')
    """
    if isinstance(data, xr.DataArray):
        if variables is not None:
            warnings.warn("'variables' parameter is ignored for DataArray input")
        return smooth2D_DataArray(data, dim1, dim2, method, **kwargs)

    elif isinstance(data, xr.Dataset):
        return smooth2D_Dataset(data, dim1, dim2, method, variables, **kwargs)

    else:
        raise TypeError(f"Input data must be xr DataArray or Dataset. "
                        f"Got {type(data).__name__}")


def interpolateValues(points: np.ndarray,
                      points_crs: CRS | None,
                      data: xr.DataArray,
                      method: InterpOptions = 'nearest') -> np.ndarray:
    """
    Interpolate values from a 2D grid-based DataArray at given x, y or lat, lon points.

    Parameters
    ----------
    points : np.ndarray
        A (N, 2) array of coordinates. Each row should contain an (x, y) or (lon, lat) pair.
    points_crs : CRS
        A coordinate system for the points.
    data : xr.DataArray
        A DataArray with coordinates either ('x', 'y') or ('lon', 'lat').
    method : {'linear', 'nearest', 'cubic'}
        Interpolation method to use.

    Returns
    -------
    np.ndarray
        A 1D array of interpolated values with length N.

    Raises
    ------
    ValueError
        If DataArray does not have suitable coordinates for interpolation.
    """
    array_crs = watershed_workflow.crs.from_xarray(data)
    if points_crs is not None and array_crs is not None:
        x, y = watershed_workflow.warp.xy(points[:, 0], points[:, 1], points_crs, array_crs)
    else:
        x, y = points[:, 0], points[:, 1]

    if { 'x', 'y'}.issubset(data.coords):
        coord_names = ('x', 'y')
    elif { 'lon', 'lat'}.issubset(data.coords):
        coord_names = ('lon', 'lat')
    else:
        raise ValueError("DataArray must have coordinates ('x', 'y') or ('lon', 'lat')")

    coords = xr.Dataset({coord_names[0]: ("points", x), coord_names[1]: ("points", y)})

    interpolated = data.interp(coords, method=method)
    return interpolated.values


def imputeHoles2D(arr: xr.DataArray, nodata: Any = np.nan, method: str = 'cubic') -> xr.DataArray:
    """
    Interpolate values for missing data in rasters using scipy griddata.
    
    Parameters
    ----------
    arr : xarray.DataArray
        Input raster data with missing values to interpolate.
    nodata : Any, optional
        Value representing missing data. Default is numpy.nan.
    method : str, optional
        Interpolation method for scipy.interpolate.griddata.
        Valid options: 'linear', 'nearest', 'cubic'. Default is 'cubic'.
        
    Returns
    -------
    numpy.ndarray
        Interpolated array with missing values filled.
        
    Notes
    -----
    This function may raise an error if there are holes on the boundary.
    The interpolation is performed using scipy.interpolate.griddata with
    the specified method.
    """
    if nodata is np.nan:
        mask = np.isnan(arr)
    else:
        mask = (arr == nodata)

    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])
    xx, yy = np.meshgrid(x, y)

    #get only the valid values
    x1 = xx[~mask]
    y1 = yy[~mask]
    newarr = arr[~mask]

    res = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method)
    return res


def rasterizeGeoDataFrame(gdf: gpd.GeoDataFrame,
                          column: str,
                          resolution: float,
                          bounds: Optional[Tuple[float, float, float, float]] = None,
                          nodata: Optional[Union[int, float]] = None) -> xr.DataArray:
    """
    Convert a GeoDataFrame to a rasterized DataArray based on a column's values.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing geometries and data.
    column : str
        Name of the column containing values to rasterize. Must be a numeric type.
    resolution : float
        Spatial resolution of the output raster in the units of the GeoDataFrame's CRS.
        This defines the size of each pixel.
    bounds : tuple of float, optional
        Bounding box as (minx, miny, maxx, maxy). If None, bounds are computed
        from the GeoDataFrame's total bounds.
    nodata : int or float, optional
        Value to use for pixels not covered by any geometry. If None, defaults
        to NaN for float columns and -999 for integer columns.
        
    Returns
    -------
    xarray.DataArray
        Rasterized data with dimensions ('y', 'x') and coordinates defined by
        the spatial extent and resolution. Areas outside geometries are set to
        the nodata value. The data type matches the column's data type.
        
    Raises
    ------
    ValueError
        If column is not found in the GeoDataFrame.
        If column is not numeric type.
        If GeoDataFrame is empty.
        If resolution is not positive.
        
    Notes
    -----
    The function uses rasterio's rasterization capabilities to burn geometries
    into a raster. When geometries overlap, the value from the last geometry
    in the GeoDataFrame is used.
    
    The output DataArray includes the CRS information in its attributes if
    the GeoDataFrame has a CRS defined.
    
    The dtype of the output array matches the dtype of the input column.
    """
    # reset the index to make 'index' a column if that is what is
    # requested to be colored.
    if column == 'index':
        gdf = gdf.reset_index()

    # Validate inputs
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in GeoDataFrame")

    if len(gdf) == 0:
        raise ValueError("GeoDataFrame is empty")

    if resolution <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}")

    # Determine output data type and fill value
    out_dtype = gdf[column].dtype

    # Set nodata value based on dtype if not provided
    fill_value = out_dtype.type()
    if nodata is None:
        if np.issubdtype(out_dtype, np.integer):
            fill_value = -999
        elif np.issubdtype(out_dtype, np.floating):
            fill_value = np.nan
        else:
            raise ValueError(f"Column '{column}' must be numeric type, got {out_dtype}")
    else:
        fill_value = out_dtype.type(nodata)

    # Validate geometry types and create a mask of geometry that are
    # valid and values that are not na
    valid_types = (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
    valid_mask = gdf[column].notna() & \
        gdf.geometry.notna() & gdf.geometry.is_valid & \
        gdf.geometry.apply(lambda s : isinstance(s, valid_types))

    # Get bounds
    if bounds is None:
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy

    minx, miny, maxx, maxy = bounds

    # Calculate raster dimensions
    width = int(np.ceil((maxx-minx) / resolution))
    height = int(np.ceil((maxy-miny) / resolution))

    # Create transform
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Initialize array with nodata value
    raster = np.full((height, width), fill_value, dtype=out_dtype)

    if valid_mask.sum() == 0:
        # No valid data to rasterize
        pass
    else:
        # Create list of (geometry, value) pairs
        shapes = [(geom, value)
                  for geom, value in zip(gdf.geometry[valid_mask], gdf[column][valid_mask])]

        # Rasterize
        rasterio.features.rasterize(shapes,
                                    out=raster,
                                    transform=transform,
                                    fill=fill_value,
                                    dtype=out_dtype)

    # Create coordinate arrays
    # X coordinates (cell centers)
    x_coords = np.arange(minx + resolution/2, maxx + resolution, resolution)[:width]
    # Y coordinates (cell centers) - note that y decreases as row index increases
    y_coords = np.arange(maxy - resolution/2, miny - resolution, -resolution)[:height]

    # Create DataArray
    da = xr.DataArray(raster, dims=['y', 'x'], coords={ 'x': x_coords, 'y': y_coords }, name=column)

    # Add attributes
    da.attrs['resolution'] = resolution
    da.attrs['source_column'] = column
    da.attrs['nodata'] = fill_value

    # Add CRS if available
    if gdf.crs is not None:
        da.attrs['crs'] = str(gdf.crs)

    return da
