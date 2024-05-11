"""Timeseries are collections of 1D arrays and a time array.

These are stored as pandas.DataFrames.  This module provides some
common functions used on timeseries data.
"""

import pandas
import cftime, datetime
import numpy as np
import watershed_workflow.utils


def removeLeapDay(df):
    """Removes leap days from the timeseries and converts calendar to noleap."""
    noleap = df[np.array([t.dayofyr != 366 for t in df['time [datetime]']])]
    df['time [datetime]'] = \
        [cftime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, calendar='noleap')
         for t in noleap['time [datetime]']]
    return df


def computeAverageYear(df, output_nyears=1, start_year=2000,
                       smooth=False, smooth_kwargs=None,
                       interpolate=True, interpolate_kwargs=None):
    """Interpolates, averages and smooths to form a "typical" year.

    Parameters
    ----------
    df : pandas.DataFrame
      A dataset object.
    output_nyears : int, optional
      Number of years to repeat the output.  Default is 1.
    start_year : int, optional
      Output will start at this year, which should be somewhat arbitrary.
      Default is 2000.
    smooth : bool, optional
      Filter the data using a Sav-Gol filter from Scipy. Default is
      False.
    smooth_kwargs : dict, optional
      Options passed to scipy.signal.savgol_filter.
    interpolate : bool, optional
      Before averaging, interpolate the dataset onto a daily interval.
      This is required for averaging, but can be turned off if the
      data is already on a daily interval.  Default is true.
    interpolate_kwargs : dict, optional
      Options passed to scipy.interp.interp1d. Of particular use is
      'kind', which can be 'linear' (default) or 'quadratic', 'cubic',
      or others.

    Returns
    -------
    pandas.DataFrame
      The smoothed data.

    """
    if df['time [datetime]'][0].calendar != 'noleap':
        df = removeLeapDay(df)

    times = df['time [datetime]'].to_numpy()
    dt = datetime.timedelta(days=1)
    if interpolate:
        if interpolate_kwargs is None:
            interpolate_kwargs = dict()
        
        t0 = times[0]
        if (t0 - dt).year < t0.year:
            start = t0.year
        else:
            start = t0.year + 1

        t1 = times[-1]
        if (t1 + dt).year > t1.year:
            end = t1.year + 1
        else:
            end = t1.year

        new_start = cftime.datetime(start, 1, 1, calendar='noleap')
        new_times = np.array([new_start + i * dt for i in range(365*(end-start))])

        df_interp = pandas.DataFrame()
        df_interp['time [datetime]'] = new_times

        for k in df.keys():
            if k != 'time [datetime]':
                df_interp[k] = \
                    watershed_workflow.utils.interpolate_in_time(times,
                                                                 df[k],
                                                                 new_times,
                                                                 units='days since 2000-1-1',
                                                                 **interpolate_kwargs)
    else:
        df_interp = df
    
    if smooth_kwargs is None:
        smooth_kwargs = dict()
    start = cftime.datetime(2000, 1, 1, calendar='noleap')
    times_out = np.array([start + i * dt for i in range(365*output_nyears)])
    df_out = pandas.DataFrame()
    df_out['time [datetime]'] = times_out
    for k in df_interp.keys():
        if k != 'time [datetime]':
            df_out[k] = watershed_workflow.utils.compute_average_year(df_interp[k],
                                                                      output_nyears,
                                                                      smooth,
                                                                      **smooth_kwargs)
    return df_out
