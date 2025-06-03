"""Datasets are collections of N-dimensional data, which vary in time."""

import attr
import numpy as np
import collections.abc
import cftime, datetime
import xarray

import watershed_workflow.utils
import watershed_workflow.crs


def interpolateDataset(points : np.ndarray,
                       points_crs : watershed_workflow.crs.CRS | None,
                       dataarray : xarray.DataArray,
                       **kwargs) -> np.ndarray:
    """Interpolate from a data array onto a set of points."""
    dataarray_crs = watershed_workflow.crs.from_xarray(dataarray)
    if points_crs is not None and dataarray_crs is not None:
        points = watershed_workflow.warp.points(points, points_crs, dataarray_crs)
    x = xarray.DataArray(points[:,0], dims="points")
    y = xarray.DataArray(points[:,1], dims="points")
    return dataarray.interp(x=x, y=y, **kwargs).values


def removeLeapDay(dataset):
    """Removes leap days from a dataset.

    Here we follow the DayMet convention of removing Dec 31, not Feb 29.
    """
    times = dataset.times
    not_leap = np.array([t.dayofyr != 366 for t in times])

    # convert times to cftime with noleap calendar
    times = np.array([
        cftime.datetime(t.year,
                        t.month,
                        t.day,
                        t.hour,
                        t.minute,
                        t.second,
                        t.microsecond,
                        calendar='noleap') for t in times[not_leap]
    ])
    dataset.times = times

    for k in dataset:
        print(type(dataset), type(k), k)
        dataset[k] = dataset[k].data[not_leap]


def computeAverageYear(dataset,
                       output_nyears=1,
                       start_year=2000,
                       smooth=False,
                       smooth_kwargs=None,
                       interpolate=True,
                       interpolate_kwargs=None):
    """Interpolates, averages and smooths to form a "typical" year.

    Parameters
    ----------
    dataset : Dataset object
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
    Dataset
      The smoothed data.

    """
    if dataset.times[0].calendar != 'noleap':
        raise ValueError('Calendar of the incoming dataset must be "noleap"')
    dt = datetime.timedelta(days=1)

    if interpolate:
        if interpolate_kwargs is None:
            interpolate_kwargs = dict()

        t0 = dataset.times[0]
        if (t0 - dt).year < t0.year:
            start = t0.year
        else:
            start = t0.year + 1

        t1 = dataset.times[-1]
        if (t1 + dt).year > t1.year:
            end = t1.year + 1
        else:
            end = t1.year

        new_start = cftime.datetime(start, 1, 1, calendar='noleap')
        new_times = np.array([new_start + i*dt for i in range(365 * (end-start))])

        dataset_interp = Dataset(dataset.profile, new_times)
        for k in dataset:
            dataset_interp[k] = \
                watershed_workflow.utils.interpolate_in_time(dataset.times,
                                                             dataset[k].data,
                                                             new_times,
                                                             units='days since 2000-1-1',
                                                             **interpolate_kwargs)
    else:
        dataset_interp = dataset

    if smooth_kwargs is None:
        smooth_kwargs = dict()
    start = cftime.datetime(2000, 1, 1, calendar='noleap')
    times_out = np.array([start + i*dt for i in range(365 * output_nyears)])
    dataset_out = Dataset(dataset_interp.profile, times_out)
    for k in dataset_interp:
        dataset_out[k] = watershed_workflow.utils.compute_average_year(
            dataset_interp[k].data, output_nyears, smooth, **smooth_kwargs)
    return dataset_out
