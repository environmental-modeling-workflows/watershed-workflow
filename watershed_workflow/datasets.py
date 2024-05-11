"""Datasets are collections of N-dimensional data, which vary in time."""

import attr
import numpy as np
import collections.abc
import cftime, datetime

import watershed_workflow.utils

def np_array_convertor(thing, *args, **kwargs):
    if isinstance(thing, np.ndarray):
        return thing
    else:
        return np.ndarray(thing, *args, **kwargs)


@attr.define
class Data:
    """Simple struct for storing time-dependent rasters.

    Note that time is always the first dimension, e.g.

      times.shape[0] == data.shape[0]
    """
    profile: dict
    times: np.ndarray = attr.field(converter=np_array_convertor)
    data: np.ndarray = attr.field(converter=np_array_convertor)


@attr.define
class Dataset(collections.abc.MutableMapping):
    """Stores a collection of datasets with shared times and profile."""
    profile: dict
    times: np.ndarray
    data: dict = attr.Factory(dict)

    def __getitem__(self, key):
        return Data(self.profile, self.times, self.data[key])

    def __setitem__(self, key, val):
        if isinstance(val, tuple):
            self.__setitem__(key, Data(*val))
        elif isinstance(val, Data):
            self.data[key] = val.data
        else:
            self.data[key] = np.array(val)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def __iter__(self):
        for k in self.data:
            yield k

    def __len__(self):
        return len(self.data)

    def canContain(self, dset):
        return (dset.profile == self.profile) and (dset.times == self.times).all()


class State(collections.abc.MutableMapping):
    """This is a multi-key dictionary.

    Each key is a string variable name.  Each value is a (profile,
    times, raster) tuple.  Profiles and times may be shared across
    multiple keys, hence the need for a special dictionary.

    Note that actual data is stored as a simple list of Dataset
    collections.

    """
    def __init__(self):
        self.collections = []

    def __getitem__(self, key):
        for col in self.collections:
            if key in col:
                return col[key]

    def __setitem__(self, key, val):
        if isinstance(val, tuple):
            self.__setitem__(key, Data(*val))
        else:
            for col in self.collections:
                if col.canContain(val):
                    col[key] = val
                    return
            self.collections.append(Dataset(val.profile, val.times, { key: val.data }))

    def __delitem__(self, key):
        for col in self.collections:
            if key in col:
                col.__delitem__(key)
                break

    def __iter__(self):
        for col in self.collections:
            for k in col:
                yield k

    def __len__(self):
        return sum(len(col) for col in self.collections)


def removeLeapDay(dataset):
    """Removes leap days from a dataset, in place.

    Here we follow the DayMet convention of removing Dec 31, not Feb 29.
    """
    times = dataset.times
    not_leap = np.array([t.dayofyr != 366 for t in times])

    # convert times to cftime with noleap calendar
    times = np.array([cftime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond, calendar='noleap') for t in times[not_leap]])
    dataset.times = times

    for k in dataset:
        print(type(dataset), type(k), k)
        dataset[k] = dataset[k].data[not_leap]

    
def computeAverageYear(dataset, output_nyears=1, start_year=2000,
                       smooth=False, smooth_kwargs=None,
                       interpolate=True, interpolate_kwargs=None):
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
        new_times = np.array([new_start + i * dt for i in range(365*(end-start))])

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
    times_out = np.array([start + i * dt for i in range(365*output_nyears)])
    dataset_out = Dataset(dataset_interp.profile, times_out)
    for k in dataset_interp:
        dataset_out[k] = watershed_workflow.utils.compute_average_year(dataset_interp[k].data,
                                                                       output_nyears,
                                                                       smooth,
                                                                       **smooth_kwargs)
    return dataset_out
        
                                                                             
        
