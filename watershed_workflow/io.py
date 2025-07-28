"""I/O Utilities"""

import os
import numpy as np
import logging
import h5py
import cftime
import rasterio.transform
import xarray as xr

import watershed_workflow.crs

def writeDatasetToHDF5(filename, dataset, attributes=None, time0=None, calendar='noleap'):
    """Writes an xarray.Dataset and attributes to an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    dataset : xarray.Dataset
        Dataset containing the data to write.
    attributes : dict
        Dictionary of attributes to write to the HDF5 file.
    time0 : str or datetime.date object, optional
        Time to use as the zero time for the time series.  If not provided, the first
        time in the time series is used.

    Returns
    -------
    None

    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    keys = list(dataset.data_vars.keys())
    times = dataset.time.values

    # construct the x,y arrays from the dataset coordinates
    x = dataset.x.values
    y = dataset.y.values
    
    if time0 is None:
        time0 = times[0]

    if type(time0) is str:
        time0_split = time0.split('-')
        time0 = cftime.datetime(int(time0_split[0]),
                                int(time0_split[1]),
                                int(time0_split[2]),
                                calendar=calendar)
    if attributes is None:
        attributes = dict()
    attributes['origin date'] = str(time0)
    times = np.array([(t - time0).total_seconds() for t in times])
    times = times.astype(np.int32)
    logging.info('Writing HDF5 file: {}'.format(filename))
    with h5py.File(filename, 'w') as fid:
        fid.create_dataset('time [s]', data=times)

        # make y increasing order
        rev_y = y[::-1]
        fid.create_dataset('y [m]', data=rev_y)
        fid.create_dataset('x [m]', data=x)

        for key in keys:
            # dat has shape (ntime, ny, nx)
            data = dataset[key].values
            assert (data.shape[0] == times.shape[0])
            assert (data.shape[1] == y.shape[0])
            assert (data.shape[2] == x.shape[0])

            grp = fid.create_group(key)
            for i in range(len(times)):
                idat = data[i, :, :]
                # flip rows to match the order of y
                rev_idat = np.flip(idat, axis=0)
                grp.create_dataset(str(i), data=rev_idat)

        if attributes is not None:
            for key, val in attributes.items():
                fid.attrs[key] = val


def writeTimeseriesToHDF5(filename, ts, attributes=None, time0=None):
    """Writes a time series and attributes to an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    ts : dict or dataframe
        Dictionary or dataframe of time series, with keys being the name of the time series
        and values being the time series data.
    attributes : dict, optional
        Dictionary of attributes to write to the HDF5 file.
    time0 : str or datetime.date object, optional
        Time to use as the zero time for the time series.  If not provided, the first
        time in the time series is used.

    Returns
    -------
    None
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    keys = list(ts.keys())
    keys.remove('time [datetime]')
    times = ts['time [datetime]']

    if time0 is None:
        time0 = times.values[0]
    if type(time0) is str:
        time0 = cftime.datetime.strptime(time0, '%Y-%m-%d').date()
    if attributes is None:
        attributes = dict()
    attributes['origin date'] = str(time0)
    attributes['start date'] = str(times.values[0])
    attributes['end date'] = str(times.values[-1])

    times = np.array([(t - time0).total_seconds() for t in times])
    times = times.astype(np.int32)
    logging.info('Writing HDF5 file: {}'.format(filename))
    with h5py.File(filename, 'w') as fid:
        fid.create_dataset('time [s]', data=times)

        for key in keys:
            fid.create_dataset(key, data=ts[key][:])
        if attributes is not None:
            for key, val in attributes.items():
                fid.attrs[key] = val
