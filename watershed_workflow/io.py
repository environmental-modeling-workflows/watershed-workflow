"""I/O Utilities"""

from typing import Optional, Dict, Union, Any
import os
import numpy as np
import logging
import h5py
import cftime
import rasterio.transform
import xarray as xr
import pandas as pd

import watershed_workflow.crs


def writeDatasetToHDF5(filename: str,
                       dataset: xr.Dataset,
                       attributes: Optional[Dict[str, Any]] = None,
                       time0: Optional[Union[str, cftime.datetime]] = None,
                       calendar: str = 'noleap') -> None:
    """
    Write an xarray.Dataset and attributes to an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    dataset : xarray.Dataset
        Dataset containing the data to write. Must have 'time', 'x', and 'y' coordinates.
    attributes : dict, optional
        Dictionary of attributes to write to the HDF5 file. Default is None.
    time0 : str or cftime.datetime, optional
        Time to use as the zero time for the time series. If not provided, the first
        time in the time series is used. If string, should be in 'YYYY-MM-DD' format.
    calendar : str, optional
        Calendar type to use for time conversion. Default is 'noleap'.

    Raises
    ------
    KeyError
        If required coordinates ('time', 'x', 'y') are missing from dataset.
    ValueError
        If dataset dimensions don't match expected shapes.
        
    Notes
    -----
    The function writes time as seconds since time0, with y coordinates stored in
    strictly increasing order (flipping if necessary). Data arrays are also flipped
    vertically to match the y coordinate ordering.
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

    if isinstance(time0, str):
        time0_split = time0.split('-')
        time0 = cftime.datetime(int(time0_split[0]),
                                int(time0_split[1]),
                                int(time0_split[2]),
                                calendar=calendar)
    if attributes is None:
        attributes = dict()
    attributes['origin date'] = str(time0)
    if dataset.attrs is not None:
        attributes.update(dataset.attrs)
    
    times = np.array([(t - time0).total_seconds() for t in times])
    times = times.astype(np.int32)
    logging.info('Writing HDF5 file: {}'.format(filename))
    with h5py.File(filename, 'w') as fid:
        fid.create_dataset('time [s]', data=times)

        # ensure y is stored in increasing order
        if y[-1] < y[0]:
            y = y[::-1]
            flip_y = True
        else:
            flip_y = False
        fid.create_dataset('y [m]', data=y)
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
                if flip_y:
                    idat = np.flip(idat, axis=0)
                grp.create_dataset(str(i), data=idat)

        if attributes is not None:
            for key, val in attributes.items():
                fid.attrs[key] = val


def writeTimeseriesToHDF5(filename: str,
                          ts: Union[Dict[str, Any], pd.DataFrame],
                          attributes: Optional[Dict[str, Any]] = None,
                          time0: Optional[Union[str, cftime.datetime]] = None) -> None:
    """
    Write a time series and attributes to an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    ts : dict or pandas.DataFrame
        Dictionary or DataFrame of time series data. Must contain a 'time [datetime]' key/column
        with datetime values. Other keys/columns contain the time series data.
    attributes : dict, optional
        Dictionary of attributes to write to the HDF5 file. Default is None.
    time0 : str or cftime.datetime, optional
        Time to use as the zero time for the time series. If not provided, the first
        time in the time series is used. If string, should be in 'YYYY-MM-DD' format.

    Raises
    ------
    KeyError
        If 'time [datetime]' key is not found in the input data.
    ValueError
        If time0 string format is invalid.
        
    Notes
    -----
    Time values are converted to seconds since time0 and stored as 32-bit integers.
    The function automatically adds 'origin date', 'start date', and 'end date' 
    attributes to the output file.
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    keys = list(ts.keys())
    keys.remove('time')
    times = ts['time']

    if time0 is None:
        time0 = times.values[0]
    if isinstance(time0, str):
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
