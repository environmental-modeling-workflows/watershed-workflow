"""I/O Utilities"""

import os
import numpy as np
import fiona
import rasterio
import shapely.geometry
import collections
import logging
import h5py
import datetime, cftime

import watershed_workflow.crs


def write_to_raster(filename, profile, array):
    """Write a numpy array to raster file."""
    assert (len(array.shape) >= 2 and len(array.shape) <= 3)
    if len(array.shape) == 2:
        array = array.reshape((1, ) + array.shape)

    profile = profile.copy()
    profile.update(count=array.shape[2], compress='lzw')

    with rasterio.open(filename, 'w', **profile) as fout:
        for i in range(array.shape[0]):
            fout.write(array[i, :, :], i + 1)


def write_to_shapefile(filename, shps, crs, extra_properties=None):
    """Write a collection of shapes to a file using fiona"""
    if len(shps) == 0:
        return

    # set up the schema
    schema = dict()
    if type(shps[0]) is shapely.geometry.Polygon:
        schema['geometry'] = 'Polygon'
    elif type(shps[0]) is shapely.geometry.LineString:
        schema['geometry'] = 'LineString'
    else:
        raise RuntimeError('Currently this function only writes Polygon or LineString types')
    schema['properties'] = collections.OrderedDict()

    # set up the properties schema
    def register_type(key, atype):
        if atype is int:
            schema['properties'][key] = 'int'
        elif atype is str:
            schema['properties'][key] = 'str'
        elif atype is float:
            schema['properties'][key] = 'float'
        else:
            pass

    if extra_properties is None:
        extra_properties = dict()
    for key, val in extra_properties.items():
        register_type(key, type(val))

    try:
        item_properties = shps[0].properties
    except AttributeError:
        pass
    else:
        for key, val in item_properties.items():
            register_type(key, type(val))

    with fiona.open(filename,
                    'w',
                    driver='ESRI Shapefile',
                    crs=watershed_workflow.crs.to_fiona(crs),
                    crs_wkt=watershed_workflow.crs.to_wkt(crs),
                    schema=schema) as c:
        for shp in shps:
            props = extra_properties.copy()
            try:
                props.update(shp.properties)
            except AttributeError:
                pass

            for key in list(props.keys()):
                if key not in schema['properties']:
                    props.pop(key)

            c.write({ 'geometry': shapely.geometry.mapping(shp), 'properties': props })


def write_dataset_to_hdf5(filename, dataset, attributes=None, time0=None, calendar='noleap'):
    """Writes a DatasetCollection and attributes to an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    ts : dict or dataframe
        Dictionary or dataframe of time series, with keys being the name of the time series
        and values being the time series data.
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

    keys = list(dataset.keys())
    profile = dataset.profile
    times = dataset.times

    # construct the x,y,t arrays
    transform = profile['transform']
    if not transform.is_rectilinear:
        raise ValueError(
            'Raster cannot be written as transform is not rectilinear, which is an assumption of simulators.'
        )
    x = np.array([(transform * (i, 0))[0] for i in range(profile['width'])])
    y = np.array([(transform * (0, j))[1] for j in range(profile['height'])])
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
            # dat has shape (nband, nrow, ncol)
            assert (dataset.data[key].shape[0] == times.shape[0])
            assert (dataset.data[key].shape[1] == y.shape[0])
            assert (dataset.data[key].shape[2] == x.shape[0])

            grp = fid.create_group(key)
            for i in range(len(times)):
                idat = dataset.data[key][i, :, :]
                # flip rows to match the order of y
                rev_idat = np.flip(idat, axis=0)
                grp.create_dataset(str(i), data=rev_idat)

        if attributes is not None:
            for key, val in attributes.items():
                fid.attrs[key] = val


def write_timeseries_to_hdf5(filename, ts, attributes=None, time0=None):
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
        time0 = datetime.datetime.strptime(time0, '%Y-%m-%d').date()
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
