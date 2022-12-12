"""Writer for ATS output from Watershed Workflow."""

def writeATS(dat, x, y, attrs, filename, **kwargs):
    """Accepts a dictionary of ATS data and writes it to HDF5 file."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    dat = daymetToATS(dat, **kwargs)

    logging.info('Writing ATS file: {}'.format(filename))
    with h5py.File(filename, 'w') as fid:
        fid.create_dataset('time [s]', data=dat['time [s]'])
        assert (len(x.shape) == 1)
        assert (len(y.shape) == 1)
        ntimes = dat['time [s]'].shape[0]

        # ATS requires increasing order for y
        rev_y = y[::-1]
        fid.create_dataset('row coordinate [m]', data=rev_y)
        fid.create_dataset('col coordinate [m]', data=x)

        for key in dat.keys():
            if key != 'time [s]':
                # dat has shape (nband, nrow, ncol)
                assert (dat[key].shape[0] == ntimes)
                assert (dat[key].shape[1] == y.shape[0])
                assert (dat[key].shape[2] == x.shape[0])
                # dat[key] = dat[key].swapaxes(1,2) # reshape to (nband, nrow, ncol)
                grp = fid.create_group(key)
                for i in range(ntimes):
                    idat = dat[key][i, :, :]
                    # flip rows to match the order of y, so it starts with (x0,y0) in the upper left
                    rev_idat = np.flip(idat, axis=0)
                    grp.create_dataset(str(i), data=rev_idat)

        for key, val in attrs.items():
            fid.attrs[key] = val

    return dat


def writeHDF5(filename, profile, times, dat, attrs):
    """Write daymet to a single HDF5 file."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    if 'time [s]' in dat:
        time = dat['time [s]'][:]
        assert (len(time) == dat[list(dat.keys())[0]].shape[0])
    else:
        time = np.arange(0, dat[list(dat.keys())[0]].shape[0], 1) * 86400.

    logging.info('Writing HDF5 file: {}'.format(filename))
    with h5py.File(filename, 'w') as fid:
        fid.create_dataset('time [s]', data=time)
        assert (len(x.shape) == 1)
        assert (len(y.shape) == 1)

        # make y increasing order
        rev_y = y[::-1]
        fid.create_dataset('y [m]', data=rev_y)
        fid.create_dataset('x [m]', data=x)

        for key in dat.keys():
            if key != 'time [s]':
                # dat has shape (nband, nrow, ncol)
                assert (dat[key].shape[0] == time.shape[0])
                assert (dat[key].shape[1] == y.shape[0])
                assert (dat[key].shape[2] == x.shape[0])

                grp = fid.create_group(key)
                for i in range(len(time)):
                    idat = dat[key][i, :, :]
                    # flip rows to match the order of y, so it starts with (x0,y0) in the upper left
                    # ETC: is this right?  Should be increasing, so in the lower left after flipped?
                    rev_idat = np.flip(idat, axis=0)
                    grp.create_dataset(str(i), data=rev_idat)

        for key, val in attrs.items():
            fid.attrs[key] = val

    return
