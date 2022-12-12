"""Manipulate DayMet data structures.

DayMet is downloaded in box mode based on watershed bounds, then it can be converted to
hdf5 files that models can read.
"""

import requests
import datetime
import logging
import h5py, netCDF4
import sys, os
import numpy as np
import time
import watershed_workflow


def daymet_to_daily_averages(dat):
    """Convert dictionary of Daymet datasets to daily average data in standard form.

    This:

    - takes tmin and tmax to compute a mean
    - splits rain and snow precip based on mean air temp
    - standardizes units and names for ATS

    """
    logging.info('Converting to ATS met input')
    dout = dict()

    # make missing values (-9999) as NaNs to do math while propagating NaNs
    for key in dat.keys():
        dat[key][dat[key] == -9999] = np.nan

    mean_air_temp_c = (dat['tmin'] + dat['tmax']) / 2.0
    precip_ms = dat['prcp'] / 1.e3 / 86400.  # mm/day --> m/s

    time = np.arange(0, dat[list(dat.keys())[0]].shape[0], 1) * 86400.

    dout['air temperature [K]'] = 273.15 + mean_air_temp_c  # K
    # note that shortwave radiation in daymet is averged over the unit daylength, not per unit day.
    dout['incoming shortwave radiation [W m^-2]'] = dat['srad'] * dat['dayl'] / 86400  # Wm2
    dout['vapor pressure air [Pa]'] = dat['vp'] # Pa
    dout['precipitation rain [m s^-1]'] = np.where(mean_air_temp_c >= 0, precip_ms, 0)
    dout['precipitation snow [m SWE s^-1]'] = np.where(mean_air_temp_c < 0, precip_ms, 0)
    dout['time [s]'] = time

    logging.debug(f"output dout shape: {dout['incoming shortwave radiation [W m^-2]'].shape}")
    return dout



