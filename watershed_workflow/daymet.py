"""Manipulate DayMet data structures.

DayMet is downloaded in box mode based on watershed bounds, then it can be converted to
hdf5 files that models can read.
"""

import logging
import numpy as np
import datetime
import watershed_workflow.datasets


def getAttributes(bounds, start, end):
    # set the wind speed height, which is made up
    attributes = dict()
    attributes['DayMet x min'] = bounds[1]
    attributes['DayMet y min'] = bounds[0]
    attributes['DayMet x max'] = bounds[3]
    attributes['DayMet y max'] = bounds[2]
    attributes['DayMet start date'] = str(start)
    attributes['DayMet end date'] = str(end)
    return attributes


def convertToATS(dat):
    """Convert dictionary of Daymet datasets to daily average data in standard form.

    This:

    - takes tmin and tmax to compute a mean
    - splits rain and snow precip based on mean air temp
    - standardizes units and names for ATS

    """
    logging.info('Converting to ATS met input')

    # make missing values (-9999) as NaNs to do math while propagating NaNs
    for key in dat.keys():
        dat[key].data[dat[key].data == -9999] = np.nan

    # note that all of these can live in the same dataset since they
    # share the same profile/times
    profile = dat['tmin'].profile
    times = dat['tmin'].times
    dout = watershed_workflow.datasets.Dataset(profile, times)

    mean_air_temp_c = (dat['tmin'].data + dat['tmax'].data) / 2.0
    dout['air temperature [K]'] = 273.15 + mean_air_temp_c  # K

    precip_ms = dat['prcp'].data / 1.e3 / 86400.  # mm/day --> m/s

    # note that shortwave radiation in daymet is averged over the unit daylength, not per unit day.
    dout['incoming shortwave radiation [W m^-2]'] = dat['srad'].data * dat[
        'dayl'].data / 86400  # Wm2
    dout['vapor pressure air [Pa]'] = dat['vp']  # Pa
    dout['precipitation rain [m s^-1]'] = np.where(mean_air_temp_c >= 0, precip_ms, 0)
    dout['precipitation snow [m SWE s^-1]'] = np.where(mean_air_temp_c < 0, precip_ms, 0)
    return dout
