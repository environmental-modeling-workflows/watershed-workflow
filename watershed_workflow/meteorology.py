"""Manipulate DayMet data structures.

DayMet is downloaded in box mode based on watershed bounds, then it can be converted to
hdf5 files that models can read.
"""
from typing import Tuple

import logging
import numpy as np
import xarray as xr
import datetime

import watershed_workflow.data

def allocatePrecipitation(precip: xr.DataArray, air_temp: xr.DataArray,
                          transition_temperature: float) -> Tuple[xr.DataArray, xr.DataArray]:
    """Allocates precipitation between rain and snow based on temperature.

    Parameters
    ----------
    precip : xr.DataArray
        Total precipitation data.
    air_temp : xr.DataArray
        Air temperature data.
    transition_temperature : float
        Temperature threshold for rain/snow transition. If < 100, assumed
        to be in Celsius; otherwise Kelvin.

    Returns
    -------
    rain : xr.DataArray
        Rain precipitation (when temp >= transition_temperature).
    snow : xr.DataArray
        Snow precipitation (when temp < transition_temperature).

    """
    if transition_temperature < 100:
        tt_K = transition_temperature + 273.15
    else:
        tt_K = transition_temperature

    rain = xr.where(air_temp >= tt_K, precip, 0)
    snow = xr.where(air_temp < tt_K, precip, 0)
    return rain, snow


def convertDayMetToATS(dat: xr.Dataset, transition_temperature: float = 0.) -> xr.Dataset:
    """Convert xarray.Dataset Daymet datasets to daily average data in standard form.

    This:
    - takes tmin and tmax to compute a mean
    - splits rain and snow precip based on mean air temp relative to transition_temperature [C]
    - standardizes units and names for ATS

    Parameters
    ----------
    dat : xr.Dataset
        Input Daymet dataset with variables: tmin, tmax, prcp, srad, dayl, vp.
    transition_temperature : float, optional
        Temperature threshold for rain/snow split in Celsius. Default is 0.

    Returns
    -------
    xr.Dataset
        Dataset with ATS-compatible variable names and units.

    """
    logging.info('Converting DayMet to ATS met input')

    # make missing values (-9999) as NaNs to do math while propagating NaNs
    for key in dat.keys():
        dat[key].data[dat[key].data == -9999] = np.nan

    # note that all of these can live in the same dataset since they
    # share the same coordinates/times
    dout = xr.Dataset(coords=dat.coords, attrs=dat.attrs.copy())

    mean_air_temp_c = (dat['tmin'] + dat['tmax']) / 2.0
    dout['air temperature [K]'] = 273.15 + mean_air_temp_c  # K

    precip_ms = dat['prcp'] / 1.e3 / 86400.  # mm/day --> m/s

    # note that shortwave radiation in daymet is averged over the unit daylength, not per unit day.
    dout['incoming shortwave radiation [W m^-2]'] = dat['srad'] * dat['dayl'] / 86400  # Wm2
    dout['vapor pressure air [Pa]'] = dat['vp']  # Pa
    dout['precipitation rain [m s^-1]'], dout['precipitation snow [m SWE s^-1]'] = \
        allocatePrecipitation(precip_ms, mean_air_temp_c, transition_temperature)
    return dout


def convertAORCToATS(dat: xr.Dataset,
                     transition_temperature: float = 0.,
                     n_hourly: int = 24,
                     remove_leap_day: bool = False) -> xr.Dataset:
    """Convert xarray.Dataset AORC datasets to standard ATS format output.

    - computes specific humidity and surface pressure to vapor pressure
    - computes total wind speed from component wind speeds
    - converts precip units to m/s
    - allocates precip to snow and rain based on transition temp

    Parameters
    ----------
    dat : xr.Dataset
      Input including AORC raw data.
    transition_temperature : float
      Temperature to transition from snow to rain [C].  Default is 0 C.
    n_hourly : int
      Convert data from 1-hourly to n_hourly to reduce data needs.
      Defaults to 24 hours (daily data).
    remove_leap_day : bool
      If True, removes day 366 any leap year (not Feb 30!).  Deafult
      is False.

    Returns
    -------
    xr.Dataset
      Dataset with ATS-standard names/units met forcing.
    
    """
    logging.info('Converting AORC to ATS met input')

    # note that all of these can live in the same dataset since they
    # share the same coordinates/times
    dout = xr.Dataset(coords=dat.coords, attrs=dat.attrs.copy())

    dout['air temperature [K]'] = dat['TMP_2maboveground']
    dout['incoming shortwave radiation [W m^-2]'] = dat['DSWRF_surface']
    dout['incoming longwave radiation [W m^-2]'] = dat['DLWRF_surface']
    dout['vapor pressure air [Pa]'] = dat['SPFH_2maboveground'] * dat['PRES_surface'] \
        / (0.622 + dat['SPFH_2maboveground'])

    dout.attrs['wind speed reference height [m]'] = 10.
    dout['wind speed [m s^-1]'] = np.sqrt(
        np.pow(dat['UGRD_10maboveground'], 2) + np.pow(dat['VGRD_10maboveground'], 2))

    # convert mm --> m, hour --> s to get m/s
    dout['precipitation total [m s^-1]'] = dat['APCP_surface'] / 1000 / 3600

    # allocate precip
    dout['precipitation rain [m s^-1]'], dout['precipitation snow [m SWE s^-1]'] = \
        allocatePrecipitation(dout['precipitation total [m s^-1]'],
                              dout['air temperature [K]'], transition_temperature)


    # convert times to standard time convention and remove leap day
    dout['time'] = watershed_workflow.data.convertTimesToCFTime(dout['time'].values)
    if remove_leap_day:
        dout = watershed_workflow.data.filterLeapDay_DataFrame(dout)

    # group to n-hourly and take the mean
    dout = dout.resample(time=datetime.timedelta(hours=n_hourly)).mean()
    return dout
