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
                     resample_interval: int = 1,
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

    return dout


def computeTypicalYear(dat: xr.Dataset,
                       repeat_nyears : int,
                       transition_temperature: float = 0.,
                       time_dim: str = 'time',
                       ) -> xr.Dataset:
    """Given an ATS-format meteorology dataset, this computes a typical year.

    - computes day-of-year averages of vapor pressure, air temp, radiation
    - computes median year of total precip and uses that year for total precip
    - allocates precip to snow and rain
    - replicates nyears times

    Parameters
    ----------
    dat : xr.Dataset
      ATS-format, daily averaged meterology data.
    repeat_nyears : int
      Number of years to replicate the data.  Note that first day of
      the returned dataset will be dat['time'][0] - repeat_nyears
    transition_temperature : float
      Temperature to transition from snow to rain [C].  Default is 0 C.

    Returns
    -------
    xr.Dataset
      Dataset with ATS-standard names/units met forcing for a typical year.

    """
    logging.info('Computing a typical year.')

    # must be done in noleap calendar
    dat = watershed_workflow.data.filterLeapDay(dat, time_dim)

    # must be done in an fixed number of whole years
    nwhole_years_in_days = dat.sizes[time_dim] // 365 * 365
    dat = dat.isel({time_dim: slice(0, nwhole_years_in_days)})

    # compute total precip
    precip = dat['precipitation rain [m s^-1]'] + dat['precipitation snow [m SWE s^-1]']

    # drop precip and compute doy-averaged quantities for remainder
    din = dat.drop_vars(['precipitation rain [m s^-1]', 'precipitation snow [m SWE s^-1]'])
    start_date = din[time_dim].values[0] - datetime.timedelta(days=365*repeat_nyears)
    dout = watershed_workflow.data.computeAverageYear(din, start_date, repeat_nyears, time_dim)

    # find the median precip year and insert this into dout
    # -- create a year-block based on the initial day
    block = xr.DataArray(
        np.arange(len(precip.time)) // 365,
        dims=time_dim,
        name="year_block"
    )
    precip_blocks = precip.assign_coords(year_block=block)

    # -- sum within each block over all days and spatial dims
    annual_precip = precip_blocks.groupby("year_block").sum(dim=(time_dim,"x","y"))

    # -- find the median...
    # note -- don't use np.median here... for even number of years it will not appear.  Instead, sort and take the halfway point
    median_i = sorted(((i,v) for (i,v) in enumerate(annual_precip)), key=lambda x : x[1])[len(annual_precip)//2][0]

    typical_precip = precip.isel({time_dim: slice(median_i * 365, (median_i+1)*365)})

    # repeat nyears times
    tiled = xr.concat([typical_precip] * repeat_nyears, dim="repeat")
    tiled = tiled.stack(time_new=("repeat", time_dim))
    tiled = tiled.drop_vars(time_dim)
    tiled = tiled.rename(time_new=time_dim)
    tiled[time_dim] = dout[time_dim]

    # allocate precip
    dout['precipitation rain [m s^-1]'], dout['precipitation snow [m SWE s^-1]'] = \
        allocatePrecipitation(tiled, dout['air temperature [K]'], transition_temperature)

    return dout
    


