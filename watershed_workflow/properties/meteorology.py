"""Manipulate meteorological data structures.

Data are downloaded in box mode based on watershed bounds, then converted to
formats that models can read.
"""
from typing import List, Tuple

import logging
import numpy as np
import xarray as xr
import datetime

import watershed_workflow.utils.data

__all__ = [
    'allocatePrecipitation',
    'convertDayMetToATS',
    'convertAORCToATS',
    'convertAORCToELM',
    'computeTypicalYear',
]


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
    dout['time'] = watershed_workflow.utils.data.convertTimesToCFTime(dout['time'].values)
    if remove_leap_day:
        dout = watershed_workflow.utils.data.filterLeapDay(dout)

    return dout


def convertAORCToELM(dat: xr.Dataset) -> xr.Dataset:
    """Convert raw AORC xarray.Dataset to ELM CPL_BYPASS variable names and units.

    AORC → ELM variable mapping:

    ========================== ============= ======= ========= =======
    AORC variable              AORC units    ELM var ELM units Notes
    ========================== ============= ======= ========= =======
    TMP_2maboveground          K             TBOT    K         no conversion
    SPFH_2maboveground         g/g           QBOT    kg/kg     1 g/g = 1 kg/kg
    UGRD_10maboveground +
    VGRD_10maboveground        m/s           WIND    m/s       magnitude
    DLWRF_surface              W/m²          FLDS    W/m²
    DSWRF_surface              W/m²          FSDS    W/m²
    PRES_surface               Pa            PSRF    Pa
    APCP_surface               mm/hr         PRECTmms mm/s     ÷ 3600
    ========================== ============= ======= ========= =======

    Parameters
    ----------
    dat : xr.Dataset
        Raw AORC dataset as returned by ``ManagerAORC.getDataset()``.

    Returns
    -------
    xr.Dataset
        Dataset with ELM CPL_BYPASS variable names and units.
    """
    logging.info('Converting AORC to ELM CPL_BYPASS met input')

    dout = xr.Dataset(coords=dat.coords, attrs=dat.attrs.copy())

    dout['TBOT'] = dat['TMP_2maboveground']
    dout['TBOT'].attrs = {'units': 'K', 'long_name': 'air temperature at 2 m'}

    # AORC SPFH is g/g which equals kg/kg — no numeric conversion needed
    dout['QBOT'] = dat['SPFH_2maboveground']
    dout['QBOT'].attrs = {'units': 'kg/kg', 'long_name': 'specific humidity at 2 m'}

    dout['WIND'] = np.sqrt(dat['UGRD_10maboveground']**2 + dat['VGRD_10maboveground']**2)
    dout['WIND'].attrs = {'units': 'm/s', 'long_name': 'wind speed at 10 m'}

    dout['FLDS'] = dat['DLWRF_surface']
    dout['FLDS'].attrs = {'units': 'W/m^2', 'long_name': 'incident longwave radiation'}

    dout['FSDS'] = dat['DSWRF_surface']
    dout['FSDS'].attrs = {'units': 'W/m^2', 'long_name': 'incident shortwave radiation'}

    dout['PSRF'] = dat['PRES_surface']
    dout['PSRF'].attrs = {'units': 'Pa', 'long_name': 'surface pressure'}

    # AORC APCP is mm/hr; ELM CPL_BYPASS expects mm/s
    dout['PRECTmms'] = dat['APCP_surface'] / 3600.0
    dout['PRECTmms'].attrs = {'units': 'mm/s', 'long_name': 'total precipitation rate'}

    return dout


def computeTypicalYear(dat: xr.Dataset,
                       repeat_nyears: int,
                       precip_vars: List[str] = None,
                       temp_var: str = 'air temperature [K]',
                       combine_precip: bool = True,
                       transition_temperature: float = 0.,
                       time_dim: str = 'time',
                       smooth_kwargs: dict = None,
                       ) -> xr.Dataset:
    """Compute a typical year from a meteorology dataset.

    Works at any timestep resolution (hourly, 3-hourly, daily, etc.) — the
    timestep is detected automatically from the data.  The function:

    - Averages all non-precipitation variables across years in each
      within-year bin (preserving the diurnal cycle for sub-daily data).
    - Identifies the median-total-precipitation year and uses its raw values
      for precipitation, avoiding the artificial drizzle that averaging
      would produce in mostly-zero bins.
    - Optionally combines multiple precipitation variables (e.g. rain + snow)
      before median-year selection, then re-splits them via
      ``allocatePrecipitation`` at the end.

    Parameters
    ----------
    dat : xr.Dataset
        Input meteorology dataset with cftime noleap time coordinates.
        May be raw AORC, ATS-format, or ELM CPL_BYPASS format.
    repeat_nyears : int
        Number of times to repeat the typical-year pattern in the output.
        The output start date is ``dat[time_dim][0] - repeat_nyears years``.
    precip_vars : list of str, optional
        Variable names to treat as precipitation (median-year, not averaged).
        Default: ``['precipitation rain [m s^-1]', 'precipitation snow [m SWE s^-1]']``
        (ATS format).  Pass ``['PRECTmms']`` for ELM CPL_BYPASS format, or
        any list of raw AORC precipitation variable names.
    temp_var : str, optional
        Variable name of air temperature, used only when ``combine_precip=True``
        to re-split combined precip into rain and snow via
        ``allocatePrecipitation``.  Default: ``'air temperature [K]'`` (ATS).
        Set to ``None`` to skip re-splitting when using ELM or AORC format.
    combine_precip : bool, optional
        If ``True`` (default) and ``len(precip_vars) > 1``, sum all precip
        variables before selecting the median year, then re-split into rain
        and snow via ``allocatePrecipitation``.  If ``False``, or if there
        is only one precip variable, use each variable's median-year values
        directly without re-splitting (the same median year is used for all).
    transition_temperature : float, optional
        Temperature threshold [°C] for rain/snow partitioning, passed to
        ``allocatePrecipitation``.  Only used when ``combine_precip=True``
        and ``temp_var`` is not ``None``.  Default: 0.
    time_dim : str, optional
        Name of the time dimension.  Default: ``'time'``.
    smooth_kwargs : dict, optional
        If provided, keyword arguments passed to
        ``watershed_workflow.utils.data.smoothTimeSeries`` to smooth
        non-precipitation variables before averaging.  Default: no smoothing.

    Returns
    -------
    xr.Dataset
        Dataset at the same timestep resolution as the input, containing
        ``repeat_nyears`` repetitions of the typical-year pattern.
    """
    if precip_vars is None:
        precip_vars = ['precipitation rain [m s^-1]', 'precipitation snow [m SWE s^-1]']

    logging.info('Computing a typical year.')

    # enforce noleap calendar
    dat = watershed_workflow.utils.data.filterLeapDay(dat, time_dim)

    # detect timestep and compute steps per year
    time_values = dat[time_dim].values
    if len(time_values) < 2:
        raise ValueError("Need at least two timesteps to detect interval.")
    dt_seconds = (time_values[1] - time_values[0]).total_seconds()
    steps_per_year = round(365 * 24 * 3600 / dt_seconds)

    # trim to whole years
    nwhole = dat.sizes[time_dim] // steps_per_year * steps_per_year
    dat = dat.isel({time_dim: slice(0, nwhole)})
    nyears = nwhole // steps_per_year

    # sum precip vars for median-year selection
    precip_total = sum(dat[v] for v in precip_vars)

    # drop precip vars and average the rest
    din = dat.drop_vars(precip_vars)
    if smooth_kwargs is not None:
        din = watershed_workflow.utils.data.smoothTimeSeries(din, time_dim=time_dim,
                                                             **smooth_kwargs)

    start_date = din[time_dim].values[0] - datetime.timedelta(seconds=steps_per_year * dt_seconds * repeat_nyears)
    dout = watershed_workflow.utils.data.computeAverageYear(din, start_date, repeat_nyears, time_dim)

    # identify median-total-precipitation year
    # (sort and take halfway point — np.median would interpolate for even N)
    block = xr.DataArray(
        np.arange(nwhole) // steps_per_year,
        dims=time_dim,
        name='year_block',
    )
    precip_blocks = precip_total.assign_coords(year_block=block)
    spatial_dims = [d for d in precip_total.dims if d != time_dim]
    annual_precip = precip_blocks.groupby('year_block').sum(dim=[time_dim] + spatial_dims)
    median_i = sorted(enumerate(annual_precip.values), key=lambda x: x[1])[nyears // 2][0]

    # extract the median year's values for each precip var
    median_slice = slice(median_i * steps_per_year, (median_i + 1) * steps_per_year)

    if combine_precip and len(precip_vars) > 1 and temp_var is not None:
        # combine → tile → re-split using averaged temperature
        typical_precip = precip_total.isel({time_dim: median_slice})
        tiled = xr.concat([typical_precip] * repeat_nyears, dim='repeat')
        tiled = tiled.stack(time_new=('repeat', time_dim)).drop_vars(time_dim).rename(time_new=time_dim)
        tiled[time_dim] = dout[time_dim]
        dout[precip_vars[0]], dout[precip_vars[1]] = \
            allocatePrecipitation(tiled, dout[temp_var], transition_temperature)
    else:
        # use each variable's median-year values directly
        for v in precip_vars:
            typical = dat[v].isel({time_dim: median_slice})
            tiled = xr.concat([typical] * repeat_nyears, dim='repeat')
            tiled = tiled.stack(time_new=('repeat', time_dim)).drop_vars(time_dim).rename(time_new=time_dim)
            # .stack() moves the stacked dims (repeat, time_dim) to the end,
            # reordering dims relative to typical/dat[v] (e.g. (time,lat,lon)
            # becomes (lat,lon,time)) -- restore the original order so this
            # variable's dims match every other variable in dout.
            tiled = tiled.transpose(*typical.dims)
            tiled[time_dim] = dout[time_dim]
            dout[v] = tiled

    return dout


