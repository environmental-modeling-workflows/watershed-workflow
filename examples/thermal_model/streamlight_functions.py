import numpy as np

'''The following functions are a python implementation 
of the code StreamLight (https://github.com/psavoy/StreamLight).
The function summarize the following references:

Savoy, P., Bernhardt, E., Kirk, L., Cohen, M. J., & Heffernan, J. B. (2021). 
    A seasonally dynamic model of light at the stream surface. Freshwater Science, 
    40(April), 000–000. https://doi.org/10.1086/714270

Savoy, P., & Harvey, J. W. (2021). Predicting light regime controls on primary 
    productivity across CONUS river networks. Geophysical Research Letters, 48, 
    e2020GL092149. https://doi.org/10.1029/2020GL092149
'''

def stream_light(lat, lon, channel_azimuth, bottom_width,bank_height,bank_slope,water_depth,tree_height,overhang,overhang_height, doy, hour, tz_offset, x_LAD=1): 
    """This function combines the SHADE2 model (Li et al. 2012) 
    and Campbell & Norman (1998) radiative transfer model. 

    Parameters
    ----------
    lat : float
        Latitude [decimal degrees].
    lon : float
        Longitude [decimal degrees].        
    channel_azimuth : float
        Channel azimuth [decimal degrees].
    bottom_width: float
        Channel width at the water-sediment interface [m].
    bank_height: float
        Bank height [m].
    bank_slope: float
        Bank slope [-].
    water_depth: float
        Water depth [m].        
    tree_height: float
        Tree height [m].
    overhang: float
        Effectively max canopy radius [m].
    overhang_height: float
        Height of the maximum canopy overhang (height at max canopy radius) [m].
    doy : int
        Day of the year.
    hour: int
        Hour of the day.
    tz_offset: int
        Time zone offset.        
    x_LAD: float
        Leaf angle distribution, default = 1 [-].

    Returns
    -------
    PAR_surface : float
        predicted light at the stream surface [umol m-2 s-1].      
    """
    # -------------------------------------------------------------
    # Defining solar geometry
    # -------------------------------------------------------------

    solar_dec, solar_altitude, sza, solar_azimuth_ini = solar_geo_calc(
        doy, hour, tz_offset, lat, lon)

    ## Generate a logical index of night and day. Night = SZA > 90
    day_index = sza <= (np.pi * 0.5)
    night_index = sza > (np.pi * 0.5)    

    # -------------------------------------------------------------
    # Predicting transmission of light through the canopy
    # -------------------------------------------------------------

    driver_file[day_index, "PAR_bc"] <- RT_CN_1998(
      driver_file = driver_file[day_index, ],
      solar_geo = solar_geo[day_index, ],
      x_LAD = x_LAD
    )


def solar_geo_calc(doy, hour, tz_offset, lat, lon):
    """This function calculates solar declination, altitude,
    zenith angle, and an initial estimate of azimuth. This initial estimate
    of solar azimuth is passed to the solar_c function where it is adjusted
    based on latitude and the solar declination angle. 

    Parameters
    ----------
    doy : int
        Day of the year.
    hour: int
        Hour of the day.
    tz_offset: float
        Time zone offset.
    lat : float
        Latitude [decimal degrees].
    lon : float
        Longitude [decimal degrees].        

    Returns
    -------
    solar_dec : float
        Solar declination.
    solar_altitude : float
        Altitude.
    sza : float
        Solar zenith angle [decimal degrees].
    solar_azimuth_ini : float
        Initial estimate of azimuth [decimal degrees].        
    """

    # # For test -----------------------------
    # doy = 200
    # hour = 12
    # tz_offset = 0
    # lat = 35.9925
    # lon = -79.0460
    # #-----------------------------------------

    jdate = (doy - 1) + (hour / 24) # numerical day (Julian date)

    # Defining solar geometry
    
    ## Solar declination
    solar_dec = 23.45 * ((np.pi) / 180) * np.sin(((2 * np.pi) * (jdate + 284)) / 365.25)

    ## Calculating true solar time
    ### Mean solar time
    mst = jdate + ((lon - tz_offset * 15) / 361)

    ### Equation of time
    b =(np.pi / 182) * (jdate - 81)
    eot = ((9.87 * np.sin(2 * b)) - (7.53 * np.cos(b)) - (1.5 * np.sin(b))) / 1440

    ### True solar time
    tst = mst + eot

    ### This is an adjustment from the Li (2006) code which deals with negative solar altitudes

    sin_solar_altitude = (np.sin(solar_dec) * np.sin(np.deg2rad(lat)) - np.cos(solar_dec) * \
        np.cos(np.deg2rad(lat)) * np.cos(2 * np.pi * tst))

    solar_altitude = np.arcsin(sin_solar_altitude)

    # Solar zenith angle
    sza = 0.5 * np.pi - solar_altitude

    # Initial estimate of the solar azimuth 
    solar_azimuth_ini = np.arccos((np.cos(solar_dec) * np.sin(2 * np.pi * tst)) / np.cos(solar_altitude))

    return solar_dec, solar_altitude, sza, solar_azimuth_ini
