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
    tz_offset: int
        Time zone offset.
    lat : int
        Latitude in decimal degrees.
    lon : int
        Longitude in decimal degrees.        

    Returns
    -------
    solar_dec : int
        Solar declination.
    solar_altitude : int
        Altitude.
    sza : int
        Solar zenith angle.
    solar_azimuth_ini : int
        Initial estimate of azimuth.        

    Note this finds and downloads files as needed.
    """

    # For test -----------------------------
    doy = 200
    hour = 12
    tz_offset = 0
    lat = 35.9925
    lon = -79.0460
    #-----------------------------------------

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
