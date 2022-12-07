import numpy as np
import itertools

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

def stream_light(lat, lon, channel_azimuth, bottom_width,bank_height,bank_slope,water_depth,tree_height,overhang,overhang_height, doy, hour, tz_offset, sw_inc, lai, x_LAD=1): 
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
    sw_inc : float
        Total incoming shortwave radiation [W m-2].  
    lai: float
        Leaf are index [-].               
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

    # -------------------------------------------------------------
    # Predicting transmission of light through the canopy
    # -------------------------------------------------------------

    par_bc = rt_cn_1998(doy, sza, solar_altitude, sw_inc, lai, x_LAD)

    #-------------------------------------------------
    #Running the SHADE2 model
    #-------------------------------------------------






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
    solar_dec_ini : float
        Solar declination (initial estimate).
    solar_altitude_ini : float
        Altitude (initial estimate).
    sza : float
        Solar zenith angle (initial estimate) [decimal degrees].
    solar_azimuth_ini : float
        Initial estimate of azimuth (initial estimate) [decimal degrees].        
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
    solar_dec_ini = 23.45 * ((np.pi) / 180) * np.sin(((2 * np.pi) * (jdate + 284)) / 365.25)

    ## Calculating true solar time
    ### Mean solar time
    mst = jdate + ((lon - tz_offset * 15) / 361)

    ### Equation of time
    b =(np.pi / 182) * (jdate - 81)
    eot = ((9.87 * np.sin(2 * b)) - (7.53 * np.cos(b)) - (1.5 * np.sin(b))) / 1440

    ### True solar time
    tst = mst + eot

    ### This is an adjustment from the Li (2006) code which deals with negative solar altitudes

    sin_solar_altitude = (np.sin(solar_dec_ini) * np.sin(np.deg2rad(lat)) - np.cos(solar_dec_ini) * \
        np.cos(np.deg2rad(lat)) * np.cos(2 * np.pi * tst))

    solar_altitude_ini = np.arcsin(sin_solar_altitude)

    # Solar zenith angle
    sza_ini = 0.5 * np.pi - solar_altitude_ini

    # Initial estimate of the solar azimuth 
    solar_azimuth_ini = np.arccos((np.cos(solar_dec_ini) * np.sin(2 * np.pi * tst)) / np.cos(solar_altitude_ini))

    return solar_dec_ini, solar_altitude_ini, sza_ini, solar_azimuth_ini

def rt_cn_1998(doy, sza, solar_altitude, sw_inc, lai, x_LAD):
    """This function calculates below canopy PAR. Main references are
    1. Campbell & Norman (1998) An introduction to Environmental biophysics (abbr C&N (1998))
    2. Spitters et al. (1986) Separating the diffuse and direct component of global
        radiation and its implications for modeling canopy photosynthesis: Part I
        components of incoming radiation
    3. Goudriaan (1977) Crop micrometeorology: A simulation study. Pudoc, Wageningen, The Netherlands.

    Parameters
    ----------
    doy : int
        Day of the year.    
    solar_altitude : float
        Altitude.
    sza : float
        Solar zenith angle [decimal degrees].       
    sw_inc : float
        Total incoming shortwave radiation [W m-2].  
    lai: float
        Leaf are index [-].   
    x_LAD: float
        Leaf angle distribution [-].      
    Returns
    -------
    ppfd : float
        Total PPFD transmitted through the canopy [umol m-2 s-1].       
    """

    #-------------------------------------------------
    # Partitioning incoming shorwave radiation into beam and diffuse components
    # Following Spitters et al. (1986)
    #-------------------------------------------------
    
    ## Generate a logical index of night and day. Night = SZA > 90
    is_night = sza > (np.pi * 0.5)  

    ## Calculate the extra-terrestrial irradiance (Spitters et al. (1986) Eq. 1)
    # I.e., global incoming shortwave radiation [W m-2]]
    Qo = 1370 * np.sin(solar_altitude) * (1 + 0.033 * np.cos(np.deg2rad(360 * doy / 365)))

    ## The relationship between fraction diffuse and atmospheric transmission
        # Spitters et al. (1986) appendix
    atm_trns = sw_inc / Qo
    R = 0.847 - (1.61 * np.sin(solar_altitude)) + (1.04 * np.sin(solar_altitude) * np.sin(solar_altitude))
    K = (1.47 - R) / 1.66

    frac_diff = np.array([diffuse_calc(at, rr, kk) for at, rr, kk in zip(atm_trns, R, K)])

    ## Partition into diffuse and beam radiation
    rad_diff = frac_diff * sw_inc # Diffuse radiation [W m-2]
    rad_beam = sw_inc - rad_diff # Beam radiation [W m-2]

    #-------------------------------------------------
    #Partition diffuse and beam radiation into PAR following Goudriaan (1977)
    #-------------------------------------------------
    I_od = 0.5 * rad_diff
    I_ob = 0.5 * rad_beam

    #-------------------------------------------------
    # Calculating beam radiation transmitted through the canopy
    #-------------------------------------------------
    
    ## Calculate the ratio of projected area to hemi-surface area for an ellipsoid
    ## C&N (1998) Eq. 15.4 sensu Campbell (1986)

    kbe = np.sqrt((x_LAD**2) + (np.tan(sza))**2)/(x_LAD + (1.774 * ((x_LAD + 1.182)**(-0.733))))

    # Fraction of incident beam radiation penetrating the canopy
    # C&N (1998) Eq. 15.1 and leaf absorptivity as 0.8 (C&N (1998) pg. 255) as per Camp
    tau_b = np.exp(-np.sqrt(0.8) * kbe * lai)

    # Beam radiation transmitted through the canopy
    beam_trans = I_ob * tau_b

    #-------------------------------------------------
    # Calculating diffuse radiation transmitted through the canopy
    #-------------------------------------------------

    # Diffuse transmission coefficient for the canopy (C&N (1998) Eq. 15.5)
    tau_d = np.array([dt_calc(ll,x_LAD=1) for ll in lai])

    # Extinction coefficient for black leaves in diffuse radiation
    kd = -np.log(tau_d) / lai

    # Diffuse radiation transmitted through the canopy
    diff_trans = I_od * np.exp(-np.sqrt(0.8) * kd * lai)

    # Get the total light transmitted through the canopy

    beam_trans[is_night] = 0
    diff_trans[is_night] = 0
    transmitted = beam_trans + diff_trans
    ppfd = transmitted * 1/0.235 #Convert from W m-2 to umol m-2 s-1

    return ppfd

def diffuse_calc(atm_trns, R, K):
    """Spitters et al. (1986) Eqs. 20a-20d
    
        So = Extra-terrestrial irradiance on a plane parallel to the earth surface [J m-2 s-1]
    
        Sdf = Diffuse flux global radiation [J m-2 s-1]
    
        Sg = Global radiation (total irradiance at the earth surface) [J m-2 s-1]
    
        atm_trns = Sg/So
    
        fdiffuse = Sdf/Sg

    Parameters
    ----------
    atm_trns : float
        Atmospheric transmission [-].  
    R, K : float
        Parameters in the regression of diffuse share on transmission [-].  
    Returns
    -------
    fdiffuse : float
        Fraction diffused [-].    
    """
    fdiffuse = 1.0

    if ((atm_trns > 0.22) and (atm_trns <= 0.35)):
        fdiffuse = 1.0-(6.4 * (atm_trns - 0.22)*(atm_trns - 0.22))
    elif ((atm_trns > 0.35) and (atm_trns <= K)):
        fdiffuse = 1.47 - (1.66 * atm_trns)
    elif (atm_trns > K):
        fdiffuse = R
    return fdiffuse


def integ_func(angle, d_sza, x_LAD, lai):
    '''Function to calculate the integral in Eq. 4 of Savoy et al. (2021)
    
    Parameters
    ----------
    angle : float
        solar zenith angle [decimal degrees].
    d_sza : float
        differential of solar zenith angle [decimal degrees].       
    x_LAD: float
        Leaf angle distribution [-].   
    lai: float
        Leaf are index [-].  

    Returns
    -------

    '''
    return np.exp(-(np.sqrt((x_LAD**2) + (np.tan(angle))**2)/(x_LAD + (1.774 * \
    ((x_LAD + 1.182)**(-0.733))))) * lai) * np.sin(angle) * np.cos(angle) * d_sza

def dt_calc(lai,x_LAD=1):
    '''Function to calculate the diffuse transmission coefficient
    
    Parameters
    ----------
    lai: float
        Leaf are index [-].  
    x_LAD: float
        Leaf angle distribution [-].  
    Returns
    -------

    '''    
    # Create a sequence of angles to integrate over
    angle_seq = np.deg2rad(np.arange(0,90))

    # Numerical integration
    d_sza = (np.pi / 2) / len(angle_seq)

    #Diffuse transmission coefficient for the canopy (C&N (1998) Eq. 15.5)
    

    return (2 * sum(integ_func(angle_seq, d_sza, x_LAD, lai)))


def shade2(lat, lon, channel_azimuth, bottom_width,bank_height,bank_slope,water_depth,tree_height,overhang,overhang_height, doy, hour, tz_offset, solar_dec_ini, solar_azimuth_ini, solar_altitude, sw_inc, lai, x_LAD=1):
    """SHADE2 model from Li et al. (2012) Modeled riparian stream shading:
    Agreement with field measurements and sensitivity to riparian conditions

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
    solar_dec_ini : float
        Solar declination (initial estimate).
    solar_azimuth_ini : float
        Initial estimate of azimuth [decimal degrees].   
    solar_altitude : float
        Altitude.         
    sw_inc : float
        Total incoming shortwave radiation [W m-2].  
    lai: float
        Leaf are index [-].               
    x_LAD: float
        Leaf angle distribution, default = 1 [-].

    Returns
    -------
    perc_shade_veg : float
        Percent of the wetted width shaded by vegetation [%].   
    perc_shade_bank : float
        Percent of the wetted width shaded by the bank [%].            
    """
    #-------------------------------------------------
    # Defining solar geometry
    #-------------------------------------------------

    solar_azimuth = solar_c(lat, lon, doy, hour, tz_offset, solar_dec_ini, solar_azimuth_ini)
    #-------------------------------------------------
    # Taking the difference between the sun and stream azimuth (sun-stream)
    #-------------------------------------------------

    ## This must be handled correctly to determine if the shadow falls towards the river
    ## [sin(delta)>0] or towards the bank
    ## Eastern shading
    delta_prime = solar_azimuth - (channel_azimuth * np.pi / 180)
    delta_prime[delta_prime < 0] = np.pi + np.abs(delta_prime[delta_prime < 0] ) % (2 * np.pi) #PS 2019
    delta_east = delta_prime % (2 * np.pi)

    ## Western shading
    if (delta_east < np.pi):
        delta_west = delta_east + np.pi
    else:
        delta_west = delta_east - np.pi

    #-------------------------------------------------
    # Doing some housekeeping related to bankfull and wetted widths
    #-------------------------------------------------
    
    ## Calculate bankfull width
    bankfull_width = bottom_width + ((bank_height/bank_slope)*2)

    ## Not sure what to do here, setting water_depth > bank_height, = bank_height
    water_depth[water_depth > bank_height] = bank_height

    ## Calculate wetted width
    water_width = bottom_width + water_depth*(1/bank_slope + 1/bank_slope)

    ## Not sure what to do here, setting widths > bankfull, = bankfull
    water_width[water_width > bankfull_width] <- bankfull_width

    #-------------------------------------------------
    # Calculate the length of shading for each bank
    #-------------------------------------------------
    
    ## Calculating shade from the "eastern" bank
    # eastern_shade_length <- matrix(ncol = 2,
    # shade_calc(
    #     delta = delta_east,
    #     solar_altitude = solar_altitude,
    #     bottom_width = bottom_width,
    #     BH = BH,
    #     BS = BS,
    #     WL = WL,
    #     TH = TH,
    #     overhang = overhang,
    #     overhang_height = overhang_height
    # )
    # )

    # east_bank_shade_length <- eastern_shade_length[, 1]
    # east_veg_shade_length <- eastern_shade_length[, 2] #- eastern_shade[, 1] #PS 7/9/2018

    # ## Calculating shade from the "western" bank
    # western_shade_length <- matrix(ncol = 2,
    # shade_calc(
    #     delta = delta_west,
    #     solar_altitude = solar_altitude,
    #     bottom_width = bottom_width,
    #     BH = BH,
    #     BS = BS,
    #     WL = WL,
    #     TH = TH,
    #     overhang = overhang,
    #     overhang_height = overhang_height
    # )
    # )

    # west_bank_shade_length <- western_shade_length[, 1]
    # west_veg_shade_length <- western_shade_length[, 2] #- western_shade[, 1] #PS 7/9/2018

    #-------------------------------------------------
    # Calculate the total length of bank shading
    #-------------------------------------------------
    
    ## Calculate the total length of bank shading
    total_bank_shade_length = east_bank_shade_length + west_bank_shade_length

    #Generate a logical index where the length of bank shading is longer than wetted width
    reset_bank_max_index = total_bank_shade_length > water_width

    ## If total bank shade length is longer than wetted width, set to wetted width
    total_bank_shade_length[reset_bank_max_index] = water_width #PS 2021

    #-------------------------------------------------
    # Calculate the total length of vegetation shading
    #-------------------------------------------------

    ## Calculate the total length of vegetation shading
    total_veg_shade_length = east_veg_shade_length + west_veg_shade_length

    ## Generate a logical index where the length of vegetation shading is longer than wetted width
    reset_veg_max_index = total_veg_shade_length > water_width

    ## If total vegetation shade length is longer than wetted width, set to wetted width
    total_veg_shade_length[total_veg_shade_length > water_width] = water_width #PS 2021

    #-------------------------------------------------
    #Calculating the percentage of water that is shaded
    #-------------------------------------------------
    perc_shade_bank = (total_bank_shade_length) / water_width
    perc_shade_bank[perc_shade_bank > 1] = 1

    perc_shade_veg = (total_veg_shade_length - total_bank_shade_length) / water_width
    perc_shade_veg[perc_shade_veg > 1] = 1

    return perc_shade_veg, perc_shade_bank


def solar_c(lat, lon, doy, hour, tz_offset, solar_dec_ini, solar_azimuth_ini):
    '''Calculates solar geometry for use in the SHADE2 model

    Parameters
    ----------
    lat : float
        Latitude [decimal degrees].
    lon : float
        Longitude [decimal degrees].  
    doy : int
        Day of the year.
    hour: int
        Hour of the day.
    tz_offset: float
        Time zone offset.   
    solar_dec_ini : float
        Solar declination (initial estimate).
    solar_azimuth_ini : float
        Initial estimate of azimuth (initial estimate) [decimal degrees].  

    Returns
    -------

    solar_azimuth : float
        Corrected estimate of azimuth (initial estimate) [decimal degrees].
    '''

    # Initialize the adjusted azimuth
    solar_azimuth = np.nan*np.ones_like(solar_azimuth_ini)

    # When latitude is > solar declination additional considerations are required to
    # determine the correct azimuth
    # Generate a logical index where latitude is greater than solar declination
    lat_greater = np.deg2rad(lat) > solar_dec_ini

    # Add a small amount of time (1 minute) and recalculate azimuth
    azimuth_tmp = azimuth_adj(lat, lon, doy, hour, tz_offset)

    # Generate a logical index where azimuth_tmp is greater than the initial estimate
    az_tmp_greater = azimuth_tmp > solar_azimuth_ini

    # Generate a logical index where both Lat > solar_dec & azimuth_tmp > azimuth
    add_az = lat_greater and az_tmp_greater
    solar_azimuth[add_az] =  (np.pi / 2) + solar_azimuth_ini[add_az]

    sub_az = lat_greater and (not az_tmp_greater)
    solar_azimuth[sub_az] =  (np.pi / 2) - solar_azimuth_ini[sub_az]

    # When Latitude is < solar declination all angles are 90 - azimuth
    solar_azimuth[not lat_greater] =  (np.pi / 2) - solar_azimuth_ini[not lat_greater]

    return solar_azimuth

def azimuth_adj(lat, lon, doy, hour, tz_offset):
    '''Helper function for determining the correct solar azimuth
    This function feeds into the solar_c function and is used to
    help determine the correct solar azimuth for locations where latitude is
    greater than the solar declination angle.'''

    # Add a minute to the hour
    _, _, _, solar_azimuth_adj = solar_geo_calc(doy, hour + (1 / 60 / 24), tz_offset, lat, lon)
    return solar_azimuth_adj


def shade_calc(delta, solar_altitude, bottom_width, bank_height, bank_slope, water_depth, tree_height, overhang, overhang_height):
    '''Calculating the percent of the wetted width shaded by banks and vegetation

    Parameters
    ----------
    delta : float
        Difference between the sun and stream azimuth (sun-stream)[decimal degrees].
    solar_altitude : float
        Altitude.    
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

    Returns
    -------
    stream_shade_bank : float
        Wetted width shaded by the bank [m].   
    stream_shade_veg_max : float
        Wetted width shaded by the vegetation [m].       
    '''

    #-------------------------------------------------
    # Doing some housekeeping related to bankfull and wetted widths
    # This is redundant from SHADE2.R, but leaving here for now while testing
    #-------------------------------------------------
    
    # Calculate bankfull width
    bankfull_width = bottom_width + ((bank_height/bank_slope)*2)

    # Setting water_depth > bank_height, = bank_height
    water_depth[water_depth > bank_height] = bank_height

    #Calculate wetted width
    water_width = bottom_width + water_depth*(1/bank_slope + 1/bank_slope)

    #Setting widths > bankfull, = bankfull
    water_width[water_width > bankfull_width] = bankfull_width

    #-------------------------------------------------
    # Calculating the shading produced by the bank
    #-------------------------------------------------

    # Calculating the length of the shadow perpendicular to the bank produced by the bank
    bank_shadow_length = (1 / np.tan(solar_altitude)) * (bank_height - water_depth) * np.sin(delta)

    # Finding the amount of exposed bank in the horizontal direction
    exposed_bank = (bank_height - water_depth) / bank_slope

    # From PS: if(BH - WL <= 0 | BS == 0) exposed_bank <- 0 #P.S. , commented this out because
    # I think I assumed that this couldn't be negative even if its confusing to be so

    # Finding how much shade falls on the surface of the water
    stream_shade_bank = bank_shadow_length - exposed_bank
    stream_shade_bank[stream_shade_bank < 0] = 0

    #-------------------------------------------------
    #Calculating the shading produced by the Vegetation
    #-------------------------------------------------

    #From top of the tree
    stream_shade_top = (1 / np.tan(solar_altitude)) * (tree_height + bank_height - water_depth) * np.sin(delta) - exposed_bank
    stream_shade_top[stream_shade_top < 0] = 0

    #From the overhang
    stream_shade_overhang =(1 / np.tan(solar_altitude)) * (overhang_height + bank_height - water_depth)* np.sin(delta) + overhang - exposed_bank
    stream_shade_overhang[stream_shade_overhang < 0] = 0

    #Selecting the maximum and minimum
    stream_shade_top.reshape((1,len(stream_shade_top)))
    stream_shade_overhang.reshape((1,len(stream_shade_overhang)))

    # Get max(shade from top, shade from overhang)
    # Note from PS: "here I take a departure from the r_shade matlab code. For some reason the code
    # Takes the maximum - min shadow length, but in the paper text it clearly states max
    # See pg 14 Li et al. (2012)"

    veg_shade_bound = np.column_stack((stream_shade_top, stream_shade_overhang))
    stream_shade_veg_max = np.amax(veg_shade_bound, axis = 0)

    # If the maximum shadow length is longer than the wetted width, set to width
    stream_shade_veg_max[stream_shade_veg_max > water_width] = water_width

    return stream_shade_bank, stream_shade_veg_max