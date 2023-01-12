import numpy as np
import itertools
import copy

class StreamLight:
    """This class is a Python implementation 
    of the code StreamLight (https://github.com/psavoy/StreamLight).
    Minor modifications have been made to improve numerical performance.

    StreamLight is summarized in the following references:

    Savoy, P., Bernhardt, E., Kirk, L., Cohen, M. J., & Heffernan, J. B. (2021). 
        A seasonally dynamic model of light at the stream surface. Freshwater Science, 
        40(April), 000–000. https://doi.org/10.1086/714270

    Savoy, P., & Harvey, J. W. (2021). Predicting light regime controls on primary 
        productivity across CONUS river networks. Geophysical Research Letters, 48, 
        e2020GL092149. https://doi.org/10.1029/2020GL092149
    """

    def __init__(self):
        """StreamLight requires channel properties and energy drivers 
        to calculate photosynthetically active radiation (PAR)
        """
        # Channel properties
        self.channel_properties = dict.fromkeys([
            'lat', 'lon', 'channel_azimuth', 'bottom_width', 'bank_height', 
            'bank_slope', 'water_depth', 'tree_height', 'overhang', 
            'overhang_height', 'x_LAD'
        ])
        
        # Energy drivers
        self.energy_drivers = dict.fromkeys([
            'doy', 'hour', 'tz_offset', 'sw_inc', 'lai'
        ])

        # Solar Angles
        self.solar_angles = dict.fromkeys([
            'solar_dec', 'solar_altitude', 'sza', 'solar_azimuth'
        ])

        # Energy response
        self.energy_response = dict.fromkeys([
            'ppfd'
        ])

    def set_channel_properties(self,lat, lon, channel_azimuth, bottom_width,
        bank_height,bank_slope,water_depth,tree_height,overhang,overhang_height, x_LAD = 1):
        """Set the channel properties. 

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
        x_LAD: float
            Leaf angle distribution, default = 1 [-]."""
        self.channel_properties['lat'] = lat
        self.channel_properties['lon'] = lon
        self.channel_properties['channel_azimuth'] = channel_azimuth
        self.channel_properties['bottom_width'] = bottom_width
        self.channel_properties['bank_height'] = bank_height
        self.channel_properties['bank_slope'] = bank_slope
        self.channel_properties['water_depth'] = water_depth
        self.channel_properties['tree_height'] = tree_height
        self.channel_properties['overhang'] = overhang
        self.channel_properties['overhang_height'] = overhang_height
        self.channel_properties['x_LAD'] = x_LAD

    def set_energy_drivers(self, doy, hour, tz_offset, sw_inc, lai):
        """Set the energy drivers. 

        Parameters
        ----------
        doy : int
            Day of the year.
        hour: int
            Hour of the day.
        tz_offset: int
            Time zone offset.    
        sw_inc : float
            Total incoming shortwave radiation [W m-2].  
        lai: float
            Leaf are index [-]."""
        self.energy_drivers['doy'] = doy
        self.energy_drivers['hour'] = hour
        self.energy_drivers['tz_offset'] = tz_offset
        self.energy_drivers['sw_inc'] = sw_inc
        self.energy_drivers['lai'] = lai

    def get_salar_angles(self):
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
            Solar declination (initial estimate) [radians]. The angle made between the ray of the Sun, extended to the centre of the Earth, and the equatorial plane of the Earth.
        solar_altitude_ini : float
            Altitude (initial estimate) [radians]. The angular distance between the rays of Sun and the horizon of the Earth. 
        sza : float
            Solar zenith angle (initial estimate) [radians]. The angle between the sun's rays and the vertical direction. It is the complement to the solar altitude.
        solar_azimuth_ini : float
            Initial estimate of azimuth (initial estimate) [radians]. The angle is the azimuth (horizontal angle with respect to north) of the Sun's position.       
        """
        jdate = (self.energy_drivers['doy'] - 1) + (self.energy_drivers['hour'] / 24) # numerical day (Julian date)

        # Defining solar geometry
        
        ## Solar declination
        solar_dec_ini = 23.45 * ((np.pi) / 180) * np.sin(((2 * np.pi) * (jdate + 284)) / 365.25)

        ## Calculating true solar time
        ### Mean solar time
        mst = jdate + ((self.channel_properties['lon'] - self.energy_drivers['tz_offset'] * 15) / 361)

        ### Equation of time
        b =(np.pi / 182) * (jdate - 81)
        eot = ((9.87 * np.sin(2 * b)) - (7.53 * np.cos(b)) - (1.5 * np.sin(b))) / 1440

        ### True solar time
        tst = mst + eot

        ### This is an adjustment from the Li (2006) code which deals with negative solar altitudes

        sin_solar_altitude = (np.sin(solar_dec_ini) * np.sin(np.deg2rad(self.channel_properties['lat'])) - np.cos(solar_dec_ini) * \
            np.cos(np.deg2rad(self.channel_properties['lat'])) * np.cos(2 * np.pi * tst))

        solar_altitude_ini = np.arcsin(sin_solar_altitude)

        # Solar zenith angle
        sza_ini = 0.5 * np.pi - solar_altitude_ini

        # Initial estimate of the solar azimuth 
        solar_azimuth_ini = np.arccos((np.cos(solar_dec_ini) * np.sin(2 * np.pi * tst)) / np.cos(solar_altitude_ini))

        self.solar_angles['solar_dec'] = solar_dec_ini
        self.solar_angles['solar_altitude'] = solar_altitude_ini
        self.solar_angles['sza'] = sza_ini
        self.solar_angles['solar_azimuth'] = solar_azimuth_ini


    def rt_cn_1998(self):
        """This function calculates below canopy PAR. Main references are
            1) Campbell & Norman (1998) An introduction to Environmental biophysics (abbr C&N (1998))
            2) Spitters et al. (1986) Separating the diffuse and direct component of global radiation and its implications for modeling canopy photosynthesis: Part I components of incoming radiation
            3) Goudriaan (1977) Crop micrometeorology: A simulation study. Pudoc, Wageningen, The Netherlands.

        Parameters
        ----------
        doy : int
            Day of the year.    
        solar_altitude : float
            Altitude [radians].
        sza : float
            Solar zenith angle [radians].       
        sw_inc : float
            Total incoming shortwave radiation [W m-2].  
        lai: float
            Leaf are index [-].   
        x_LAD: float
            Leaf angle distribution [-].      
        Returns
        -------
        diff_trans : float
            Diffuse energy transmitted through the canopy [W m-2].
        beam_trans : float
            Beam energy transmitted through the canopy [W m-2].
        total_trans : float
            Total energy transmitted through the canopy [W m-2].
        ppfd : float
            Total PPFD transmitted through the canopy [umol m-2 s-1]. The StreamLight model estimates photosynthetically active radiation (PAR), which relates to the conversion of solar energy by autotrophs via photosynthesis. Estimates are expressed in terms of the quanta of light in PAR (umol m-2 s-1), otherwise referred to as the photosynthetic photon flux density (PPFD).    
        """
        #-------------------------------------------------
        # Partitioning incoming shorwave radiation into beam and diffuse components
        # Following Spitters et al. (1986)
        #-------------------------------------------------
        
        ## Generate a logical index of night and day. Night = SZA > 90
        is_night = self.salar_angles['sza'] > (np.pi * 0.5)  

        ## Calculate the extra-terrestrial irradiance (Spitters et al. (1986) Eq. 1)
        # I.e., global incoming shortwave radiation [W m-2]]
        Qo = 1370 * np.sin(self.salar_angles['solar_altitude']) * (1 + 0.033 * np.cos(np.deg2rad(360 * self.energy_drivers['doy'] / 365)))

        ## The relationship between fraction diffuse and atmospheric transmission
            # Spitters et al. (1986) appendix
        atm_trns = self.energy_drivers['sw_inc'] / Qo
        R = 0.847 - (1.61 * np.sin(self.salar_angles['solar_altitude'])) + (1.04 * np.sin(self.salar_angles['solar_altitude']) * np.sin(self.salar_angles['solar_altitude']))
        K = (1.47 - R) / 1.66

        frac_diff = np.array([_diffuse_calc(at, rr, kk) for at, rr, kk in zip(atm_trns, R, K)])

        ## Partition into diffuse and beam radiation
        rad_diff = frac_diff * self.energy_drivers['sw_inc'] # Diffuse radiation [W m-2]
        rad_beam = self.energy_drivers['sw_inc'] - rad_diff # Beam radiation [W m-2]

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

        kbe = np.sqrt((self.channel_properties['x_LAD']**2) + (np.tan(self.salar_angles['sza']))**2)/(self.channel_properties['x_LAD'] + (1.774 * ((self.channel_properties['x_LAD'] + 1.182)**(-0.733))))

        # Fraction of incident beam radiation penetrating the canopy
        # C&N (1998) Eq. 15.1 and leaf absorptivity as 0.8 (C&N (1998) pg. 255) as per Camp
        tau_b = np.exp(-np.sqrt(0.8) * kbe * self.energy_drivers['lai'])

        # Beam radiation transmitted through the canopy
        beam_trans = I_ob * tau_b

        #-------------------------------------------------
        # Calculating diffuse radiation transmitted through the canopy
        #-------------------------------------------------

        # Diffuse transmission coefficient for the canopy (C&N (1998) Eq. 15.5)
        tau_d = np.array([_dt_calc(ll,x_LAD=1) for ll in self.energy_drivers['lai']])

        # Extinction coefficient for black leaves in diffuse radiation
        kd = -np.log(tau_d) / self.energy_drivers['lai']

        # Diffuse radiation transmitted through the canopy
        diff_trans = I_od * np.exp(-np.sqrt(0.8) * kd * self.energy_drivers['lai'])

        # Get the total light transmitted through the canopy

        beam_trans[is_night] = 0
        diff_trans[is_night] = 0
        transmitted = beam_trans + diff_trans
        ppfd = transmitted * 1/0.235 #Convert from W m-2 to umol m-2 s-1

        self.energy_response['diff_trans'] = diff_trans
        self.energy_response['beam_trans'] = beam_trans
        self.energy_response['total_trans'] = transmitted
        self.energy_response['ppfd'] = ppfd



    def _diffuse_calc(atm_trns, R, K):
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


    def _integ_func(angle, d_sza, x_LAD, lai):
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

    def _dt_calc(lai,x_LAD=1):
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
        

        return (2 * sum(_integ_func(angle_seq, d_sza, x_LAD, lai)))

