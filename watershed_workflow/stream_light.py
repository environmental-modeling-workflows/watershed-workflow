import numpy as np
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
            'lat', 'lon', 'channel_azimuth', 'bottom_width', 'bank_height', 'bank_slope',
            'water_depth', 'tree_height', 'overhang', 'overhang_height', 'x_LAD'
        ])

        # Energy drivers
        self.energy_drivers = dict.fromkeys(['doy', 'hour', 'tz_offset', 'sw_inc', 'lai'])

        # Solar Angles
        self.solar_angles = dict.fromkeys(
            ['solar_dec', 'solar_altitude', 'sza', 'solar_azimuth', 'solar_azimuth_shade2'])

        # Energy response
        self.energy_response = dict.fromkeys([
            'rad_diff_PAR', 'rad_beam_PAR', 'rad_diff', 'rad_beam', 'diff_trans_PAR',
            'beam_trans_PAR', 'total_trans_PAR', 'diff_trans', 'beam_trans', 'total_trans',
            'total_trans_PAR_ppfd', 'par_surface_original', 'energy_diff_surface_PAR',
            'energy_beam_surface_PAR', 'energy_total_surface_PAR', 'energy_diff_surface',
            'energy_beam_surface', 'energy_total_surface', 'energy_total_surface_PAR_ppfd'
        ])

        # Shading response
        self.shading_response = dict.fromkeys(['fraction_shade_veg', 'fraction_shade_bank'])

    def set_channel_properties(self,
                               lat,
                               lon,
                               channel_azimuth,
                               bottom_width,
                               bank_height,
                               bank_slope,
                               water_depth,
                               tree_height,
                               overhang,
                               overhang_height,
                               x_LAD=1):
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

    def get_solar_angles(self):

        solar_dec, solar_altitude, sza, solar_azimuth = self._calc_solar_angles(
            self.energy_drivers['doy'], self.energy_drivers['hour'],
            self.energy_drivers['tz_offset'], self.channel_properties['lat'],
            self.channel_properties['lon'])

        self.solar_angles['solar_dec'] = solar_dec
        self.solar_angles['solar_altitude'] = solar_altitude
        self.solar_angles['sza'] = sza
        self.solar_angles['solar_azimuth'] = solar_azimuth

        self._correct_solar_azimuth()

    def _calc_solar_angles(self, doy, hour, tz_offset, lat, lon):
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
            Solar declination (initial estimate) [radians]. The angle made between the ray of the Sun, extended to the centre of the Earth, and the equatorial plane of the Earth.
        solar_altitude : float
            Altitude (initial estimate) [radians]. The angular distance between the rays of Sun and the horizon of the Earth. 
        sza : float
            Solar zenith angle (initial estimate) [radians]. The angle between the sun's rays and the vertical direction. It is the complement to the solar altitude.
        solar_azimuth : float
            Initial estimate of azimuth (initial estimate) [radians]. The angle is the azimuth (horizontal angle with respect to north) of the Sun's position.       
        """
        jdate = (doy-1) + (hour/24)  # numerical day (Julian date)

        # Defining solar geometry

        ## Solar declination
        solar_dec = 23.45 * ((np.pi) / 180) * np.sin(((2 * np.pi) * (jdate+284)) / 365.25)

        ## Calculating true solar time
        ### Mean solar time
        mst = jdate + ((lon - tz_offset*15) / 361)

        ### Equation of time
        b = (np.pi / 182) * (jdate-81)
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
        solar_azimuth = np.arccos(
            (np.cos(solar_dec) * np.sin(2 * np.pi * tst)) / np.cos(solar_altitude))

        return solar_dec, solar_altitude, sza, solar_azimuth

    def get_radiative_transfer_estimates_cn_1998(self):
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

        rad_diff_PAR : float
            Diffusive PAR incoming shortwave radiation [W m-2].
        rad_beam_PAR : float
            Beam PAR incoming shortwave radiation [W m-2].
        rad_diff : float
            Diffusive incoming shortwave radiation [W m-2].
        rad_beam : float
            Beam incoming shortwave radiation [W m-2].
        diff_trans_PAR : float
            Diffuse PAR energy transmitted through the canopy [W m-2].
        beam_trans_PAR : float
            Beam PAR energy transmitted through the canopy [W m-2].
        total_trans_PAR : float
            Total PAR energy transmitted through the canopy [W m-2].        
        diff_trans : float
            Diffuse energy transmitted through the canopy [W m-2].
        beam_trans : float
            Beam energy transmitted through the canopy [W m-2].
        total_trans : float
            Total energy transmitted through the canopy [W m-2].
        total_trans_PAR_ppfd : float
            Total PPFD transmitted through the canopy [umol m-2 s-1]. The StreamLight model estimates photosynthetically active radiation (PAR), which relates to the conversion of solar energy by autotrophs via photosynthesis. Estimates are expressed in terms of the quanta of light in PAR (umol m-2 s-1), otherwise referred to as the photosynthetic photon flux density (PPFD).    
        """
        #-------------------------------------------------
        # Partitioning incoming shorwave radiation into beam and diffuse components
        # Following Spitters et al. (1986)
        #-------------------------------------------------

        ## Generate a logical index of night and day. Night = SZA > 90
        is_night = self.solar_angles['sza'] > (np.pi * 0.5)

        ## Calculate the extra-terrestrial irradiance at a plane parallel to the earth surface(Spitters et al. (1986) Eq. 1), i.e., global incoming shortwave radiation [W m-2]
        Q_sc = 1370  # Solar constant [W m-2]
        Qo = Q_sc * np.sin(self.solar_angles['solar_altitude']) * (
            1 + 0.033 * np.cos(np.deg2rad(360 * self.energy_drivers['doy'] / 365)))

        ## The relationship between fraction diffuse and atmospheric transmission from Spitters et al. (1986) appendix.
        # "The radiation incident upon the earth surface is partly direct, with angle of incidence equal to the angle of the sun, and partly diffuse, with incidence under different angles. The diffuse flux arises from scattering (reflection and transmission) of the sun's rays in the atmosphere. The share of the diffuse flux will therefore be related to the transmission of the total radiation through the atmosphere." (Spitters et al., 1986).

        atm_trns = self.energy_drivers['sw_inc'] / Qo  # Atmospheric transmission
        R = 0.847 - (1.61 * np.sin(self.solar_angles['solar_altitude'])) + (1.04 * np.sin(
            self.solar_angles['solar_altitude']) * np.sin(self.solar_angles['solar_altitude']))
        K = (1.47-R) / 1.66

        frac_diff = np.array([self._diffuse_calc(at, rr, kk) for at, rr, kk in zip(atm_trns, R, K)])

        ## Partition into diffuse and beam radiation

        rad_diff = frac_diff * self.energy_drivers['sw_inc']  # Diffuse radiation [W m-2]
        rad_beam = self.energy_drivers['sw_inc'] - rad_diff  # Beam radiation [W m-2]

        #-------------------------------------------------
        #Partition diffuse and beam radiation into PAR following Goudriaan (1977)
        #-------------------------------------------------
        # "Up to now global radiation (300--3000 nm) has been considered, but only the 400--700nm wavebands are photosynthetically active (PAR); the fraction PAR amounts to 0.50 and is remarkably constant over different atmospheric conditions and solar elevation, provided that the angle of sun above horizon (solar altitude) is > 10 degrees (Szeicz, 1974)." (Spitters et al., 1986).

        I_od = 0.5 * rad_diff
        I_ob = 0.5 * rad_beam

        #-------------------------------------------------
        # Calculating beam radiation transmitted through the canopy
        #-------------------------------------------------

        ## Calculate the ratio of projected area to hemi-surface area for an ellipsoid
        ## C&N (1998) Eq. 15.4 sensu Campbell (1986)

        kbe = np.sqrt((self.channel_properties['x_LAD']**2) + (np.tan(self.solar_angles['sza']))**2
                      ) / (self.channel_properties['x_LAD'] +
                           (1.774 * ((self.channel_properties['x_LAD'] + 1.182)**(-0.733))))

        # Fraction of incident beam radiation penetrating the canopy
        # C&N (1998) Eq. 15.1 and leaf absorptivity as 0.8 (C&N (1998) pg. 255) as per Camp
        tau_b = np.exp(-np.sqrt(0.8) * kbe * self.energy_drivers['lai'])

        # Beam radiation transmitted through the canopy
        beam_trans = I_ob * tau_b

        #-------------------------------------------------
        # Calculating diffuse radiation transmitted through the canopy
        #-------------------------------------------------

        # Diffuse transmission coefficient for the canopy (C&N (1998) Eq. 15.5)
        tau_d = np.array([self._dt_calc(ll, x_LAD=1) for ll in self.energy_drivers['lai']])

        # Extinction coefficient for black leaves in diffuse radiation
        kd = -np.log(tau_d) / self.energy_drivers['lai']

        # Diffuse radiation transmitted through the canopy
        diff_trans = I_od * np.exp(-np.sqrt(0.8) * kd * self.energy_drivers['lai'])

        # Get the total light transmitted through the canopy

        beam_trans[is_night] = 0
        diff_trans[is_night] = 0
        transmitted = beam_trans + diff_trans
        ppfd = transmitted * 1 / 0.235  #Convert from W m-2 to umol m-2 s-1

        self.energy_response['rad_diff_PAR'] = rad_diff * 0.5
        self.energy_response['rad_beam_PAR'] = rad_beam * 0.5
        self.energy_response['rad_diff'] = rad_diff
        self.energy_response['rad_beam'] = rad_beam
        self.energy_response['diff_trans_PAR'] = diff_trans
        self.energy_response['beam_trans_PAR'] = beam_trans
        self.energy_response['total_trans_PAR'] = transmitted
        self.energy_response['diff_trans'] = diff_trans / 0.5
        self.energy_response['beam_trans'] = beam_trans / 0.5
        self.energy_response['total_trans'] = transmitted / 0.5
        self.energy_response['total_trans_PAR_ppfd'] = ppfd

    def _diffuse_calc(self, atm_trns, R, K):
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
            fdiffuse = 1.0 - (6.4 * (atm_trns-0.22) * (atm_trns-0.22))
        elif ((atm_trns > 0.35) and (atm_trns <= K)):
            fdiffuse = 1.47 - (1.66*atm_trns)
        elif (atm_trns > K):
            fdiffuse = R
        return fdiffuse

    def _integ_func(self, angle, d_sza, x_LAD, lai):
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

    def _dt_calc(self, lai, x_LAD=1):
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
        angle_seq = np.deg2rad(np.arange(0, 90))

        # Numerical integration
        d_sza = (np.pi / 2) / len(angle_seq)

        #Diffuse transmission coefficient for the canopy (C&N (1998) Eq. 15.5)

        return (2 * sum(self._integ_func(angle_seq, d_sza, x_LAD, lai)))

    def get_riparian_stream_shading(self):
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
        solar_altitude_ini : float
            Initial estimate of solar altitude.         
        sw_inc : float
            Total incoming shortwave radiation [W m-2].  
        lai: float
            Leaf are index [-].               
        x_LAD: float
            Leaf angle distribution, default = 1 [-].

        Returns
        -------
        fraction_shade_veg : float
            Percent of the wetted width shaded by vegetation [%].   
        fraction_shade_bank : float
            Percent of the wetted width shaded by the bank [%].            
        """

        #-------------------------------------------------
        # Taking the difference between the sun and stream azimuth (sun-stream)
        #-------------------------------------------------

        ## This must be handled correctly to determine if the shadow falls towards the river
        ## [sin(delta)>0] or towards the bank
        ## Eastern shading
        delta_prime = self.solar_angles['solar_azimuth_shade2'] - (
            self.channel_properties['channel_azimuth'] * np.pi / 180)
        delta_prime[delta_prime <
                    0] = np.pi + np.abs(delta_prime[delta_prime < 0]) % (2 * np.pi)  #PS 2019
        delta_east = delta_prime % (2 * np.pi)

        ## Western shading
        delta_west = copy.deepcopy(delta_east) + np.pi
        delta_west[delta_east >= np.pi] = delta_east[delta_east >= np.pi] - np.pi

        #-------------------------------------------------
        # Doing some housekeeping related to bankfull and wetted widths
        #-------------------------------------------------

        ## Calculate bankfull width
        self.channel_properties['bankfull_width'] = self.channel_properties['bottom_width'] + (
            (self.channel_properties['bank_height'] / self.channel_properties['bank_slope']) * 2)

        ## Not sure what to do here, setting water_depth > bank_height, = bank_height
        if self.channel_properties['water_depth'] > self.channel_properties['bank_height']:
            self.channel_properties['water_depth'] = self.channel_properties['bank_height']

        ## Calculate wetted width
        water_width = self.channel_properties['bottom_width'] + self.channel_properties[
            'water_depth'] * (1 / self.channel_properties['bank_slope']
                              + 1 / self.channel_properties['bank_slope'])

        ## Not sure what to do here, setting widths > bankfull, = bankfull
        if water_width > self.channel_properties['bankfull_width']:
            water_width = self.channel_properties['bankfull_width']

        self.channel_properties['water_width'] = water_width

        #-------------------------------------------------
        # Calculate the length of shading for each bank
        #-------------------------------------------------

        ## Calculating shade from the "eastern" bank

        east_bank_shade_length, east_veg_shade_length = self._shade_calc(delta_east)

        # ## Calculating shade from the "western" bank
        west_bank_shade_length, west_veg_shade_length = self._shade_calc(delta_west)

        #-------------------------------------------------
        # Calculate the total length of bank shading
        #-------------------------------------------------

        ## Calculate the total length of bank shading
        total_bank_shade_length = east_bank_shade_length + west_bank_shade_length

        #Generate a logical index where the length of bank shading is longer than wetted width
        reset_bank_max_index = total_bank_shade_length > water_width

        ## If total bank shade length is longer than wetted width, set to wetted width
        total_bank_shade_length[reset_bank_max_index] = water_width  #PS 2021

        #-------------------------------------------------
        # Calculate the total length of vegetation shading
        #-------------------------------------------------

        ## Calculate the total length of vegetation shading
        total_veg_shade_length = east_veg_shade_length + west_veg_shade_length

        ## Generate a logical index where the length of vegetation shading is longer than wetted width
        reset_veg_max_index = total_veg_shade_length > water_width

        ## If total vegetation shade length is longer than wetted width, set to wetted width
        total_veg_shade_length[total_veg_shade_length > water_width] = water_width  #PS 2021

        #-------------------------------------------------
        #Calculating the percentage of water that is shaded
        #-------------------------------------------------

        fraction_shade_bank = (total_bank_shade_length) / water_width
        fraction_shade_bank[fraction_shade_bank > 1] = 1

        fraction_shade_veg = (total_veg_shade_length-total_bank_shade_length) / water_width
        fraction_shade_veg[fraction_shade_veg > 1] = 1

        self.shading_response['fraction_shade_veg'] = fraction_shade_veg
        self.shading_response['fraction_shade_bank'] = fraction_shade_bank

    def _correct_solar_azimuth(self):
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
            Solar declination (initial estimate) [radians].
        solar_azimuth_ini : float
            Initial estimate of azimuth (initial estimate) [radians].  

        Returns
        -------

        solar_azimuth : float
            Corrected estimate of azimuth (initial estimate) [radians].
        '''

        # Initialize the adjusted azimuth
        solar_azimuth = np.nan * np.ones_like(self.solar_angles['solar_azimuth'])

        # When latitude is > solar declination additional considerations are required to
        # determine the correct azimuth
        # Generate a logical index where latitude is greater than solar declination
        lat_greater = np.deg2rad(self.channel_properties['lat']) > self.solar_angles['solar_dec']

        # Add a small amount of time (1 minute) and recalculate azimuth. This is used to
        # help determine the correct solar azimuth for locations where latitude is
        # greater than the solar declination angle.

        _, _, _, azimuth_tmp = self._calc_solar_angles(self.energy_drivers['doy'],
                                                       self.energy_drivers['hour'] + (1/60/24),
                                                       self.energy_drivers['tz_offset'],
                                                       self.channel_properties['lat'],
                                                       self.channel_properties['lon'])

        # Generate a logical index where azimuth_tmp is greater than the initial estimate
        az_tmp_greater = azimuth_tmp > self.solar_angles['solar_azimuth']

        # Generate a logical index where both Lat > solar_dec & azimuth_tmp > azimuth
        add_az = np.logical_and(lat_greater, az_tmp_greater)
        solar_azimuth[add_az] = (np.pi / 2) + self.solar_angles['solar_azimuth'][add_az]

        sub_az = np.logical_and(lat_greater, np.logical_not(az_tmp_greater))
        solar_azimuth[sub_az] = (np.pi / 2) - self.solar_angles['solar_azimuth'][sub_az]

        # When Latitude is < solar declination all angles are 90 - azimuth
        solar_azimuth[np.logical_not(lat_greater)] = (
            np.pi / 2) - self.solar_angles['solar_azimuth'][np.logical_not(lat_greater)]

        self.solar_angles['solar_azimuth_shade2'] = solar_azimuth

    def _shade_calc(self, delta):
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
        # Calculating the shading produced by the bank
        #-------------------------------------------------

        # Calculating the length of the shadow perpendicular to the bank produced by the bank
        bank_shadow_length = (1 / np.tan(self.solar_angles['solar_altitude'])) * (
            self.channel_properties['bank_height']
            - self.channel_properties['water_depth']) * np.sin(delta)

        # Finding the amount of exposed bank in the horizontal direction
        exposed_bank = (
            self.channel_properties['bank_height']
            - self.channel_properties['water_depth']) / self.channel_properties['bank_slope']

        # From PS: if(BH - WL <= 0 | BS == 0) exposed_bank <- 0 #P.S. , commented this out because
        # I think I assumed that this couldn't be negative even if its confusing to be so

        # Finding how much shade falls on the surface of the water
        stream_shade_bank = bank_shadow_length - exposed_bank
        stream_shade_bank[stream_shade_bank < 0] = 0

        #-------------------------------------------------
        #Calculating the shading produced by the Vegetation
        #-------------------------------------------------

        #From top of the tree
        stream_shade_top = (1 / np.tan(self.solar_angles['solar_altitude'])) * (
            self.channel_properties['tree_height'] + self.channel_properties['bank_height']
            - self.channel_properties['water_depth']) * np.sin(delta) - exposed_bank
        stream_shade_top[stream_shade_top < 0] = 0

        #From the overhang
        stream_shade_overhang = (1 / np.tan(self.solar_angles['solar_altitude'])) * (
            self.channel_properties['overhang_height'] + self.channel_properties['bank_height']
            - self.channel_properties['water_depth']
        ) * np.sin(delta) + self.channel_properties['overhang'] - exposed_bank
        stream_shade_overhang[stream_shade_overhang < 0] = 0

        #Selecting the maximum and minimum
        stream_shade_top.reshape((1, len(stream_shade_top)))
        stream_shade_overhang.reshape((1, len(stream_shade_overhang)))

        # Get max(shade from top, shade from overhang)
        # Note from PS: "here I take a departure from the r_shade matlab code. For some reason the code
        # Takes the maximum - min shadow length, but in the paper text it clearly states max
        # See pg 14 Li et al. (2012)"

        veg_shade_bound = np.column_stack((stream_shade_top, stream_shade_overhang))
        stream_shade_veg_max = np.amax(veg_shade_bound, axis=1)

        # If the maximum shadow length is longer than the wetted width, set to width
        stream_shade_veg_max[
            stream_shade_veg_max >
            self.channel_properties['water_width']] = self.channel_properties['water_width']

        return stream_shade_bank, stream_shade_veg_max

    def get_energy_stream(self):
        #-------------------------------------------------
        # Calculating the weighted mean of light reaching the stream surface
        #-------------------------------------------------
        #Calculating weighted mean of irradiance at the stream surface

        # From Savoy's code
        #par_surface_original = (self.energy_response['total_trans_PAR_ppfd'] * self.shading_response['fraction_shade_veg']) + (self.energy_drivers['sw_inc']* 2.114 * (1 - (self.shading_response['fraction_shade_veg'] + self.shading_response['fraction_shade_bank'])))

        # Corrected estimates

        energy_diff_surface_PAR = (self.energy_response['diff_trans_PAR']
                                   * self.shading_response['fraction_shade_veg']) + (
                                       self.energy_response['rad_diff_PAR'] *
                                       (1 - (self.shading_response['fraction_shade_veg']
                                             + self.shading_response['fraction_shade_bank'])))

        energy_beam_surface_PAR = (self.energy_response['beam_trans_PAR']
                                   * self.shading_response['fraction_shade_veg']) + (
                                       self.energy_response['rad_beam_PAR'] *
                                       (1 - (self.shading_response['fraction_shade_veg']
                                             + self.shading_response['fraction_shade_bank'])))

        energy_total_surface_PAR = (self.energy_response['total_trans_PAR']
                                    * self.shading_response['fraction_shade_veg']) + (
                                        self.energy_drivers['sw_inc'] * 0.5 *
                                        (1 - (self.shading_response['fraction_shade_veg']
                                              + self.shading_response['fraction_shade_bank'])))

        energy_diff_surface = (self.energy_response['diff_trans']
                               * self.shading_response['fraction_shade_veg']) + (
                                   self.energy_response['rad_diff'] *
                                   (1 - (self.shading_response['fraction_shade_veg']
                                         + self.shading_response['fraction_shade_bank'])))

        energy_beam_surface = (self.energy_response['beam_trans']
                               * self.shading_response['fraction_shade_veg']) + (
                                   self.energy_response['rad_beam'] *
                                   (1 - (self.shading_response['fraction_shade_veg']
                                         + self.shading_response['fraction_shade_bank'])))

        energy_total_surface = (self.energy_response['total_trans']
                                * self.shading_response['fraction_shade_veg']) + (
                                    self.energy_drivers['sw_inc'] *
                                    (1 - (self.shading_response['fraction_shade_veg']
                                          + self.shading_response['fraction_shade_bank'])))

        #energy_total_surface_PAR_ppfd = energy_total_surface_PAR * (1/0.235)
        energy_total_surface_PAR_ppfd = (self.energy_response['total_trans_PAR_ppfd']
                                         * self.shading_response['fraction_shade_veg']) + (
                                             self.energy_drivers['sw_inc'] * 2.114 *
                                             (1 - (self.shading_response['fraction_shade_veg']
                                                   + self.shading_response['fraction_shade_bank'])))

        ## Generate a logical index of night and day. Night = SZA > 90
        is_night = self.solar_angles['sza'] > (np.pi * 0.5)

        #par_surface_original[is_night] = 0
        energy_diff_surface_PAR[is_night] = 0
        energy_beam_surface_PAR[is_night] = 0
        energy_total_surface_PAR[is_night] = 0
        energy_diff_surface[is_night] = 0
        energy_beam_surface[is_night] = 0
        energy_total_surface[is_night] = 0
        energy_total_surface_PAR_ppfd[is_night] = 0

        #self.energy_response['par_surface_original'] = par_surface_original
        self.energy_response['energy_diff_surface_PAR'] = energy_diff_surface_PAR
        self.energy_response['energy_beam_surface_PAR'] = energy_beam_surface_PAR
        self.energy_response['energy_total_surface_PAR'] = energy_total_surface_PAR
        self.energy_response['energy_diff_surface'] = energy_diff_surface
        self.energy_response['energy_beam_surface'] = energy_beam_surface
        self.energy_response['energy_total_surface'] = energy_total_surface
        self.energy_response['energy_total_surface_PAR_ppfd'] = energy_total_surface_PAR_ppfd

    def run_streamlight(self):
        self.get_solar_angles()
        self.get_radiative_transfer_estimates_cn_1998()
        self.get_riparian_stream_shading()
        self.get_energy_stream()

    def __str__(self):
        return f"StreamLight Object"
