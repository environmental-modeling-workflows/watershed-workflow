#!/usr/bin/env python
import os
import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer
from pyproj import CRS

import xarray as xr
import rasterio
from rasterio.transform import Affine

# default PFT classes by ELM
natural_pfts={'pftname':[
                    "not_vegetated                           ",
                    "needleleaf_evergreen_temperate_tree     ",
                    "needleleaf_evergreen_boreal_tree        ",
                    "needleleaf_deciduous_boreal_tree        ",
                    "broadleaf_evergreen_tropical_tree       ",
                    "broadleaf_evergreen_temperate_tree      ",
                    "broadleaf_deciduous_tropical_tree       ",
                    "broadleaf_deciduous_temperate_tree      ",
                    "broadleaf_deciduous_boreal_tree         ",
                    "broadleaf_evergreen_shrub               ",
                    "broadleaf_deciduous_temperate_shrub     ",
                    "broadleaf_deciduous_boreal_shrub        ",
                    "c3_arctic_grass                         ",
                    "c3_non-arctic_grass                     ",
                    "c4_grass                                ",
                    "c3_crop                                 ",  # called generic crop, will drop off if create_crop_landunit = .true.
                    "c3_irrigated                            "   # called generic crop, will drop off if create_crop_landunit = .true.
                    ],
             'pftnum': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            };
    
# the cfts when 'create_crop_landunit' is on in ELM option,
# and surfdata includes two dims of natpft, 0-14, and cft, 15-16/24/50.
cfts       ={'pftname':[
                    "c3_crop                                 ",  # called generic crop
                    "c3_irrigated                            "   # called generic crop
                    "corn                                    ",
                    "irrigated_corn                          ",
                    "spring_temperate_cereal                 ",
                    "irrigated_spring_temperate_cereal       ",
                    "winter_temperate_cereal                 ",
                    "irrigated_winter_temperate_cereal       ",
                    "soybean                                 ",
                    "irrigated_soybean                       "
                    ],
             'pftnum': [15,16,17,18,19,20,21,22,23,24]
            };


fates_pfts ={'pftname':[
                    "broadleaf_evergreen_tropical_tree",
                    "needleleaf_evergreen_extratrop_tree",
                    "needleleaf_colddecid_extratrop_tree",
                    "broadleaf_evergreen_extratrop_tree",
                    "broadleaf_hydrodecid_tropical_tree",
                    "broadleaf_colddecid_extratrop_tree",
                    "broadleaf_evergreen_extratrop_shrub",
                    "broadleaf_hydrodecid_extratrop_shrub",
                    "broadleaf_colddecid_extratrop_shrub",
                    "broadleaf_evergreen_arctic_shrub",
                    "broadleaf_colddecid_arctic_shrub",
                    "arctic_c3_grass",
                    "cool_c3_grass",
                    "c4_grass"
                    ],
             'pftnum': [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            };

# NLCD LC indices and names
nlcd_lc = {
    11: ('Open Water'),
    12: ('Perrenial Ice/Snow'),
    21: ('Developed, Open Space'),
    22: ('Developed, Low Intensity'),
    23: ('Developed, Medium Intensity'),
    24: ('Developed, High Intensity',),
    31: ('Barren Land'),
    41: ('Deciduous Forest'),
    42: ('Evergreen Forest'),
    43: ('Mixed Forest'),
    51: ('Dwarf Scrub'),
    52: ('Shrub/Scrub'),
    71: ('Grassland/Herbaceous'),
    72: ('Sedge/Herbaceous'),
    73: ('Lichens'),
    74: ('Moss'),
    81: ('Pasture/Hay'),
    82: ('Cultivated Crops'),
    90: ('Woody Wetlands'),
    95: ('Emergent Herbaceous Wetlands'),
}
    
#--- # land cover type of nlcd <==> ELM default PFTs and landunits
def mksrfdata_lupft_fromNLCD(fsurfnc_in, nlcd_xr, natvegLUonly=False, 
                             defaultPFT=True, grid_aggregated=False):
    
    '''
        fsurfnc_in - a standard ELM surfdata file, matching with nlcd data spatial extent and resolution
        nlcd_xf    - xarray of nlcd data
        natvegLUonly    - (optional) assign all nlcd data as so-called naturally-vegetated land unit
        defaultPFT      - (optional) standard ELM PFTs of 17 types. User-provided PFTs not yet (TODO)
        grid_aggregated - (optional) in case, resolution and spatial extent are different. (TODO)
        
    '''
    
    # Rooting profiles of nlcd -> pft from WW's default 
    # 41: Deciduous Forest → BDT Temperate
    # 42: Evergreen Forest → NET Temperate
    # 43: Mixed Forest → NET Temperate (per your instruction)
    # 51: Dwarf Scrub → BDS Boreal
    # 52: Shrub/Scrub → BDS Temperate
    # 82: Cultivated Crops → Crop (all crop PFTs share same values)
    # 90: Woody Wetlands → BDT Temperate

    # stomatal max. resis -> pft from WW's default 
    # 21: Developed Open Space → C3 grass
    # 22: Developed Low Intensity → C3 grass
    # 41: Deciduous Forest → BDT Temperate
    # 42: Evergreen Forest → NET Temperate
    # 43: Mixed Forest → NET Temperate
    # 51: Dwarf Scrub → BDS Boreal
    # 52: Shrub/Scrub → BDS Temperate
    # 71: Grassland/Herbaceous → C3/C4 grass
    # 81: Pasture/Hay → C3 grass
    # 82: Crops → Crop PFT
    # 90: Woody Wetlands → BDT Temperate
    # 95: Herbaceous Wetlands → C3 grass
    # rest: Barren/impervious NLCD types
    
    '''
    ELM's LandUnit: 
      ! --------------------------------------------------------
      !   1  => (istsoil)    soil (vegetated or bare soil landunit) --> PCT_NATVEG,  PCT_NAT_PFT(17 or 15)
      !   2  => (istcrop)    crop (only for crop configuration)     --> PCT_CROP,    PCT_CFT(0 or 2+)
      !   3  => (istice)     land ice                               --> PCT_GLACIER
      !   4  => (istice_mec) land ice (multiple elevation classes)  
      !   5  => (istdlak)    deep lake                              --> PCT_LAKE
      !   6  => (istwet)     wetland                                --> PCT_WETLAND                     
      !   7  => (isturb_tbd) urban tbd                              --> PCT_URBAN(3)  
      !   8  => (isturb_hd)  urban hd
      !   9  => (isturb_md)  urban md
      !
      
    '''
    
    # assuming input 'fsrfnc_in' includes climatezone info implicitly in PCT_NAT_PFT
    with Dataset(fsurfnc_in,'r') as src:
        pct_nat_pft0 = src.variables['PCT_NAT_PFT'][...]
        if defaultPFT:
            # 17 default PFTs, classified into 5 likely distributed climate zones:
            #  0 = any zone, 1 = tropical, 2 = temperate, 3 = boreal, 4 = arctic
            # NOTE: this is NOT geographical concept rather geo-climate-veg-terrain complex
            trop =                  pct_nat_pft0[4,]+pct_nat_pft0[6,]
            tmpr = pct_nat_pft0[1,]+pct_nat_pft0[5,]+pct_nat_pft0[7,]+pct_nat_pft0[10,]
            borl = pct_nat_pft0[2,]+pct_nat_pft0[3,]+pct_nat_pft0[8,]+pct_nat_pft0[11,]
            tndr = pct_nat_pft0[12,]
            # likely the 5 geo-climate-veg zones have overlaps.
            cz_arr=np.zeros(np.append(5,pct_nat_pft0.shape[1:]))
            cz_arr[0,] = cz_arr[0,]+0.01 # 0.01 would allow all zone indices, which all 4 above summed to near zero
            cz_arr[1,] = trop
            cz_arr[2,] = tmpr
            cz_arr[3,] = borl
            cz_arr[4,] = tndr
            # Currently assumping each grid cell only has one single NLCD lclu-type
            # so, the max. fraction of domainate zone would be used
            climatezone = np.argmax(cz_arr,axis=0)
    
    #
    srf_lu_pft = {}
    if not grid_aggregated:
        # single land unit and PFT for exactly surface grids as in nlcd_xr
        srf_lu_pft['PCT_NATVEG']  = np.zeros(pct_nat_pft0.shape[1:], dtype=float)
        srf_lu_pft['PCT_CROP']    = np.zeros(pct_nat_pft0.shape[1:], dtype=float)
        srf_lu_pft['PCT_GLACIER'] = np.zeros(pct_nat_pft0.shape[1:], dtype=float)
        srf_lu_pft['PCT_LAKE']    = np.zeros(pct_nat_pft0.shape[1:], dtype=float)
        srf_lu_pft['PCT_WETLAND'] = np.zeros(pct_nat_pft0.shape[1:], dtype=float)
        srf_lu_pft['PCT_URBAN']   = np.zeros(np.append(3, pct_nat_pft0.shape[1:]), dtype=float)
        srf_lu_pft['PCT_NAT_PFT'] = np.zeros(pct_nat_pft0.shape, dtype=float)

        for lcidx, lcname in nlcd_lc.items():
            lc_xy = (nlcd_xr==lcidx)
            if not any(lc_xy): continue  #skip
            
            if lcidx == 11: # Open Water  - LAKE
                if natvegLUonly: 
                    srf_lu_pft['PCT_NATVEG'][lc_xy] = 100.0
                else:
                    srf_lu_pft['PCT_LAKE'][lc_xy]   = 100.0
                srf_lu_pft['PCT_NAT_PFT'][0][lc_xy] = 100.0   # not_vegetaged, not used in ELM but just in case
            elif lcidx == 12: # Perrenial ICE/SNOW - GLACIER (land ice)?
                if natvegLUonly: 
                    srf_lu_pft['PCT_NATVEG'][lc_xy] = 100.0
                else:
                    srf_lu_pft['PCT_GLACIER'][lc_xy]= 100.0
                srf_lu_pft['PCT_NAT_PFT'][0][lc_xy] = 100.0   # not_vegetaged
            
            elif lcidx == 82: # crop
                if natvegLUonly: 
                    srf_lu_pft['PCT_NATVEG'][lc_xy] = 100.0
                else:
                    srf_lu_pft['PCT_CROP'][lc_xy]   = 100.0
                srf_lu_pft['PCT_NAT_PFT'][15][lc_xy]= 100.0  # in c3 crop, non-irrigated 
            
            #wetland ? (TODO - currently assigned to NATVEG)
            
            else: # naturally-vegetaged or barren land
                srf_lu_pft['PCT_NATVEG'][lc_xy] = 100.0
                
                #barrens
                if lcidx == 31: 
                    srf_lu_pft['PCT_NAT_PFT'][0][lc_xy] = 100.0
                
                #graminoids  
                elif lcidx == 21 or lcidx == 22 \
                    or lcidx == 71 or lcidx == 72 or lcidx == 73 or lcidx == 74 \
                    or lcidx ==81 or lcidx == 95: 
                    # all above - C3 grass
                    tndr_xy = (tndr>1.0)
                    srf_lu_pft['PCT_NAT_PFT'][12][(lc_xy & tndr_xy)] = 100.0
                    srf_lu_pft['PCT_NAT_PFT'][13][(lc_xy & ~tndr_xy)] = 100.0
                
                # trees
                elif lcidx == 41 or lcidx == 90:
                    # to broadleaf deciduous trees (BDT)
                    trop_xy = (climatezone==1)
                    tmpr_xy = (climatezone==2)
                    borl_xy = (climatezone==3)
                    srf_lu_pft['PCT_NAT_PFT'][6][(lc_xy & trop_xy)] = 100.0
                    srf_lu_pft['PCT_NAT_PFT'][7][(lc_xy & tmpr_xy)] = 100.0
                    srf_lu_pft['PCT_NAT_PFT'][8][(lc_xy & borl_xy)] = 100.0
                    # maybe none in climatezone 
                    if not any(trop_xy) and not any(tmpr_xy) and not any(borl_xy):
                        srf_lu_pft['PCT_NAT_PFT'][7][lc_xy] = 100.0 # assign to BDT temperate                    
                elif lcidx == 42:
                    # to needleleaf evergreen trees (NET)
                    tmpr_xy = (climatezone==2)
                    borl_xy = (climatezone==3)
                    srf_lu_pft['PCT_NAT_PFT'][1][(lc_xy & tmpr_xy)] = 100.0
                    srf_lu_pft['PCT_NAT_PFT'][2][(lc_xy & borl_xy)] = 100.0
                    # maybe none in climatezone 
                    if not any(tmpr_xy) and not any(borl_xy):
                        srf_lu_pft['PCT_NAT_PFT'][1][lc_xy] = 100.0 # assign all to NET temperate
                elif lcidx == 43:
                    # to either needleleaf deciduous or broadleaf evergreen trees
                    trop_xy = (climatezone==1)
                    tmpr_xy = (climatezone==2)
                    borl_xy = (climatezone==3)
                    srf_lu_pft['PCT_NAT_PFT'][3][(lc_xy & borl_xy)] = 100.0 # needleleaf deciduous boreal 
                    srf_lu_pft['PCT_NAT_PFT'][4][(lc_xy & trop_xy)] = 100.0 # broadleaf evergreen tropical
                    srf_lu_pft['PCT_NAT_PFT'][5][(lc_xy & tmpr_xy)] = 100.0 # broadleaf evergreen temperate
                    # maybe none in climatezone 
                    if not any(tmpr_xy) and not any(borl_xy):
                        srf_lu_pft['PCT_NAT_PFT'][5][lc_xy] = 100.0         # assign all to broadleaf evergreen temperate
                    
                #shrubs
                elif lcidx == 51 or lcidx == 52:
                    # to broadleaf deciduous boreal shrub 
                    tmpr_xy = (climatezone==2)
                    borl_xy = (climatezone==3)
                    srf_lu_pft['PCT_NAT_PFT'][10][(lc_xy & tmpr_xy)] = 100.0
                    srf_lu_pft['PCT_NAT_PFT'][11][(lc_xy & borl_xy)] = 100.0
                    # maybe none in climatezone 
                    if not any(tmpr_xy) and not any(borl_xy):
                        srf_lu_pft['PCT_NAT_PFT'][9][lc_xy] = 100.0 # assign all to broadleaf evergreen shrub
                 
                # whatever missed (just in case)
                else:
                    srf_lu_pft['PCT_NAT_PFT'][0][lc_xy] = 100.0
        #
    else:
        print('TODO!')
    #
       
    # 100% checking
    lu_sum100 = srf_lu_pft['PCT_NATVEG']+srf_lu_pft['PCT_CROP']+ \
            srf_lu_pft['PCT_LAKE']+srf_lu_pft['PCT_WETLAND']+ \
            srf_lu_pft['PCT_GLACIER']+np.sum(srf_lu_pft['PCT_URBAN'],axis=0)

    pft_sum100 = np.sum(srf_lu_pft['PCT_NAT_PFT'],axis=0)
    if any(abs(pft_sum100-100.0)>1.0e-8):
        print('Error: grid summed Land Units not 100%:', np.where(abs(pft_sum100-100.0)>1.0e-8))
        exit(-1)
    
    return srf_lu_pft
    

#--- # downloading a boxed SoilGrid data (v2.0.1)
# 
# Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: 
# SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty, 
# SOIL, 7, 217–240, https://doi.org/10.5194/soil-7-217-2021, 2021.

# Following a notebook script found at: https://git.wur.nl/isric/soilgrids/soilgrids.notebooks.git
#
def download_geotiff_soilgrids(Range_XLONG=[], Range_YLATI=[], outputpath='./original', \
                                soilvars=['ocd','bdod','sand','silt','clay'], \
                                value='mean'):
    
    from owslib.wcs import WebCoverageService
    
    # SoilGrids map's CRS
    crs_wcs = "http://www.opengis.net/def/crs/EPSG/0/152160"
    # from above def, its equvalent is:
    crs_proj4 = CRS.from_proj4('+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs')
    # standard horizons
    horizons = ['0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm']   
    # mean
    #value='mean'
     
    # coverage of rectangle-boxed area
    #Range_XLONG = [-150.0, -149.0]
    #Range_YLATI =[68.5, 69.0]
    
    # EPSG: 4326
    # Proj4: +proj=longlat +datum=WGS84 +no_defs
    lonlatProj = CRS.from_epsg(4326) # in lon/lat coordinates
    Tlonlat2xy = Transformer.from_proj(lonlatProj, crs_proj4, always_xy=True)

    X1,Y1 = Tlonlat2xy.transform(Range_XLONG[0],Range_YLATI[0]) #left-bottom
    X2,Y2 = Tlonlat2xy.transform(Range_XLONG[0],Range_YLATI[1]) #right-bottom
    X3,Y3 = Tlonlat2xy.transform(Range_XLONG[1],Range_YLATI[1]) #right-top
    X4,Y4 = Tlonlat2xy.transform(Range_XLONG[1],Range_YLATI[0]) #left-top

    # when projection-transformed, not rectangle anymore, so need to re-do min/max (otherwise, coverage may be incompleted)
    subsets = [('X', min(X1,X2,X3,X4), max(X1,X2,X3,X4)), ('Y', min(Y1,Y2,Y3,Y4), max(Y1,Y2,Y3,Y4))]
    
    # organic carbon density: ocd, hg/m3 (aka 0.1kg/m3)
    
    # obtain a full geotiff profile for tiff writing
    template_profile = rasterio.open(outputpath+'/soilgrids_template_withcrs.tif').profile.copy()
    
    for ivar in soilvars:
        soilgrids_wcs = WebCoverageService('http://maps.isric.org/mapserv?map=/map/'+ivar+'.map',
                         version='2.0.1')
        #infos for checking
        

        for iz in horizons:
            svar_horizon_id = ivar+'_'+iz+'_'+value
            svar_horizon = soilgrids_wcs.contents[svar_horizon_id]
            
            response = soilgrids_wcs.getCoverage(identifier=[svar_horizon_id], 
                                   crs=crs_wcs,
                                   subsets=subsets, 
                                   resx=250, resy=250, 
                                   format=svar_horizon.supportedFormats[0])  
            with open('./tmp.tif', 'wb') as file:
                # better to save to xarray, but didn't figure out how to from response.read() -> xarray
                file.write(response.read())
            
            # ideally, the above file in tiff should have 'CRS' info but not
            # so we need to redo
            rdata=rasterio.open('./tmp.tif')
            newprofile = rdata.profile.copy()
            newprofile['crs'] = template_profile['crs']            
            svar_horizon_value_tif = outputpath+'/'+svar_horizon_id+'.tif'
            with rasterio.open(
                svar_horizon_value_tif,
                'w',
                **newprofile,
            ) as file:
                file.write(rdata.read(1), 1)
            
            if os.path.exists('./tmp.tif'): os.remove('./tmp.tif')
            
        # layer loop
    #variable loop
    

#--- #
# interplating layered soil data to match with ELM soil vertical structure
def mksrfdata_soilcolumn_interp(srf_soildata=np.empty((0)), srf_soilnode=np.empty((0)), \
                                nlevsoi=10, fill_method='zero'):
    
    from watershed_workflow.elm_domain import soilcolumn
    
    if srf_soildata.size!= srf_soilnode.size \
        or srf_soildata.size<=0: return np.empty((0))
    
    # default ELM soil column
    zisoi, dzsoi, zsoi = soilcolumn()
    # by default, ELM soil layer no  is 10, the rest 5 layers are as rock or alike 
    zisoi = zisoi[0:nlevsoi+1]  
    dzsoi = dzsoi[1:nlevsoi+1]
    zsoi = zsoi[1:nlevsoi+1]  # so-called node-depth, within a soil layer (but not middle)    
    znodes = xr.Dataset({'z': ("points", zsoi)})
    
    if srf_soilnode.size<=1:
        # but not sure if 2 data points works with 'data.interp()' function
        interp_vert = np.zeros(len(zsoi))
        if fill_method == 'extrapolate':
            interp_vert[:] = srf_soildata
        else:
            for iz in range(len(zsoi)):
                if zsoi[iz]<= srf_soilnode:
                    interp_vert[iz] = srf_soildata
                else:
                    continue
                           
        return interp_vert
    
    else:
        data = xr.DataArray(srf_soildata, 
                        dims=("z"), 
                        coords={"z": srf_soilnode})
        
        if fill_method == 'zero':
            interp_vert = data.interp(znodes, method='linear', \
                                  kwargs={'fill_value': 0})
        else:
            interp_vert = data.interp(znodes, method='linear', \
                                  kwargs={'fill_value': fill_method})        
        return interp_vert.as_numpy()
            
#--- # 
# 
def mksrfdata_updatevals(fsurfnc_in, fsurfnc_out=None, \
                         user_srf_data={}, user_srfnc_file=None, user_srf_vars=None):
    
    print('#--------------------------------------------------#')
    print("Replacing values in surface data by merging user-provided dataset")
    if fsurfnc_out==None: fsurfnc_out ='./'+fsurfnc_in.split('/')[-1]+'-merged'
    
       
    natpft = np.asarray(range(max(natural_pfts['pftnum'])+1))
 
    #---------------------------------------------------------------------------------------

    UNSTRUCTURED = False    
    if not user_srfnc_file==None:
        print('read data from: ', user_srfnc_file)
        f=Dataset(user_srfnc_file)
        if 'gridcell' in f.dimensions.items(): UNSTRUCTURED = True
        
        # in case natpft is different in user_srfnc_file than in fsurfnc_in
        if 'natpft' in f.dimensions.items(): 
            natpft = np.asarray(range(f.dimensions['natpft'].size))
        
        user_srf = {}
        user_srf['LATIXY'] = f.variables['LATIXY']
        user_srf['LONGXY'] = f.variables['LONGXY']
        user_srf['AREA'] = f.variables['AREA']
        
        if user_srf_vars==None:
            user_vname = user_srf_vars.keys()
        else:
            user_vname = user_srf_vars.split(',')
        for v in user_vname:
            if v in f.variables.keys(): user_srf[v] =f.variables[v][...] 
    elif not len(user_srf_data)<=0:
        if len(np.squeeze(user_srf_data['LATIXY']).shape)==1:
            UNSTRUCTURED = True            
        if user_srf_vars==None:
            user_vname = user_srf_data.keys()
        else:
            user_vname = user_srf_vars.split(',')
        user_srf = user_srf_data

        # in case natpft is different in user_srf_data (list)
        if 'PCT_NAT_PFT' in user_srf_data.keys():
            # PCT_NAT_PFT in srf_data either in (natpft, gridcell) or (natpft, lsmlat, lsmlon)
            shp = user_srf_data['PCT_NAT_PFT'].shape
            natpft = np.asarray(range(shp[0]))
        
    #---------------------------------------------------------------------------------------
    #                    
    # write into nc file
    with Dataset(fsurfnc_in,'r') as src, Dataset(fsurfnc_out, "w") as dst:
            
        # new surfdata dimensions
        for dname, dimension in src.dimensions.items():
            if dname == 'natpft':
                len_dimension = len(natpft)            # dim length from new data or original
            elif dname == 'gridcell':
                len_dimension = user_srf['LATIXY'].flatten().size
            elif dname in ['lsmlat','lat']:
                len_dimension = user_srf['LATIXY'].shape[0]
            elif dname in ['lsmlon', 'lon']:
                len_dimension = user_srf['LONGXY'].shape[1]                
            else:
                len_dimension = len(dimension)
            dst.createDimension(dname, len_dimension if not dimension.isunlimited() else None)
            #
            
        # create variables and write to dst
        for vname, variable in src.variables.items():

            if UNSTRUCTURED:
                vdim = variable.dimensions
                if 'gridcell' not in vdim and \
                    ('lsmlat' in vdim and 'lsmlon' in vdim):
                    vdim = vdim.replace('lsmlon', 'gridcell')
                    vdim = vdim.remove('lsmlat')
            else:
                vdim = variable.dimensions
    
            # create variables, but will update its values later 
            # NOTE: here the variable value size updated due to newly-created dimensions above
            dst.createVariable(vname, variable.datatype, vdim)
            # copy variable attributes all at once via dictionary after created
            dst[vname].setncatts(src[vname].__dict__)
                  
            # values
            src_vals = src[vname][...]

            # dimension length may change, so need to 
            if vname in user_vname:
                varvals = user_srf[vname][...]
                #
            else:
                varvals = src[vname][...]
                #
            
            #                                
            dst[vname][...] = varvals
                    
        # end of variable-loop        
                
        
        print('user surfdata merged and nc file created and written successfully!')
        
    #
#

#--------------------------------------------------------------------
def test(surf_from_atsm2={}, surf_vars=''):
    input_path  = './'
    output_path = './'
    
    '''
    mksrfdata_updatevals(os.path.joint((input_path, \
                         'surfdata_2687x1pt_simyr1850_c240308_TOP-coweeta.nc')), \
                         user_srf_data=surf_from_atsm2, \
                         user_srf_vars=surf_vars)
    '''
    
     
#if __name__ == '__main__':
#    test()




