#!/usr/bin/env python
import os
import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer
from pyproj import CRS

import xarray as xr
import rasterio
from rasterio.transform import Affine

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
                         user_srf_data={}, user_srfnc_file=None, user_srf_vars=None, OriginPFTclass=True):
    
    print('#--------------------------------------------------#')
    print("Replacing values in surface data by merging user-provided dataset")
    if fsurfnc_out==None: fsurfnc_out ='./'+fsurfnc_in.split('/')[-1]+'-merged'
    
    
    #---------------------------------------------------------------------------------------
    #
    # Arctic PFT classes in B. Sulman et al (2021) paper: 12 arctic PFTs + 2 additional tree PFTs   
    user_pfts={'pftname':[
                    "non_vegetated",
                    "arctic_lichen",
                    "arctic_bryophyte",
                    "arctic_needleleaf_tree",
                    "arctic_broadleaf_tree",
                    "arctic_evergreen_shrub",
                    "arctic_evergreen_tall_shrub",
                    "arctic_deciduous_dwarf_shrub",
                    "arctic_deciduous_low_shrub",
                    "arctic_low_to_tall_willowbirch_shrub",
                    "arctic_low_to_tall_alder_shrub",
                    "arctic_forb",
                    "arctic_dry_graminoid",
                    "arctic_wet_graminoid"
                    ],
                'pftnum': [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
               };
    
    
    if OriginPFTclass:
        # lichen as not_vegetated (0), moss/forb/graminoids as c3 arctic grass (12),
        # evergreen shrub(9), deci. boreal_shrub(11),
        # evergreen boreal tree(2), deci boreal tree (3)
        user_pfts['pftnum'] = [0,0,12,2,3,9,9,11,11,11,11,12,12,12]
        natpft = np.asarray(range(17))
    else:
        natpft = np.asarray(range(max(user_pfts['pftnum'])+1)) # this is the real arcticpft order number 
 
    #---------------------------------------------------------------------------------------

    UNSTRUCTURED = False    
    if not user_srfnc_file==None:
        print('read data from: ', user_srfnc_file)
        f=Dataset(user_srfnc_file)
        if 'gridcell' in f.dimensions.items(): UNSTRUCTURED = True
        
        user_srf = {}
        user_srf['LATIXY'] = f.variables['LATIXY']
        
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
            user_vname = user_srf_vars.keys()
        else:
            user_vname = user_srf_vars.split(',')
        user_srf = user_srf_data
        
    #---------------------------------------------------------------------------------------
    #                    
    # write into nc file
    with Dataset(fsurfnc_in,'r') as src, Dataset(fsurfnc_out, "w") as dst:
            
        # new surfdata dimensions
        for dname, dimension in src.dimensions.items():
            if dname == 'natpft':
                len_dimension = len(natpft)            # dim length from new data
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
    
    mksrfdata_updatevals(os.path.joint((input_path, \
                         'surfdata_2687x1pt_simyr1850_c240308_TOP-coweeta.nc')), \
                         user_srf_data=surf_from_atsm2, \
                         user_srf_vars=surf_vars)
    
      
#if __name__ == '__main__':
#    test()




