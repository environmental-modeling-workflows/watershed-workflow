
# utility to generate 1D and 2D domain
# use previous dataPartition module. Code need to be clean up

import os
import math
import netCDF4 as nc
import numpy as np

DIN_LOC_ROOT = None
def set_e3sm_input(path):
    """Sets the root path to search for E3SM input data"""
    global DIN_LOC_ROOT
    DIN_LOC_ROOT = path

def elm_arcradians2_from_km2(area_km2, R_meters=6.37122e6):
    one_over_re2 = 1.0e6/R_meters/R_meters
    return area_km2 * one_over_re2
#

#--- #------ ELM soil column layer thickness and depths (nodes and interfaces) ------#
def soilcolumn(more_vertlayers=False, nlevgrnd=15, printout=False):

    # a few ELM constants relevant to soil column
    scalez      = 0.025   # Soil layer thickness discretization (m)
    thick_equal = 0.2
    nlev_equalspace   = 15
    toplev_equalspace =  6



    '''
    The following are python scripts for algorithm in elm/src/main/initVerticalMod.F90: L112-156
    '''
    # Soil layers and interfaces (assumed same for all non-lake patches)
    # "0" refers to soil surface and "nlevbed" refers to the bottom of model soil

    # node depths.

    # NOTE: These elm soil-layer nodes are within a layer, but NOT the centroids if varying layerthickness.
    # the middle-point of two adjacent nodes are called 'interfaces' (layer's up/bottom)
    # In ELM, fluxes are occured btw adjacent nodes. but 'mass' are within a layer volume.

    # For convenience, index 0 in all np array is for 'surface' so that layer real indices are matching with ELM's

    if ( more_vertlayers ):
        # replace standard exponential grid with a grid that starts out exponential, 
        # then has several evenly spaced layers, then finishes off exponential. 
        # this allows the upper soil to behave as standard, but then continues 
        # with higher resolution to a deeper depth, so that, for example, permafrost
        # dynamics are not lost due to an inability to resolve temperature, moisture, 
        # and biogeochemical dynamics at the base of the active layer

        nlevgrnd    =  15 + nlev_equalspace

        zsoi = np.empty(nlevgrnd+1); zsoi[:] = 0.0
        for j in range(1, toplev_equalspace+1):
            zsoi[j] = scalez*(math.exp(0.5*(j-0.5))-1.0)

        for j in range(toplev_equalspace+1,toplev_equalspace + nlev_equalspace+1):
            zsoi[j] = zsoi[j-1] + thick_equal

        for j in range(toplev_equalspace + nlev_equalspace +1, nlevgrnd+1):
            zsoi[j] = scalez*(math.exp(0.5*((j - nlev_equalspace)-0.5))-1.0) + nlev_equalspace * thick_equal

    else:

        zsoi = np.empty(nlevgrnd+1); zsoi[:] = 0.0
        for j in range(1, nlevgrnd+1):
            zsoi[j] = scalez*(math.exp(0.5*(j-0.5))-1.0)

    # thickness b/n two interfaces (i.e. ATS z-coordinated-nodes)
    dzsoi = np.empty(nlevgrnd+1); dzsoi[:] = 0.0
    dzsoi[1] = 0.5*(zsoi[1]+zsoi[2])
    for j in range(2,nlevgrnd):
       dzsoi[j]= 0.5*(zsoi[j+1]-zsoi[j-1])
    dzsoi[nlevgrnd] = zsoi[nlevgrnd]-zsoi[nlevgrnd-1]

    # interface, i.e. ATS z-coordinated-node depths
    zisoi = np.empty(nlevgrnd+1); zisoi[:] = 0.0
    zisoi[0] = 0.0
    for j in range(1, nlevgrnd):
       zisoi[j] = 0.5*(zsoi[j]+zsoi[j+1])
    zisoi[nlevgrnd] = zsoi[nlevgrnd] + 0.5*dzsoi[nlevgrnd]

    if printout:
        print('zsoi with 0 indexing as surface: ', zsoi)
        print('zisoi with 0 indexing as surface: ', zisoi)
        print('dzsoi with 0 indexing as surface: ', dzsoi)

    return zisoi, dzsoi, zsoi

#

'''
 standardilized ELM domain.nc write
'''
def domain_ncwrite(elmdomain_data, WRITE2D=True, ncfile='domain.nc', coord_system=True):

    # (optional) full 2D spatial extent
    if coord_system and 'X_axis' in elmdomain_data.keys():
        X_axis = elmdomain_data['X_axis']  # either longitude or projected geox, indiced as gridXID
        Y_axis = elmdomain_data['Y_axis']  # either latitude or projected geoy, indiced as gridYID
        XX = elmdomain_data['XX']  # longitude[Y_axis,X_axis], indiced as gridID
        YY = elmdomain_data['YY']  # latitude[Y_axis,X_axis], indiced as gridID

        # (optional) the following is for map conversion of 1D <==> 2D
        grid_id_arr = elmdomain_data['gridID']
        grid_xid_arr = elmdomain_data['gridXID']
        grid_yid_arr = elmdomain_data['gridYID']

    # standard
    lon_arr = elmdomain_data['xc']
    lat_arr = elmdomain_data['yc']
    xv_arr  = elmdomain_data['xv']
    yv_arr  = elmdomain_data['yv']
    if 'area' in elmdomain_data.keys():
        area_arr = elmdomain_data['area']
    elif 'area_km2' in elmdomain_data.keys():
        area_arr = elm_arcradians2_from_km2(elmdomain_data['area_km2'])
    else:
        print('Error: either "area" or "area_km2" not existed!')
        os.sys.exit(-1)
    mask_arr = elmdomain_data['mask']
    landfrac_arr = elmdomain_data['frac']

    # extra
    if 'area_km2' in elmdomain_data.keys():
        area_km2_arr = elmdomain_data['area_km2']

    nv = elmdomain_data['xv'].shape[1]
    if nv!= elmdomain_data['yv'].shape[1]:
        print('not equal vertices of xv: ', nv, ' and yv: ',elmdomain_data['yv'].shape[1])
        os.sys.exit(-2)

    # write to nc file
    file_name = ncfile
    print("The domain file to be write is " + file_name)

    # Open a new NetCDF file to write the domain information. For format, you can choose from
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    w_nc_fid = nc.Dataset(file_name, 'w', format='NETCDF4')
    if WRITE2D and len(grid_id_arr.shape)>1:
        w_nc_fid.title = '2D domain file for running ELM'
    else:
        w_nc_fid.title = '1D (unstructured) domain file for running ELM'
    w_nc_fid.Conventions = "CF-1.0"
    if coord_system and 'X_axis' in elmdomain_data.keys():
        w_nc_fid.note = "a 2d-grid coordinates system, ('y','x'), added, with 2d lon/lat variables. " + \
                    " With this, GridXID, GridYID, and GridID added for actual grids(nj,ni) " + \
                    " so that transform between 2d and 1d may be possible"

    # Create new dimensions of new coordinate system
    if coord_system and 'X_axis' in elmdomain_data.keys():

        # create the gridIDs, lon, and lat variable
        w_nc_fid.createDimension('x', len(X_axis))
        w_nc_fid.createDimension('y', len(Y_axis))
        w_nc_var = w_nc_fid.createVariable('x', np.float64, ('x'))
        w_nc_var.long_name = 'longitude of x-axis'
        w_nc_var.units = "degrees_east"
        w_nc_var.comment = "could be in unit of projected x such as meters"
        w_nc_fid.variables['x'][...] = X_axis

        w_nc_var = w_nc_fid.createVariable('y', np.float64, ('y'))
        w_nc_var.long_name = 'latitude of y-axis'
        w_nc_var.units = "degrees_north"
        w_nc_var.comment = "could be in unit of projected y such as meters"
        w_nc_fid.variables['y'][...] = Y_axis

        w_nc_var = w_nc_fid.createVariable('lon', np.float64, ('y','x'))
        w_nc_var.long_name = 'longitude of 2D land gridcell center (GCS_WGS_84), increasing from west to east'
        w_nc_var.units = "degrees_east"
        w_nc_fid.variables['lon'][...] = XX

        w_nc_var = w_nc_fid.createVariable('lat', np.float64, ('y','x'))
        w_nc_var.long_name = 'latitude of 2D land gridcell center (GCS_WGS_84), increasing from south to north'
        w_nc_var.units = "degrees_north"
        w_nc_fid.variables['lat'][...] = YY

    if WRITE2D and (not 1 in lon_arr.shape):
        w_nc_fid.createDimension('ni', lon_arr.shape[1])
        w_nc_fid.createDimension('nj', lon_arr.shape[0])
        w_nc_fid.createDimension('nv', nv)
    else:
        w_nc_fid.createDimension('ni', lon_arr.size)
        w_nc_fid.createDimension('nj', 1)
        w_nc_fid.createDimension('nv', nv)

    if coord_system and 'X_axis' in elmdomain_data.keys():
        # for 2D <--> 1D indices
        w_nc_var = w_nc_fid.createVariable('gridID', np.int32, ('nj','ni'))
        w_nc_var.long_name = '1d unique gridId in boxed range of coordinates y, x'
        w_nc_var.decription = "start from #0 at the lower left corner of the domain, covering all land and ocean gridcells" 
        w_nc_fid.variables['gridID'][...] = grid_id_arr

        w_nc_var = w_nc_fid.createVariable('gridXID', np.int32, ('nj','ni'))
        w_nc_var.long_name = 'gridId x in boxed range of coordinate x'
        w_nc_var.decription = "start from #0 at the lower left corner and from west to east of the domain, with gridID=gridXID+gridYID*len(y)" 
        w_nc_fid.variables['gridXID'][...] = grid_xid_arr

        w_nc_var = w_nc_fid.createVariable('gridYID', np.int32, ('nj','ni'))
        w_nc_var.long_name = 'gridId y in boxed range of coordinate y'
        w_nc_var.decription = "start from #0 at the lower left corner and from south to north of the domain, with gridID=gridXID+gridYID*len(y)" 
        w_nc_fid.variables['gridYID'][...] = grid_yid_arr

    #
    w_nc_var = w_nc_fid.createVariable('xc', np.float64, ('nj','ni'))
    w_nc_var.long_name = 'longitude of land gridcell center'
    w_nc_var.units = "degrees_east"
    w_nc_var.bounds = "xv"
    w_nc_var.comment = 'increasing from west to east'
    w_nc_fid.variables['xc'][...] = lon_arr

    w_nc_var = w_nc_fid.createVariable('yc', np.float64, ('nj','ni'))
    w_nc_var.long_name = 'latitude of land gridcell center'
    w_nc_var.units = "degrees_north"
    w_nc_var.comment = 'decreasing from north to south'
    w_nc_fid.variables['yc'][...] = lat_arr

    # create the XC, YC variable
    #
    w_nc_var = w_nc_fid.createVariable('xv', np.float64, ('nj','ni','nv'))
    w_nc_var.long_name = 'longitude of land gridcell verticles'
    w_nc_var.units = "degrees_east"
    w_nc_var.comment = 'increasing from west (-180) to east (180), vertices ordering anti-clock from left-low corner'
    w_nc_var = w_nc_fid.createVariable('yv', np.float64, ('nj','ni','nv'))
    w_nc_var.long_name = 'latitude of land gridcell verticles'
    w_nc_var.units = "degrees_north"
    w_nc_var.comment = 'decreasing from north to south, vertices ordering anti-clock from left-low corner'

    w_nc_fid.variables['xv'][...]= xv_arr
    w_nc_fid.variables['yv'][...]= yv_arr

    w_nc_var = w_nc_fid.createVariable('area', np.float64, ('nj','ni'))
    w_nc_var.long_name = "area of grid cell in radians squared" ;
    w_nc_var.coordinates = "xc yc" ;
    w_nc_var.units = "radian2" ;
    w_nc_fid.variables['area'][...] = area_arr

    if 'area_km2' in elmdomain_data.keys():
        w_nc_var = w_nc_fid.createVariable('area_km2', np.float64, ('nj','ni'))
        w_nc_var.long_name = "area of grid cell in kilometers squared" ;
        w_nc_var.coordinates = "xc yc" ;
        w_nc_var.units = "km^2" ;
        w_nc_fid.variables['area_km2'][...] = area_km2_arr

    w_nc_var = w_nc_fid.createVariable('mask', np.int32, ('nj','ni'))
    w_nc_var.long_name = "domain mask" ;
    w_nc_var.note = "unitless" ;
    w_nc_var.coordinates = "xc yc" ;
    w_nc_var.comment = "0 value indicates cell is not active" ;
    w_nc_fid.variables['mask'][...] = mask_arr

    w_nc_var = w_nc_fid.createVariable('frac', np.float64, ('nj','ni'))
    w_nc_var.long_name = "fraction of grid cell that is active" ;
    w_nc_var.coordinates = "xc yc" ;
    w_nc_var.note = "unitless" ;
    w_nc_var.filter1 = "error if frac> 1.0+eps or frac < 0.0-eps; eps = 0.1000000E-11" ;
    w_nc_var.filter2 = "limit frac to [fminval,fmaxval]; fminval= 0.1000000E-02 fmaxval=  1.000000" ;
    w_nc_fid.variables['frac'][...] = landfrac_arr

    w_nc_fid.close()  # close the new file


def domain_remask(input_pathfile='./share/domains/domain.lnd.r05_RRSwISC6to18E3r5.240328.nc', 
                     masked_pts={}, reorder_src=False, keep_duplicated=False, \
                     unlimit_xmin=False, unlimit_xmax=False, unlimit_ymin=False, unlimit_ymax=False,\
                     out2d=False, out='subdomain', ncwrite_coords=True):
    '''
    re-masking ELM domain grids, by providing centroids which also may be masked
        input_pathfile - source ELM domain nc file, with required variables of 'xc', 'yc', 'xv', 'yv', 'mask','frac'
                         grid system is lat/lon with dimension names of (nj, ni) of nj for latitude and ni for longitude
        masked_pts     - list of np arrays of 'xc', 'yc', 'mask' of user-provided grid-centroids
        reorder_src    - trunked source domain will re-order following that in masked_pts, 
                        otherwise NOT except 'mask' and maybe 'frac'
        out2d          - output data in 2D or flatten (so-called unstructured)
        unlimit_x(y)min(max) - x/y axis bounds are from source domain, when True; otherwise, from masked_pts
        out            - output type: 'subdomain' for outputing a domain-style dataset,
                                      'mask' for outputing a np.where style tuple of indices of grids, with a new mask
                                      'domain_ncwrite' for writing out a domain.nc file. 
        ncwrite_coords - when output type is 'domain_ncwrite', write coordinates and info OR not
    '''

    from watershed_workflow.elm_gridlocator import grids_nearest_or_within

    srcnc = nc.Dataset(input_pathfile, 'r')
    src_grids = {}
    src_grids['xc'] = srcnc['xc'][...]
    src_grids['yc'] = srcnc['yc'][...]
    if 'xv' in srcnc.variables.keys() \
        and 'yv' in srcnc.variables.keys():
        src_grids['xv'] = srcnc['xv'][...]
        src_grids['yv'] = srcnc['yv'][...]
        nv = src_grids['xv'].shape[-1]
        remask_approach = 'within'

        '''        
        if np.isnan(src_grids['xv']).any() or np.isnan(src_grids['yv']).any():
            xc=np.mean(src_grids['xv'],axis=-1)
            yc=np.mean(src_grids['yv'],axis=-1)
            idx = np.where((np.isnan(xc) | (np.isnan(yc))))
            x=np.delete(src_grids['xc'],idx)
            src_grids['xc']=np.reshape(x,(1,x.size))  # have to do so to maintain shape not changed
            y=np.delete(src_grids['yc'],idx)
            src_grids['yc']=np.reshape(y,(1,y.size))
            for v in range(nv):
                x=np.delete(src_grids['xv'][...,v],idx)
                x=np.reshape(x,(1,x.size,1))
                y=np.delete(src_grids['yv'][...,v],idx)
                y=np.reshape(x,(1,y.size,1))
                if v==0:
                    xv=x
                    yv=y
                else:
                    xv=np.dstack([xv,x])  # to maintain shape of data
                    yv=np.dstack([yv,y])
            src_grids['xv'] = xv
            src_grids['yv'] = yv
            '''            
                    
    else:
        remask_approach = 'nearest'

    subdomain, boxed_idx, grids_uid, grids_maskptsid = grids_nearest_or_within( \
                    src_grids=src_grids, masked_pts=masked_pts, \
                    remask_approach = remask_approach, \
                    unlimit_xmin=unlimit_xmin, unlimit_xmax=unlimit_xmax, \
                    unlimit_ymin=unlimit_ymin, unlimit_ymax=unlimit_ymax,\
                    out2d=out2d, reorder_src=reorder_src, keep_duplicated=keep_duplicated)

    if reorder_src and keep_duplicated:
        # in this case, sub-domain's xc/yc pair should be using those of masked_pts rather than of src_grids
        if 'xc' in masked_pts.keys(): subdomain['xc'] = masked_pts['xc']
        if 'yc' in masked_pts.keys(): subdomain['yc'] = masked_pts['yc']
        if 'xv' in masked_pts.keys(): subdomain['xv'] = masked_pts['xv']
        if 'yv' in masked_pts.keys(): subdomain['yv'] = masked_pts['yv']
        if 'km2perpt' in masked_pts.keys():
            subdomain['area_km2']=masked_pts['km2perpt']
        elif 'area_km2' in masked_pts.keys():
            subdomain['area_km2']=masked_pts['area_km2']

    # re-do frac of landed mask, if option ON, i.e. pts in a source grid are km2 of land
    landfrac = srcnc['frac'][...]
    if 'km2perpt' in masked_pts.keys() and not reorder_src:
        km2perpt = masked_pts['km2perpt']

        # grid area in km2, assuming a rectangle grid (TODO - refining this later)
        yc = srcnc['yc'][...].flatten()
        kmratio_lon2lat = np.cos(np.radians(yc))
        re_km = 6370.997
        xv = subdomain['xv']; yv = subdomain['yv']
        xv0=xv[...,0];xv1=xv[...,1]; xv2=xv[...,2];xv3=xv[...,3]
        yv0=yv[...,0];yv1=yv[...,1]; yv2=yv[...,2];yv3=yv[...,3]
        xside_km = (abs(xv0-xv1)+abs(xv2-xv3))/2.0*(math.pi*re_km/180.0)   #
        yside_km = (abs(yv0-yv2)+abs(yv1-yv3))/2.0*(math.pi*re_km/180.0*kmratio_lon2lat)
        area_km2 = xside_km*yside_km

        frac = landfrac.flatten()
        for i in range(len(grids_uid)):
            igrid = grids_uid[i]
            pts_counts = len(grids_maskptsid[igrid])
            frac[igrid] = pts_counts*km2perpt/area_km2[igrid]
        landarea_km2=area_km2.reshape(landfrac.shape)
        landfrac = frac.reshape(landfrac.shape)
        landfrac[np.where(landfrac>1.0)]=1.0

    # truncked landmask, fraction, and area
    landmask = srcnc['mask'][...][boxed_idx]
    landmask[...] = 0 # remasked
    landmask[np.where(subdomain['remasked'])]= 1

    landfrac = landfrac[...][boxed_idx]
    # in case landmask changed to 0, i.e. NOT a land cell, better reassign its fraction to 0
    # but not do so on 'area' which are for a whole grid
    landfrac[landmask==0] = 0.0

    landarea=srcnc['area'][...][boxed_idx]
    if 'area_km2' in srcnc.variables.keys(): landarea_km2=srcnc['area_km2'][...][boxed_idx]
    if 'area_LAEA' in srcnc.variables.keys(): landarea_km2=srcnc['area_LAEA'][...][boxed_idx]
    if 'area_LCC' in srcnc.variables.keys(): landarea_km2=srcnc['area_LCC'][...][boxed_idx]
     
    srcnc.close()
    
    # 
    # additional subdomain variables
    subdomain['area'] = landarea
    subdomain['mask'] = landmask
    subdomain['frac'] = landfrac
    if 'landarea_km2' in locals():
        subdomain['area_km2'] = landarea_km2

    if out == 'mask':
        if 'area_km2' in subdomain.keys():
            return boxed_idx, landmask, landfrac, subdomain['xc'], subdomain['yc'], subdomain['area_km2']
        else:
            return boxed_idx, landmask, landfrac, subdomain['xc'], subdomain['yc']
            # trunked domain with updated land mask and land fraction only
        
    elif out == 'subdomain':
        return subdomain
        # a new domain, may or may not same as boxed-truncked
    
    elif out == 'domain_ncwrite':
        file_name = input_pathfile.split('/')[-1]+'_remasked'
        print("new domain file is " + file_name)            
                 
        domain_ncwrite(subdomain, WRITE2D=out2d, ncfile=file_name, \
                       coord_system=ncwrite_coords)
    #          
#

''' 
 subset an ELM domain.nc by another user provided domain (but only needs: xc, yc, mask)
'''

def domain_subsetbymaskncf(srcdomain_pathfile='./share/domains/domain.lnd.r05_RRSwISC6to18E3r5.240328.nc', \
                        maskncf='./share/domains/domain.clm/domain.lnd.pan-arctic_CAVM.1km.1d.c241018.nc', \
                        maskncv='mask', masknc_area=np.empty((0,0)), reorder_src=False, keep_duplicated=False, \
                        unlimit_xmin=False, unlimit_xmax=False, unlimit_ymin=False, unlimit_ymax=False,\
                        out2D=True, out_type='subdomain'):
    
    # user provided mask file, which may or may not in same resolution as source domain,
    #                          and only needed are: xc, yc, mask
    mask_new = {}

    mask_f = nc.Dataset(maskncf,'r')
    if maskncv=='':
        #in this case, all points from maskncf will be extracted
        mask_v = mask_f['mask'][...]
        mask_checked = np.where(mask_v>=0)
    else:
        mask_v = mask_f[maskncv][...]
        mask_checked = np.where(mask_v==1)
    
    mask_new['xc']= mask_f['xc'][...][mask_checked] # this will flatten xc/yc, if in 2D
    mask_new['yc']= mask_f['yc'][...][mask_checked]
    mask_new['mask']= mask_v[mask_checked]
    if 'area_LAEA' in mask_f.variables.keys():
        # if need to convert new masked grid land area or fraction
        # the unit is km^2 per points included in the source domain 
        mask_new['area_LAEA'] = mask_f['area_LAEA'][...][mask_checked]
    elif 'area_LCC' in mask_f.variables.keys():
        # if need to convert new masked grid land area or fraction
        # the unit is km^2 per points included in the source domain 
        mask_new['area_LCC'] = mask_f['area_LCC'][...][mask_checked]
    elif 'area_km2' in mask_f.variables.keys():
        # if need to convert new masked grid land area or fraction
        # the unit is km^2 per points included in the source domain 
        mask_new['area_km2'] = mask_f['area_km2'][...][mask_checked]
    elif not np.array_equal(masknc_area,np.empty((0,0))) : 
        # if user provided
        mask_new['km2perpt'] = masknc_area
    
    #
    if out_type == 'subdomain':
        return domain_remask(input_pathfile=srcdomain_pathfile, \
                      masked_pts=mask_new, reorder_src=reorder_src, \
                      keep_duplicated=keep_duplicated, \
                      unlimit_xmin=unlimit_xmin, unlimit_xmax=unlimit_xmax, \
                      unlimit_ymin=unlimit_ymin, unlimit_ymax=unlimit_ymax, \
                      out2d=out2D, out=out_type)
    elif out_type == 'domain_ncwrite':
        domain_remask(input_pathfile=srcdomain_pathfile, \
                      masked_pts=mask_new, reorder_src=reorder_src, \
                      keep_duplicated=keep_duplicated, \
                      unlimit_xmin=unlimit_xmin, unlimit_xmax=unlimit_xmax, \
                      unlimit_ymin=unlimit_ymin, unlimit_ymax=unlimit_ymax, \
                      out2d=out2D, out=out_type)
    elif out_type == 'mask':
        return domain_remask(input_pathfile=srcdomain_pathfile, \
                      masked_pts=mask_new, reorder_src=reorder_src, \
                      keep_duplicated=keep_duplicated, \
                      unlimit_xmin=unlimit_xmin, unlimit_xmax=unlimit_xmax, \
                      unlimit_ymin=unlimit_ymin, unlimit_ymax=unlimit_ymax, \
                      out2d=out2D, out=out_type)
            
    # 
#--------------------------------------------------------------------------------------------------------

''' 
subset an ELM domain.nc by user provided paired latlons[[lats],[lons]] (but only needs: xc, yc)
'''
def domain_subsetbylatlon(srcdomain_pathfile='./share/domains/domain.lnd.r05_RRSwISC6to18E3r5.240328.nc', \
                          latlons=np.empty((0,0)), reorder_src=False, keep_duplicated=False, \
                          unlimit_xmin=False, unlimit_xmax=False, unlimit_ymin=False, unlimit_ymax=False, \
                          out2D=False, out_type='subdomain'):

    
    # new masked domain pts in np.array [[lats],[lons]]
    if (latlons.shape[0]<2):
        print('latlons must have paired location points: y/x or latitude/longidue',latlons.shape[0])
        return
    
    mask_new = {}
    mask_new['yc'] = latlons[0] # y or latitudes
    mask_new['xc'] = latlons[1] # x or longitudes
    mask_new['mask']= np.ones_like(latlons[0]) # assume all pts are masked land grids    
    #
    if out_type == 'subdomain':
        return domain_remask(input_pathfile=srcdomain_pathfile, \
                      masked_pts=mask_new, reorder_src=reorder_src, keep_duplicated=keep_duplicated, \
                      out2d=out2D, out=out_type)
    elif out_type == 'domain_ncwrite':
        domain_remask(input_pathfile=srcdomain_pathfile, \
                      masked_pts=mask_new, reorder_src=reorder_src, keep_duplicated=keep_duplicated, \
                      unlimit_xmin=unlimit_xmin, unlimit_xmax=unlimit_xmax, \
                      unlimit_ymin=unlimit_ymin, unlimit_ymax=unlimit_ymax,\
                      out2d=out2D, out=out_type)
    elif out_type == 'mask':
        return domain_remask(input_pathfile=srcdomain_pathfile, \
                      masked_pts=mask_new, reorder_src=reorder_src, keep_duplicated=keep_duplicated, \
                      out2d=out2D, out=out_type)
            
    #

#--------------------------------------------------------------------------------------------------------

def ncdata_subsetbynpwhereindex(npwhere_indices, npwhere_mask=np.empty((0,0)), npwhere_frac=np.empty((0,0)), \
                        newxc_box=np.empty((0,0)), newyc_box=np.empty((0,0)), newkm2_box=np.empty((0,0)), reordered_box=False, \
                        srcnc_pathfile='./lnd/clm2/surfdata_map/surfdata_0.5x0.5_simyr1850_c240308_TOP.nc', \
                        indx_dim=['lsmlat','lsmlon'], indx_dim_flatten=''):
    '''
    npwhere_box      - a tuple of np.where outputs from domain or surfdata, on 'indx_dim'
    npwheremask_box  - mask of npwhere_box, if provided
    indx_dim_flatten - if not empty, it's the new dimension name for flatten 'indx_dim' if 2d 
    '''
    
    # check if empty indices: 
    if len(npwhere_indices[0])>0:
        #        
        #
        ncfilein  = srcnc_pathfile
        ncfileout = srcnc_pathfile.split('/')[-1]
        ncfileout = ncfileout.split('.nc')[0]+'_subset.nc'
                    
        #subset and write to ncfileout
        with nc.Dataset(ncfilein,'r') as src, nc.Dataset(ncfileout, "w") as dst:
            # copy global attributes all at once via dictionary
            dst.setncatts(src.__dict__)
            # explicitly add a global attr for visualizing in gis tools
            dst.Conventions = "CF-1.0"

            # dimensions for dst
            for dname, dimension in src.dimensions.items():
                len_dimension = len(dimension)
                if dname in indx_dim:
                    if indx_dim_flatten!='':
                        # indx_dim are multiple-D, needs to flatten, i.e. forcing to be length of 1 
                        if indx_dim.index(dname)==len(indx_dim)-1: 
                            dim_len = sum(npwhere_mask>0)
                        else: 
                            continue
                    else:
                        dim_len = npwhere_indices[indx_dim.index(dname)].max() - \
                                  npwhere_indices[indx_dim.index(dname)].min() + 1
                    
                    if (dim_len>0):
                        len_dimension = dim_len
                    elif (dim_len==0):
                        dimension=None
                    
                    if indx_dim_flatten!='':
                        if len(dst.dimensions.keys())<=0:
                            dst.createDimension(indx_dim_flatten, len_dimension if not dimension.isunlimited() else None)
                        elif indx_dim_flatten not in dst.dimensions.keys():
                            #only need once, i.e. indx_dim will be flatten into index_dim_new
                            dst.createDimension(indx_dim_flatten, len_dimension if not dimension.isunlimited() else None)
                    else:
                        dst.createDimension(dname, len_dimension if not dimension.isunlimited() else None)                        
                    
                else:
                    dst.createDimension(dname, len_dimension if not dimension.isunlimited() else None)
            #   
        
            # all variables data
            for vname, variable in src.variables.items():

                # create dim for dst.variable
                if all(d in variable.dimensions for d in indx_dim):
                    vdim = np.array(variable.dimensions)
                    if indx_dim_flatten!='':
                        vdim = [d.replace(indx_dim[0], indx_dim_flatten) for d in vdim]
                        for i in range(1,len(indx_dim)):
                            vdim.remove(indx_dim[i])
                else:
                    vdim = variable.dimensions
    
                # create variables, but will update its values later 
                # NOTE: here the variable value size updated due to newly-created dimensions above
                dst.createVariable(vname, variable.datatype, vdim)
                # copy variable attributes all at once via dictionary after created
                dst[vname].setncatts(src[vname].__dict__)
                  
                # values
                print(vname)

                # dimensional slicing to extract data and fill into dst
                if all(d in variable.dimensions for d in indx_dim):
                    src_vdim = variable.dimensions
                    ix = range(src_vdim.index(indx_dim[0]),
                               src_vdim.index(indx_dim[-1])+1) # position of dims in 'indx_dim'
                    
                    # build a tuple of indices, with masked only in indx_dim
                    ix_mask = 0
                    # for unstructured surfdata, ix_mask will be fixed
                    if 'nj' in src_vdim and 'ni' in src_vdim:
                        if src.dimensions['nj'].size == 1: ix_mask = 1
                    elif len(npwhere_indices)>1:
                        if 'gridcell' in src_vdim or 'n' in src_vdim: ix_mask = 1
                    for i in range(len(src_vdim)):
                        if i==0:
                            newidx = (slice(None),)
                            if (i in ix and i==ix_mask+ix[0]) or \
                                (i in ix and 'gridcell' in src_vdim):
                                if indx_dim_flatten!='':
                                    newidx = (npwhere_indices[ix_mask][npwhere_mask>0],)
                                else:
                                    newidx = (npwhere_indices[ix_mask],)
                                    
                                # for structured 2D, ix_mask will move 1 dimension 
                                if 'nj' in src_vdim and 'ni' in src_vdim:
                                    if src.dimensions['nj'].size != 1: 
                                        ix_mask = ix_mask+1
                                else:
                                    if not 'gridcell' in src_vdim or not 'n' in src_vdim:
                                        ix_mask = ix_mask+1                          
          
                        else:
                            if (i in ix and i==ix_mask+ix[0]) or \
                                (i in ix and 'gridcell' in src_vdim):
                                if indx_dim_flatten!='':
                                    newidx = newidx+(npwhere_indices[ix_mask][npwhere_mask>0],)
                                else:
                                    newidx = newidx+(npwhere_indices[ix_mask],)

                                
                                # for structured 2D, ix_mask will move 1 dimension 
                                if 'nj' in src_vdim and 'ni' in src_vdim:
                                    if src.dimensions['nj'].size != 1: 
                                        ix_mask = ix_mask+1
                                elif not 'gridcell' in src_vdim or not 'n' in src_vdim:
                                        ix_mask = ix_mask+1
                            else:
                                newidx = newidx+(slice(None),)

                    varvals = src[vname][...][newidx]
                    #

                else:
                    varvals = src[vname][...]
                    #
                #
                if reordered_box:
                    if not np.array_equal(newxc_box,np.empty((0,0))) \
                        and (vname=='LONGXY' or vname=='xc'): 
                        varvals[...] = newxc_box
                    if not np.array_equal(newyc_box,np.empty((0,0))) \
                        and (vname=='LATIXY' or vname=='yc'): 
                        varvals[...] = newyc_box
                    if not np.array_equal(newkm2_box,np.empty((0,0))) \
                        and 'AREA' in vname:
                        varvals[...] = newkm2_box # km^2 
                
                #                            
                dst[vname][...] = varvals
                    
            # variable-loop        
                
        
        print('subsetting done successfully!')
        return ncfileout
        
    else:
        print('NO subsetting done due to indices NOT provided!')
        return None
    #
#
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------

def refine_surfdata(outdir='./', \
                    lnd_domain_file='domain.lnd.r05_RRSwISC6to18E3r5.240328.nc', \
                    fsurdat='surfdata_0.5x0.5_simyr1850_c240308_TOP.nc', \
                    flanduse_timeseries='landuse.timeseries_0.5x0.5_hist_simyr1850-2015_c240308.nc', \
                    userdomain='./domain.lnd.2687x1pt.coweeta.nc', \
                    domain_included=False):

    import glob
    try:
        from mpi4py import MPI
        HAS_MPI4PY=True
    except ImportError:
        HAS_MPI4PY=False

    #inputs of standard E3SM datasets
    e3sm_inputdata_root = DIN_LOC_ROOT
    if not e3sm_inputdata_root.endswith('/'):
        e3sm_inputdata_root=e3sm_inputdata_root+'/'

    if not fsurdat.startswith('./'):
        fsurdat_path= os.path.join(e3sm_inputdata_root, \
                            'lnd/clm2/surfdata_map')
    else:
        # relative to current workdir
        fsurdat_path = os.path.abspath(fsurdat)

    if not lnd_domain_file.startswith('./'):
        # treated as in standard E3SM inputdata structure
        lnd_domain_pathfile = os.path.join(e3sm_inputdata_root, \
                            'share/domains/domain.clm', \
                            lnd_domain_file)
    else:
        # relative to current workdir
        lnd_domain_pathfile = os.path.abspath(lnd_domain_file)

    #outputs
    output_path = './'
    SUBDOMAIN_REORDER = True  # True: masked file re-ordered by userdomain below, otherwise only mask and trunck
    KEEP_DUPLICATED = True

    # a domain.nc to aim to generate a new domain and surfdata
    # userdomain = './domain.lnd.2687x1pt.coweeta.nc'
    # user-grid area in km^2
    km2perpt = []
    lats=[]
    lons=[]
    #userdomain = './zone_mappings.txt' # OR, using a lat/lon paired pt list.
    #km2perpt = 1.0 # user-grid area in km^2
    # e.g.
    #  [f9y@baseline-login3 atm_forcing.datm7.GSWP3.0.5d.v2.c180716_ngee4]$ cat info_TFS_meq2_sites.txt 
    #    site_name lat lon km2(optional)
    #    MEQ2-MAT 68.6611 -149.3705 1.0
    #    MEQ2-DAT 68.607947 -149.401596 1.0
    #    MEQ2-PF 68.579315 -149.442279 1.0
    #    MEQ2-ST 68.606131 -149.505794 1.0
    if userdomain.endswith('.txt'):
        with open(userdomain) as f:
            dtxt=f.readlines()
            dtxt=filter(lambda x: x.strip(), dtxt)
            for d in dtxt:
                allgrds=np.array(d.split()[1:],dtype=float)
                lons.append(allgrds[1])
                lats.append(allgrds[0])
                if len(allgrds)==3: km2perpt.append(allgrds[2]) # optional

        f.close()
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        km2perpt = np.asarray(km2perpt)


    # continue for subsetting, if option is ON
    if userdomain.endswith('.nc'):
        idx_box, newmask_box, newfrac_box, newxc_box, newyc_box, newkm2_box = \
            domain_subsetbymaskncf( \
            srcdomain_pathfile=lnd_domain_pathfile, \
            maskncf=userdomain, masknc_area=np.empty((0,0)), maskncv='', \
            reorder_src=SUBDOMAIN_REORDER, \
            keep_duplicated=KEEP_DUPLICATED, \
            #unlimit_xmin=True, unlimit_xmax=True, unlimit_ymax=True, \
            out2D=False, out_type='mask')

    elif (len(lats)>0 and len(lats)==len(lons)):
        idx_box, newmask_box, newfrac_box, newxc_box, newyc_box = \
            domain_subsetbylatlon( \
            srcdomain_pathfile=lnd_domain_pathfile,  \
            latlons=np.asarray([lats, lons]), \
            reorder_src=SUBDOMAIN_REORDER, \
            keep_duplicated=KEEP_DUPLICATED, \
            out2D=False, out_type='mask')
            
        newkm2_box=np.empty((0,0))

    #
    allncfiles = [os.path.join(fsurdat_path,fsurdat)]
    if flanduse_timeseries!=None:
        allncfiles.append(os.path.join(fsurdat_path,flanduse_timeseries))
    if domain_included:
        allncfiles.append(lnd_domain_pathfile)

    allncout = []
    for ncfile in allncfiles:
        if ncfile=='': continue

        ncfile_dims = ['lsmlat', 'lsmlon']
        dims_new =''

        srfnc = nc.Dataset(ncfile)
        if 'surfdata' in ncfile or 'landuse' in ncfile:
            # the following is for surfdata, standard grid dims are ['lsmlat', 'lsmlon'], 
            # while unstructured dim name is 'gridcell'
            if 'gridcell' in srfnc.dimensions.keys():
                ncfile_dims = ['gridcell']
            dims_new= 'gridcell' # if want to reduce dimensions to single and given a new dimension

        elif 'domain' in ncfile:
            ncfile_dims = ['nj', 'ni']
            dims_new = '' 
            # if no change of dimensions, but will force dimension len to be 1, 
            # except the last which will be total unmasked grid number
            # (TODO - maybe another option of masked only?)

        else:
            print('NO subsetting done for surfdata, domain, or forcing data - file NOT exists')
            return
        srfnc.close()

        fout = ncdata_subsetbynpwhereindex(idx_box, npwhere_mask=newmask_box, npwhere_frac=newfrac_box, \
                                    newxc_box=newxc_box, newyc_box=newyc_box, newkm2_box=newkm2_box, \
                                    reordered_box=SUBDOMAIN_REORDER, \
                                    srcnc_pathfile=ncfile, \
                                    indx_dim=ncfile_dims, indx_dim_flatten=dims_new)

        #
        if outdir!='./': os.rename(fout, os.path.join(outdir,fout))
        allncout.append(os.path.join(outdir,fout))

    # output files
    return allncout

'''
if __name__ == '__main__':
    #set_e3sm_input('/Users/f9y/project_ww_elmats/test-e3sm-inputdata')
    set_e3sm_input('/Users/f9y/e3sm_inputdata')
    #allout = refine_surfdata(userdomain='./domain.lnd.2683x1pt_US-coweeta.nc')

    # extrac a pt of domain/surfdata
    allout = refine_surfdata( \
                    lnd_domain_file='domain.lnd.r05_RRSwISC6to18E3r5.240328_cavm1d.nc', \
                    fsurdat='surfdata_0.5x0.5_simyr1850_c240308_TOP_cavm1d.nc', \
                    flanduse_timeseries='landuse.timeseries_0.5x0.5_hist_simyr1850-2015_c240308_cavm1d.nc', \
                    userdomain='test_extract_pts.txt', \
                    domain_included=True)

    print(allout)
#    zisois, dzsois, zsoil =  soilcolumn(more_vertlayers=True, nlevgrnd=15)   
'''