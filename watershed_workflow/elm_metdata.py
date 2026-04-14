
import sys
from datetime import datetime
import glob
import numpy as np
from netCDF4 import Dataset

#----------------------------------------------------------------------------             
def putvar(ncfile, varname, varvals, varatts=''):
    # varatts: 'varname::att=att_val; varname::att=att_val; ...'
    try:
        f = Dataset(ncfile,'r+')
        #
        if varname=="": 
            print('FILE: '+ncfile+' Variable(s) is(are): ')
            for key in f.variables.keys(): 
                print (key)
            return -1
        #
        else:
            if varatts!='':
                varatts=varatts.split(";")
                for varatt in varatts:
                    v=varatt.split('::')[0].strip()
                    att=varatt.split('::')[1].strip().split('=')[0]
                    att_val=varatt.split('::')[1].strip().split('=')[1]
                    if (att=='add_offset' or att=='scale_factor'):
                        # this must be done before data written, 
                        # otherwise the written would use the original add_offset and scale_factor
                        # TIP: when data written, the input valule is in unpacked and the program will do packing
                        f.variables[v].setncattr(att,float(att_val))
                    else:
                        f.variables[v].setncattr(att,att_val)

            
            if isinstance(varvals, dict): #multiple dataset for corresponding same numbers of varname
                for v in varname:
                    if v in f.variables.keys():
                        if f.variables[v].dtype!=float:
                            val=np.ma.masked_where(np.isnan(varvals[v]), varvals[v])
                        else:
                            val=np.copy(varvals[v])
                        f.variables[v][...]=val
            else:                        #exactly 1 dataset in np.array for 1 varname
                v=varname[0]
                if v in f.variables.keys():
                    if f.variables[v].dtype!=float or \
                      (f.variables[v].dtype!=np.float64 or f.variables[v].dtype!=np.float32):
                        # nan is not dtype of integers
                        val=np.ma.masked_where(np.isnan(varvals), varvals)
                    else:
                        val=np.copy(varvals)
                    f.variables[v][...]=val
            #
            f.sync()
            f.close()
            
            return 0

    except:
        return -2

#----------------------------------------------------------------------------  
def elm_metdata_write(options, metdata, time_dim=0):
    
    """
    options.*
            "met_idir, default="./", \
                help = "E3SM met data directory for template files to look up")
            "met_odir, default="./", \
                help = "output data directory" )
            "nc_create", default=False, \
                help = "output new nc file", action="store_true")
            "nc_write", default=False, \
                help = "output to a existed nc file", action="store_true")
            "nc_write_mettype", default="", \
                help = "output to nc files in defined format (default = '', i.e., as original)")

    metdata[*]
            LATIXY
            LONGXY
            EDGEE
            EDGEW
            EDGEN
            EDGES
            YEAR
            DOY
            time
            ZBOT
            FSDS
            FLDS
            PRECTmms
            PRSF
            RH or QBOT
            TBOT
            WIND
     
    time_dim=0
            time dimension indice in met data, by default 0, i.e. [time, ...]
            Note: if not, may need to swap for either standard ELM format or cpl_bypass format     
    
    """
   
    if('site' in options.nc_write_mettype.lower() \
       or 'cplbypass_site' in options.nc_write_mettype.lower()): 
        vnames=['LONGXY','LATIXY','EDGEE','EDGEW','EDGEN','EDGES','time', \
                'ZBOT','TBOT', 'PRECTmms', 'RH', 'FSDS', 'FLDS', 'PSRF', 'WIND']
    else:
        vnames=['LONGXY','LATIXY','time', \
                'TBOT', 'PRECTmms', 'QBOT', 'FSDS', 'FLDS', 'PSRF', 'WIND']
    
    
    #--------------------------------------------------------------------------------------
    mdoy=[0,31,59,90,120,151,181,212,243,273,304,334,365] #monthly starting DOY of no-leap year
    if ('YEAR' not in metdata.keys() or 'DOY' not in metdata.keys()) \
        and ('time' in metdata.keys() and 'tunit' in metdata.keys()):
        # in this case both 'time' and 'tunit' must be included
        t = metdata['time']
        tunit = metdata['tunit']
        if ('days since' in tunit):
            t0_str=str(tunit).strip('days since')
            if(t0_str.endswith(' 00') and not t0_str.endswith(' 00:00:00')):
                t0_str=t0_str+':00:00'
            datetime0=datetime.strptime(t0_str,'%Y-%m-%d %X')
            y0=datetime0.year
            #m0=datetime0.month
            #d0=datetime0.day-1.0
            #t0=(y0-1)*365+mdoy[m0-1]+d0+datetime0.second/86400.0    # in days from 0000-00-00 00:00:00
            metdata['YEAR'] = np.floor(t/365.0)+y0
            metdata['DOY'] = t - np.floor(t/365.0)*365.0            # DOY is starting from 0, and hh:mm:ss as fraction of day

    TIME0_AS_CURRENT = False   
    # mathematically, 24:00:00 is same as 00:00:00,
    # but on different DOY and may cause issue when dividing dataset into individual months and even years. 
    if 'DOY' in metdata.keys():
        if metdata['DOY'][0] == int(metdata['DOY'][0]):
            # when DOY is exactly an integer, i.e. time is 00:00:00, assuming doy0 or time0 is included as current day
            TIME0_AS_CURRENT = True
            
            # truncating the last time data if its time is 00:00:00 
            # (this causes DOY as next day but only 1 t-data, and if this is last day of year, it will be next year) 
            if metdata['DOY'][-1] == int(metdata['DOY'][-1]):
                for v in metdata.keys():
                    if v not in ['time','YEAR','DOY', \
                        'ZBOT','TBOT','PRECTmms','RH','QBOT','FSDS', 'FLDS', 'PSRF', 'WIND']:
                        continue            
                    if time_dim==0: 
                        metdata[v]= metdata[v][:-1,...]
                    else:
                        metdata[v]= metdata[v][...,:-1]
        
        
    #--------------------------------------------------------------------------------------
    # save in ELM forcing data format
    if (options.nc_create or options.nc_write):
    
        
        met_type = options.nc_write_mettype
        if 'cplbypass' in met_type:
            NCOUT_CPLBYPASS=True
        else:
            NCOUT_CPLBYPASS=False
    
        if (options.nc_create and options.nc_write):
            print('Error: cannot have both "--nc_create" and "--nc_write" ')
            sys.exit(-1)
        elif (options.nc_create):
            print('Create new ELM forcing data ? ', options.nc_create)
        elif (options.nc_write):
            print('Write to existed ELM forcing data ? ', options.nc_write)
        
        # get a template ELM forcing data nc file 
        ncfilein = ''; ncfilein_prv = ''
        ncfilein_cplbypass = ''; ncfilein_cplbypass_prv = ''
        metidir = options.met_idir
        if metidir=='': metidir='./'
        if not metidir.endswith('/'): metidir = metidir+'/'

        metodir = options.met_odir
        if metodir=='': metodir='./'
        if not metodir.endswith('/'): metodir = metodir+'/'
        
        for varname in vnames:
            if varname in ['LONGXY','LATIXY','EDGEE','EDGEW','EDGEN','EDGES','time']: continue  #skip and always write if new output nc file
            if varname not in metdata.keys(): continue

            
            if 'GSWP3' in met_type:
                if (varname == 'FSDS'):
                    fdir = metidir+'/Solar3Hrly/'
                elif (varname == 'PRECTmms'):
                    fdir = metidir+'/Precip3Hrly/'
                else:
                    fdir = metidir+'/TPHWL3Hrly/'
                ncfilein = sorted(glob.glob("%s*.nc" % fdir))
                if len(ncfilein)>0: ncfilein = ncfilein[0]
                
                if '/cpl_bypass' not in metidir:
                    if 'daymet' in met_type:
                        fdirheader = metidir+'/cpl_bypass_template/GSWP3_daymet4_'+varname+'_'
                    else:
                        fdirheader = metidir+'/cpl_bypass_template/GSWP3_'+varname+'_'
                else:
                    if 'daymet' in met_type:
                        fdirheader = metidir+'/GSWP3_daymet4_'+varname+'_'
                    else:
                        fdirheader = metidir+'/GSWP3_'+varname+'_'
                ncfilein_cplbypass = sorted(glob.glob("%s*.nc" % fdirheader))
                if len(ncfilein)<=0 and len(ncfilein_cplbypass)<=0:
                    sys.exit('there is NO file as -'+fdirheader)
                ncfilein_cplbypass = ncfilein_cplbypass[0]
                
            elif 'site' in met_type.lower():
                fdir = './CLM1PT_data_template'
                # So 'metdir' must be full path, e.g. ../atm/datm7/CLM1PT_data/1x1pt_US-Brw
                ncfilein_cplbypass=fdir+'/all_hourly.nc'
                
                ncfilein = sorted(glob.glob("%s/????-??.nc" % fdir))
                ncfilein = ncfilein[0]
            elif 'crujra' in met_type or 'CRUJRA' in met_type:
                if (varname == 'FSDS'):
                    fdir = metidir+'/Solar6Hrly/'
                elif (varname == 'PRECTmms'):
                    fdir = metidir+'/Precip6Hrly/'
                else:
                    fdir = metidir+'/TPHWL6Hrly/'
                ncfilein = sorted(glob.glob("%s*.nc" % fdir))
                if len(ncfilein)>0: ncfilein = ncfilein[0]

                if '/cpl_bypass' not in metidir:                
                    fdirheader = metidir+'/cpl_bypass_full/CRUJRAV2.3.c2023.0.5x0.5_'+varname+'_'
                else:
                    fdirheader = metidir+'/CRUJRAV2.3.c2023.0.5x0.5_'+varname+'_'
                ncfilein_cplbypass = sorted(glob.glob("%s*.nc" % fdirheader))
                if len(ncfilein)<=0 and len(ncfilein_cplbypass)<=0:
                    sys.exit('there is NO file as -'+fdirheader)
                ncfilein_cplbypass = ncfilein_cplbypass[0]

            elif 'ERA5' in met_type or 'era5' in met_type:
                if (varname == 'FSDS'):
                    fdir = metidir+'/SolarHrly/'
                elif (varname == 'PRECTmms'):
                    fdir = metidir+'/PrecipHrly/'
                else:
                    fdir = metidir+'/TPHWLHrly/'
                ncfilein = sorted(glob.glob("%s*.nc" % fdir))
                if len(ncfilein)>0: ncfilein = ncfilein[0]
                
                if '/cpl_bypass' not in metidir:                
                    fdirheader = metidir+'/cpl_bypass_template/Daymet_ERA5.1km_'+varname+'_'
                else:
                    fdirheader = metidir+'/Daymet_ERA5.1km_'+varname+'_'
                ncfilein_cplbypass = sorted(glob.glob("%s*.nc" % fdirheader))
                if len(ncfilein)<=0 and len(ncfilein_cplbypass)<=0:
                    sys.exit('there is NO file as -'+fdirheader)
                ncfilein_cplbypass = ncfilein_cplbypass[0]

            elif 'ATS-subdaily' in met_type:
                if (varname == 'FSDS'):
                    fdir = metidir+'/SolarSubdaily/'
                elif (varname == 'PRECTmms'):
                    fdir = metidir+'/PrecipSubdaily/'
                else:
                    fdir = metidir+'/TPHWLSubdaily/'
                ncfilein = sorted(glob.glob("%s*.nc" % fdir))
                if len(ncfilein)>0: ncfilein = ncfilein[0]
                
                if '/cpl_bypass' not in metidir:                
                    fdirheader = metidir+'/cpl_bypass_template/ATS-subdaily_'+varname+'_'
                else:
                    fdirheader = metidir+'/ATS-subdaily_'+varname+'_'
                ncfilein_cplbypass = sorted(glob.glob("%s*.nc" % fdirheader))
                if len(ncfilein)<=0 and len(ncfilein_cplbypass)<=0:
                    sys.exit('there is NO file as -'+fdirheader)
                ncfilein_cplbypass = ncfilein_cplbypass[0]
            
            # new template nc file, and has a prv file 
            if options.nc_write and ncfilein_prv != ncfilein and \
                                    ncfilein_prv != '':
                options.nc_create = True
                options.nc_write = False
            if options.nc_write and ncfilein_cplbypass_prv != ncfilein_cplbypass and \
                                    ncfilein_cplbypass_prv != '':
                options.nc_create = True
                options.nc_write = False
                        
            # new met nc files to create or write
            
            #(1) in non-cplbypass format
            if (ncfilein!='' and len(ncfilein))>0:
                if 'site' in met_type.lower():
                    tname = 'time'
                    for iyr in range(int(np.min(metdata['YEAR'])), int(np.max(metdata['YEAR']))+1):
                        for imon in range(1,13):
                            ncfileout = metodir+str(iyr)+'-'+str(imon).zfill(2)+'.nc'
                            
                            if not TIME0_AS_CURRENT:
                                tidx = np.argwhere((metdata['YEAR']==iyr) & 
                                               (metdata['DOY']>mdoy[imon-1]) & (metdata['DOY']<=mdoy[imon])) #DOY starting from 1
                            else:
                                tidx = np.argwhere((metdata['YEAR']==iyr) & 
                                               (metdata['DOY']>=mdoy[imon-1]) & (metdata['DOY']<mdoy[imon]))  #DOY starting from 1
                                
                            t_jointed=metdata['time'][tidx]
                            sdata = metdata[varname][tidx]
                            sdata = sdata[..., np.newaxis] # in 'Site' forcing-data format, dimension in (time, lat, lon), while 'sdata' is in (time, gridcell) 
                            
                            # create nc file
                            if options.nc_create:
                                with Dataset(ncfilein,'r') as src, Dataset(ncfileout, "w") as dst:
                                    
                                    #------- new nc dimensions
                                    for dname, dimension in src.dimensions.items():
                                        len_dimension = len(dimension)
                                        if dname == 'LATIXY': len_dimension = len(metdata['LATIXY'])
                                        if dname == 'LONGXY': len_dimension = len(metdata['LONGXY'])                                    
                                        if dname == tname: len_dimension = len(t_jointed)
                                        
                                        dst.createDimension(dname, len_dimension if not dimension.isunlimited() else None)

                                    # variables
                                    for vname in src.variables.keys():
                                        vtype = src.variables[vname].datatype
                                        vdim = src.variables[vname].dimensions
                                        dst.createVariable(vname, vtype, vdim)
                                        dst[vname].setncatts(src[vname].__dict__)
    
                                # time and locations
                                try:
                                    tunit = Dataset(ncfilein).variables[tname].getncattr('units')
                                    t0=str(tunit.lower()).strip('days since')
                                    if (t0.endswith(' 00')):
                                        t0=t0+':00:00'
                                    elif(t0.endswith(' 00:00')):
                                        t0=t0+':00'
                                    t0=datetime.strptime(t0,'%Y-%m-%d %X') # time must be in format '00:00:00'
                                    yr0=np.floor(t_jointed[0]/365.0)+1
                                    t=t_jointed - (yr0-1)*365.0 - mdoy[imon-1]
                                    tunit = tunit.replace(str(t0.year).zfill(4)+'-', str(int(iyr)).zfill(4)+'-')
                                    tunit = tunit.replace('-'+str(t0.month).zfill(2)+'-', '-'+str(imon).zfill(2)+'-')
                                    if(tunit.endswith(' 00') and not tunit.endswith(' 00:00:00')):
                                        tunit=tunit+':00:00'
                                except Exception as e:
                                    print(e)
                                    tunit = ''
                                   
                                error=putvar(ncfileout, [tname], t, varatts=tname+'::units='+tunit)
                                error=putvar(ncfileout,['LONGXY'], metdata['LONGXY'])
                                error=putvar(ncfileout,['LATIXY'], metdata['LATIXY'])
                                error=putvar(ncfileout,['EDGEE'], metdata['EDGEE'])
                                error=putvar(ncfileout,['EDGEW'], metdata['EDGEW'])
                                error=putvar(ncfileout,['EDGEN'], metdata['EDGEN'])
                                error=putvar(ncfileout,['EDGES'], metdata['EDGES'])
                                
                                
                            #end of if create nc file
                            #
                            error=putvar(ncfileout, [varname], sdata)
                            if error!=0: sys.exit('nfmod.putvar WRONG-'+varname+'-'+ncfileout)
                        #end of month loop
                    #end of year loop
                    if (not NCOUT_CPLBYPASS and options.nc_create): # If not create cpl_bypass format file
                        options.nc_write = True
                        options.nc_create= False
            
            #(2) in cplbypass format (NOTE: can output both original and cplbypass)
            if (NCOUT_CPLBYPASS): print('cpl_bypass template file: '+ ncfilein_cplbypass)
            if (NCOUT_CPLBYPASS and ncfilein_cplbypass!=''):
                if 'site' in met_type.lower() or \
                   'GSWP3' in met_type or \
                   'crujra' in met_type.lower() or \
                   'ERA5' in met_type or \
                   'ATS-subdaily' in met_type: 
                    
                    if 'site' in met_type.lower():
                        ncfileout_cplbypass=metodir+'all_hourly.nc'
                    elif 'gswp3' in met_type.lower() or \
                         'crujra' in met_type.lower() or \
                         'era5' in met_type.lower() or \
                         'ATS-subdaily' in met_type: 
                        ncfileout_cplbypass=metodir+ncfilein_cplbypass.split('/')[-1]
                        if 'daymet' in met_type.lower():
                            ncfileout_cplbypass=ncfileout_cplbypass.replace('1901', '1980')
                    else:
                        print('currently only support types of cpl_bypass: Site or GSWP3* or CRUJRA or ERA5* or ATS-subdaily')
                        sys.exit(-1)
                        
                    tname = 'DTIME'
                    t_jointed=metdata['time']
                        
                    sdata = metdata[varname]
                    
                    if (options.nc_create):
                        with Dataset(ncfilein_cplbypass,'r') as src, Dataset(ncfileout_cplbypass, "w") as dst:
                            
                            #------- new nc dimensions
                            for dname, dimension in src.dimensions.items():
                                len_dimension = len(dimension)
                                if dname == 'n': len_dimension = len(metdata['LONGXY'])
                                if dname == tname: len_dimension = len(t_jointed)
                                
                                dst.createDimension(dname, len_dimension if not dimension.isunlimited() else None)
    
                            # variables
                            for vname in src.variables.keys():
                                vtype = src.variables[vname].datatype
                                vdim = src.variables[vname].dimensions
                                dst.createVariable(vname, vtype, vdim)
                                dst[vname].setncatts(src[vname].__dict__)
    
    
                        # ONLY create new nc ONCE, and save file I/O name
                        options.nc_create = False
                        options.nc_write = True
                        ncfilein_cplbypass_prv = ncfilein_cplbypass
                        #ncfileout_cplbypass_prv= ncfileout_cplbypass
                        
                        # time
                        try:
                            # tunit from template file, which need to be replaced by that from metdata
                            tunit = Dataset(ncfilein_cplbypass).variables[tname].getncattr('units')
                            t0=str(tunit.lower()).strip('days since')
                            if(t0.endswith(' 00:00:00')):
                                t0=t0
                            elif(t0.endswith(' 00:00')):
                                t0=t0+':00'
                            elif (t0.endswith(' 00')):
                                t0=t0+':00:00'
                            else:
                                t0=t0+' 00:00:00'
                            t0=datetime.strptime(t0,'%Y-%m-%d %X') # time must be in format '00:00:00'
                            
                            
                            if 'tunit' in metdata.keys():
                                if 'days since' in metdata['tunit']:
                                    #t_jointed alread in unit of days since ...
                                    tunit = metdata['tunit']
                                    t = t_jointed
                            
                            else:
                                #default, assuming days since year 0
                                yr0=np.floor(t_jointed[0]/365.0)+1
                                t=t_jointed - (yr0-1)*365.0
                                tunit = tunit.replace(str(t0.year).zfill(4)+'-', str(int(yr0)).zfill(4)+'-')
                                tunit = tunit.replace('-'+str(t0.month).zfill(2)+'-', '-01-')
                            
                            if(tunit.endswith(' 00') and not tunit.endswith(' 00:00:00')):
                                tunit=tunit+':00:00'
                        except Exception as e:
                            print(e)
                            tunit = ''
                        error=putvar(ncfileout_cplbypass, [tname], np.asarray(t), varatts=tname+'::units='+tunit)
                        # varatts must in format: 'varname::att=att_val; varname::att=att_val; ...'
                        if error!=0: sys.exit('nfmod.putvar WRONG')
                        
                        error=putvar(ncfileout_cplbypass,['LONGXY'], metdata['LONGXY'])
                        error=putvar(ncfileout_cplbypass,['LATIXY'], metdata['LATIXY'])
                        if 'site' in met_type.lower():
                            error=putvar(ncfileout_cplbypass,['start_year'], np.floor(t[0]/365.0)+1)
                            error=putvar(ncfileout_cplbypass,['end_year'], np.floor(t[-1]/365.0)+1)

                        else:
                            # except for 'site', other type of cpl_bypass requires zone_mapping.txt file
                            lon = metdata['LONGXY']
                            lat = metdata['LATIXY']
                            g_zno = 1
                            f = open(metodir+'zone_mappings.txt', 'w')
                            for ig in range(len(lon)):
                                f.write('%12.5f ' % lon[ig] )
                                f.write('%12.6f ' % lat[ig] )
                                f.write('%5d ' % g_zno )
                                f.write('%5d ' % (ig+1) )
                                f.write('\n')
                            f.close()
                            
                            if 'ATS-subdaily' in met_type:
                                error=putvar(ncfileout_cplbypass,['start_year'], metdata['YEAR'][0])
                                error=putvar(ncfileout_cplbypass,['end_year'], metdata['YEAR'][-1])


                                            
                    # scaling data as initeger
                    if (varname=='PRECTmms'):
                        data_ranges = [0.0, 0.08000]
                    elif (varname=='FSDS'):
                        data_ranges = [0.0, 2000.0]
                    elif (varname=='TBOT'):
                        data_ranges = [175.0, 350.0]
                    elif (varname=='RH'):
                        data_ranges = [5.0, 100.0]
                    elif (varname=='QBOT'):
                        data_ranges = [0.0001, 0.10]
                    elif (varname=='FLDS'):
                        data_ranges = [0.0, 1000.0]
                    elif (varname=='PSRF'):
                        data_ranges = [40000.0, 120000.0]
                    elif (varname=='WIND'):
                        data_ranges = [0.0, 100.0]
                    else:
                        continue
                    
                    add_offset = (data_ranges[1]+data_ranges[0])/2.0
                    scale_factor = (data_ranges[1]-data_ranges[0])*1.1/(2**15)
                    
                    # need to adjust offset to make sure short integer zero is zero absolutely
                    zeroint = np.int32(-add_offset/scale_factor)
                    zerotiny = add_offset+scale_factor*float(zeroint)
                    add_offset = add_offset - zerotiny 
                    
                    # TIP: when data written, the input valule is in unpacked and the python nc4 program will do packing
                    # the following line IS WRONG
                    # varvals = (sdata_jointed-add_offset)/scale_factor # this IS WRONG when written, i.e. NOT NEEDED absolutely
                    
                    # note if in CPL_BYPASS forcing-data format, dimension is in (gridcell, DTIME)
                    # but source met data may not.
                    varvals = sdata
                    if time_dim!=len(sdata.shape)-1:
                        varvals = np.swapaxes(sdata, time_dim, -1)
                    
                    # varatts must in format: 'varname::att=att_val; varname::att=att_val; ...'
                    varatts = varname+'::add_offset='+str(add_offset)+ \
                        ';'+varname+'::scale_factor='+str(scale_factor)
                    error=putvar(ncfileout_cplbypass, [varname], varvals, varatts=varatts)
                    #error=nfmod.putvar(ncfileout_cplbypass, [varname], varvals)
                    if error!=0: sys.exit('nfmod.putvar WRONG - '+varname+', - '+ncfileout_cplbypass)
                    
                
                else:
                    print('TODO: CPL_BYPASS format other than "Site/GSWP3/crujra/ERA5" not yet supported')
                    sys.exit(-1)
            #
            elif(NCOUT_CPLBYPASS):
                print('CPL_BYPASS format output required but cannot find a template netcdf file, such as: all_hourly.nc or GSWP3_TBOT_1901-2014_z14.nc')
                sys.exit(-1)
        # for varname in vnames
    
    # if options.nc_create or options.nc_write
    print('DONE!')
    
    return 0

#

# ---------------------------------------------------------------

