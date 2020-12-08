"""
Utilities for pulling soil and geology information from various data source

- Get soil property such as permeability, porosity, and van Genutchen parameters from SSURGO/gSSURGO/gNATSGO .gdb files (https://nrcs.app.box.com/v/soils/folder/17971946225).
- Get geology property (i.e., permeability and porosity) from GLHYMPS v2 (https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/TTJNIU) 

Usage:
-----
    soil_prop = get_soil_property_from_SSURGO(SSURGO_gdb_file, sqlite_path, rosetta_model = 3, outfile= None)
    geol_prop = get_GLHYMPSv2_property(GLHYMPS_file, outfile = None)

Author: Pin Shuai (pin.shuai@pnnl.gov)
Date: 11/01/2020
"""


import numpy as np, pandas as pd
import fiona
import geopandas as gpd
import re
import workflow.ui
workflow.ui.setup_logging(1,None) # default is printing logging.INFO

from ANN_Module import PTF_MODEL
from DB_Module import DB
import time as T

import logging
import matplotlib.pyplot as plt

def read_attr_tables_from_gdb(gdb_file):
    """
    Read geodatabase file using geopandas and get the attribute tables (ie. chorizon and component). Preprocess tables to get soil layer thickness and porosity. For missing porosity values, assume porosity = saturated water content.
    Parameters:
    ----
    gdb_file: path to gSSURGO or gNATSGO .gdb file.
    
    Returns:
    ----
    df_chorizon: preprocessed horizon dataframe
    df_component: preprocessed component dataframe
    """
    df_chorizon = gpd.read_file(gdb_file, driver='FileGDB', layer='chorizon')
    df_component = gpd.read_file(gdb_file, driver='FileGDB', layer='component')
    
    # rename columns    
    horizon_rename_list = {'hzdept_r':'top depth [cm]', 'hzdepb_r':'bot depth [cm]', 'ksat_r':'sat K [um/s]', 
                          'sandtotal_r':'total sand pct [%]', 'silttotal_r':'total silt pct [%]', 
                           'claytotal_r':'total clay pct [%]',
                         'dbthirdbar_r':'bulk density [g/cm^3]', 'partdensity':'particle density [g/cm^3]',
                          'wsatiated_r':'saturated water content [%]'}

    df_chorizon.rename(columns = horizon_rename_list, inplace = True)
    df_component.rename(columns={'comppct_r':'component pct [%]'}, inplace = True)
    
    # preprocess data
    df_chorizon['thickness [cm]'] = df_chorizon['bot depth [cm]'] - df_chorizon['top depth [cm]']

    df_chorizon['porosity [-]'] = 1 - df_chorizon['bulk density [g/cm^3]']/df_chorizon['particle density [g/cm^3]']
    # assume null porosity = saturated water content
    df_chorizon.loc[pd.isnull(df_chorizon['porosity [-]']), 'porosity [-]'] = df_chorizon.loc[pd.isnull(df_chorizon['porosity [-]']), 'saturated water content [%]']/100    
    
    logging.info(f'found {len(df_component["mukey"].unique())} unique MUKEYs.')
    return df_chorizon, df_component        

def get_aggregated_mukey_values(df_chorizon, df_component):
    """
    Aggregate component values by component percentage to get MUKEY property.
    
    Parameters:
    ----
    df_chorizon: chorizon table.
    df_component: component table.
    
    Returns:
    ----
    mukey_agg_df: aggregated mukey property.
    
    """
    # get column name and variables. These are hard coded for now.
    horizon_selected_cols = ['cokey', 'chkey', 'thickness [cm]', 'top depth [cm]', 'bot depth [cm]', 'sat K [um/s]', 
                             'total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]',
                           'bulk density [g/cm^3]', 'particle density [g/cm^3]', 'porosity [-]']
    mukey_agg_var = ['mukey', 'agg_Ksat [um/s]', 'agg_sand_pct [%]', 
           'agg_silt_pct [%]', 'agg_clay_pct [%]', 'agg_bulk_density [g/cm^3]', 'agg_porosity [-]', 'agg_soil_depth [cm]'
          ]
    depth_ave_var = ['sat K [um/s]', 'total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]', 'bulk density [g/cm^3]', 'porosity [-]']
    area_ave_var = depth_ave_var + ['soil depth [cm]']
    
    mukey_agg_df = pd.DataFrame(columns = mukey_agg_var)
    
    for imukey in df_component['mukey'].unique()[:]:
        imukey_df = df_component.loc[df_component['mukey'] == imukey, ['mukey', 'cokey', 'component pct [%]']]
        
        # depth-average based on layer thickness
        comp_agg_df = get_aggregated_component_values(df_chorizon, imukey_df, horizon_selected_cols, area_ave_var, depth_ave_var)
        
        imukey_cokey_df = pd.merge(imukey_df, comp_agg_df, how = 'outer', left_on = 'cokey', right_on = 'cokey')

        # area-average based on component pct
        area_agg_value = []
        area_agg_value.append(imukey)
        for ivar in area_ave_var[:]:
            idf = imukey_cokey_df[['component pct [%]', ivar]].dropna()
            if idf.empty:
                ivalue = np.nan
            else:
                ivalue = sum(idf['component pct [%]']/idf['component pct [%]'].sum()*idf[ivar])
            area_agg_value.append(ivalue)

        idf_mukey_agg = pd.DataFrame(np.array(area_agg_value).reshape(1, len(area_agg_value)), columns=mukey_agg_var)

        mukey_agg_df = mukey_agg_df.append(idf_mukey_agg)

    mukey_agg_df[mukey_agg_var[1:]] = mukey_agg_df[mukey_agg_var[1:]].apply(pd.to_numeric, errors = 'coerce')
    
    return mukey_agg_df

def get_aggregated_component_values(df_chorizon, imukey_df, horizon_selected_cols, area_ave_var, depth_ave_var):
    """
    Aggregate horizon value by layer thickness to get component property. 
    
    Parameters:
    ----
    df_chorizon: horizon table
    imukey_df: individual mukey df contains component keys
    horizon_selected_cols: selected columns for horizon table
    area_ave_var: area-averaged variables
    depth_ave_var: depth-averaged variables
    
    Returns:
    ----
    comp_agg_df: aggregated component property
    """
#     area_ave_var = depth_ave_var + ['soil depth [cm]']
    comp_list = area_ave_var + ['cokey']
    
    comp_agg_df = pd.DataFrame(columns = comp_list)
    
    for icokey in imukey_df['cokey'].values[:]:
        
        if icokey in df_chorizon['cokey'].unique():

            idf_horizon = df_chorizon.loc[df_chorizon['cokey'] == icokey, horizon_selected_cols]

            depth_agg_value = []
            
            # depth-average based on layer thickness
            for ivar in depth_ave_var:
                idf = idf_horizon[['thickness [cm]', ivar]].dropna()
                if idf.empty:
                    ivalue = np.nan
                else:
                    ivalue = sum(idf['thickness [cm]']/idf['thickness [cm]'].sum()*idf[ivar])
                depth_agg_value.append(ivalue)

            idepth = idf_horizon['bot depth [cm]'].dropna().max()

            depth_agg_value.append(idepth)
            depth_agg_value.append(icokey)

            idf_comp = pd.DataFrame(np.array(depth_agg_value).reshape(1, len(depth_agg_value)), columns=comp_list)
            
            
            idf_comp[area_ave_var] = idf_comp[area_ave_var].apply(pd.to_numeric, errors = 'coerce')
            
            # normalize sand/silt/clay pct to make the sum(%sand, %silt, %clay)=1
            sum_soil = idf_comp.loc[:, 'total sand pct [%]':'total clay pct [%]'].sum().sum()
            if sum_soil !=100:
                for isoil in ['total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]']:
                    idf_comp[isoil] = idf_comp[isoil]/sum_soil*100
            
            comp_agg_df = comp_agg_df.append(idf_comp)
            
    return comp_agg_df

def get_vgm_from_Rosetta(data, sqlite_path, model_type):
    """
    Return van Genutchen model parameters using Rosetta v3 model ( Zhang and Schaap, 2017 WRR).
    
    Parameters: 
    -----
    data: numpy array
        Array contains input data in dim(nvar, nsamples)
    model_type: int. 2 or 3
        Rosetta model type. 2--using sand/silt/clay pct; 3--using sand/silt/clay pct + bulk density
        
    Returns:
    ----
    vgm_new: numpy array
        van Genutchen model parameters correspond to model type.
    """
    with DB(host='localhost', user='root', db_name='Rosetta', sqlite_path=sqlite_path) as db:
        
        #convert data from 1d array to nd matrix if necessary
        if data.ndim ==1:
            logging.info(f"data is 1-D array, reshaping to (nvar,1)")
            data = data.reshape(data.shape[0],1)
        # choose the right model corresponding to data inputs
        ptf_model=PTF_MODEL(model_type, db) 
        logging.info(f"--Processing--\n get van Genutchen parameters from Rosetta (model {model_type})")
        # with sum_data=False you get the raw output WITHOUT Summary statistics

        res_dict = ptf_model.predict(data, sum_data=True) 
            
        vgm_name=res_dict['var_names']

        # res_dict['sum_res_mean'] output log10 of VG-alpha,VG-n, and Ks
        vgm_mean=res_dict['sum_res_mean']
        vgm_new=np.stack((vgm_mean[0],vgm_mean[1],10**vgm_mean[2],10**vgm_mean[3],10**vgm_mean[4]))
        # transpose to match the data input format
        vgm_new=vgm_new.transpose()
        logging.debug(f'output van Genutchen parameters:')
        logging.debug(f'\n|theta_r [cm^3/cm^3]|theta_s [cm^3/cm^3]|alpha [1/cm]| n [-] |Ks [cm/day]|\n{vgm_new}')
    return vgm_new

def get_soil_property_from_SSURGO(gdb_file, sqlite_path, rosetta_model = 3, outfile= None):
    """
    Aggregate soil property for each MUKEY from SSURGO/NATSGO; get van Genutchen model parameters using Rosetta v3.
    
    Parameters:
    ----
    gdb_file: SSURGO or NATSGO gdb file path
    
    sqlite_path: relative path to Rosetta.sqlite. 
                It is located in watershed-workflow/workflow_tpls/rosetta_v3.0beta/sqlite/Rosetta.sqlite
    rosetta_model: int
                Type of Rosetta model. Default is 3 (i.e., need sand/silt/clay pct and bulk density)
    outfile: exported csv file
    
    Returns:
    ----
    soil_prop: datafrme
    """
    T0=T.time()

    
    vgm_input_header = ['agg_sand_pct [%]', 'agg_silt_pct [%]', 'agg_clay_pct [%]', 'agg_bulk_density [g/cm^3]']
    vgm_output_header = ['theta_r [cm^3/cm^3]', 'theta_s [cm^3/cm^3]', 'alpha [1/cm]', 'n [-]', 'Rosetta_Ks [cm/day]']
    
    logging.info(f'reading GDB file...')
    if gdb_file is not None:
        logging.info(f'found GDB file at {gdb_file}')
    else:
        raise RuntimeError(f'SSURGO gdb file not found. Try download it from https://nrcs.app.box.com/v/soils/folder/108070520682')
    
    logging.info(f'reading attribute tables (e.g., chorizon, compnent) from SSURGO')
    df_chorizon, df_component = read_attr_tables_from_gdb(gdb_file)
    
    logging.info(f'getting aggregated MUKEY properties...')
    mukey_agg_df = get_aggregated_mukey_values(df_chorizon, df_component)
    
    df = mukey_agg_df.dropna(subset = vgm_input_header).copy()    
    # need to transpose the data so that the array have the shape (nvar, nsample) 
    data = df[vgm_input_header].values.T
    
    vgm = get_vgm_from_Rosetta(data, sqlite_path, model_type = rosetta_model)
    
    vgm_df = pd.DataFrame(vgm, columns=vgm_output_header)
    vgm_df['mukey'] = df['mukey'].values
    soil_prop = pd.merge(mukey_agg_df, vgm_df, how = 'outer', left_on = 'mukey', right_on = 'mukey')
    
    # get permeability [m^2] from hydraulic conductivity
    soil_prop['agg_k [m^2]'] = soil_prop['agg_Ksat [um/s]']*1e-6*9.1e-8
    soil_prop['Rosetta_k [m^2]'] = soil_prop['Rosetta_Ks [cm/day]']/100/86400*9.1e-8
    logging.info(f"--Processing done-- \n time spent:{T.time()-T0} s")
    
    if outfile == None:
        statename = re.split('_|\.', gdb_file)[-2]
        fname = f'./soil_prop_{statename}.csv'
    else: 
        fname = outfile
    logging.info(f"saving soil parameters to {fname}")
    soil_prop.to_csv(fname, index = False)
    
    return soil_prop

def get_GLHYMPSv2_property(shp_file, outfile = None):
    """
    Retrieve geology property such as K and porosity from GLHYMPS.
    
    Arguments:
    ----
        shp_file: path to shapefile of GLHYMPS.
        outfile: path to exported csv file.
    """
    with fiona.open(shp_file, mode='r') as fid:
        profile = fid.profile
        geology = [r for (i,r) in fid.items()]
    # get attributes from shapefile
    obj_ids = [shp['properties']['OBJECTID_1'] for shp in geology]
    k = [10**(shp['properties']['logK_Ferr_']/100) for shp in geology] # logK_Ferr: permeability without permafrost [logk*100]
    k_std = [shp['properties']['K_stdev_x1']/100 for shp in geology] # std of logK_Ferr
    porosity = [shp['properties']['Porosity_x']/100 for shp in geology]    
    
    geol_property = pd.DataFrame({'ID':obj_ids, 'permeability [m^2]':k, 'logk_stdev [-]':k_std, 'porosity [-]':porosity})
    
    # saving geology parameter file
    if outfile == None:
        fname = './geol_prop.csv'
    else:
        fname = outfile
    logging.info(f"saving geology parameters to {fname}")
    geol_property.to_csv(fname, index = False)
    
    geol_property.set_index('ID', inplace = True)
    
    return geol_property