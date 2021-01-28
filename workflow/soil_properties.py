"""
Leverages Rosetta to go from soil properties to van Genucten curves.

- Get soil property such as permeability, porosity, and van Genutchen parameters from SSURGO/gSSURGO/gNATSGO .gdb files (https://nrcs.app.box.com/v/soils/folder/17971946225).
- Get geology property (i.e., permeability and porosity) from GLHYMPS v2 (https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/TTJNIU) 

Usage:
-----
    soil_prop = get_soil_property_from_SSURGO(SSURGO_gdb_file, sqlite_path, rosetta_model = 3, outfile= None)
    geol_prop = get_GLHYMPSv2_property(GLHYMPS_file, outfile = None)

Author: Pin Shuai (pin.shuai@pnnl.gov)
Date: 11/01/2020
"""

import numpy as np
import logging
import pandas

import workflow.conf


def vgm_Rosetta(data, model_type):
    """
    Return van Genutchen model parameters using Rosetta v3 model ( Zhang and Schaap, 2017 WRR).
    
    Parameters: 
    -----
    data : numpy.ndarray(nvar, nsamples)
      Input data.
    model_type : int
      Rosetta model type: 2--using sand/silt/clay pct, 3--using sand/silt/clay pct + bulk density
        
    Returns:
    ----
    params : pandas dataframe
        van Genutchen model parameters
    """
    import rosetta.DB_Module
    import rosetta.ANN_Module

    db_path = workflow.conf.rcParams['DEFAULT']['rosetta db path'] 
    
    logging.info(f'Running Rosetta for van Genutchen parameters')
    logging.info(f'  database: {db_path}')
    logging.info(f'  model type: {model_type}')

    #convert data from 1d array to nd matrix if necessary
    if data.ndim == 1:
        data = data.reshape(data.shape[0],1)
    
    with rosetta.DB_Module.DB(host='localhost', user='root', db_name='Rosetta',
                              sqlite_path=db_path) as db:

        # choose the right model corresponding to data inputs
        ptf_model = rosetta.ANN_Module.PTF_MODEL(model_type, db)
        res_dict = ptf_model.predict(data, sum_data=True) 

    logging.info(f'  ... done')
        
    # check results
    #   output log10 of VG-alpha,VG-n, and Ks
    vgm_mean = res_dict['sum_res_mean']
    df = pandas.DataFrame(columns=['Rosetta residual volumetric water content [cm^3 cm^-3]',
                                   'Rosetta saturated volumetric water content [cm^3 cm^-3]',
                                   'Rosetta log van Genuchten alpha [cm^-1]',
                                   'Rosetta log van Genuchten n [-]',
                                   'Rosetta log Ksat [um s^-1]'], dtype=float)
    df['Rosetta residual volumetric water content [cm^3 cm^-3]'] = vgm_mean[0]
    df['Rosetta saturated volumetric water content [cm^3 cm^-3]'] = vgm_mean[1]
    df['Rosetta log van Genuchten alpha [cm^-1]'] = vgm_mean[2]
    df['Rosetta log van Genuchten n [-]'] = vgm_mean[3]
    df['Rosetta log Ksat [um s^-1]'] = np.log10( (10**vgm_mean[4]) / 86400 * 1e4 ) # log cm/d --> log um/s
    return df

def vgm_from_SSURGO(df, rosetta_model=3):
    """Get van Genutchen model parameters using Rosetta v3.
    
    Parameters
    ----------
    df : pandas dataframe
      SSURGO properties dataframe, from manager_nrcs.FileManagerNRCS().get_properties()
    rosetta_model: int
                Type of Rosetta model. Default is 3 (i.e., need sand/silt/clay pct and bulk density)
    
    Returns
    -------
    vgm : pandas dataframe
      df with new properties defining the van Genuchten model.  Note that this may
      be smaller than df as entries in df that have NaN values in soil
      composition (and therefore cannot calculate a VGM) will be dropped.

    """
    
    rosetta_input_header = ['total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]', 'bulk density [g/cm^3]']
    df_rosetta = df.dropna(subset=rosetta_input_header)

    # need to transpose the data so that the array have the shape (nvar, nsample) 
    data = df_rosetta[rosetta_input_header].values.T
    vgm = vgm_Rosetta(data, model_type=rosetta_model)
    assert(len(vgm) == len(df_rosetta))
    vgm['mukey'] = df_rosetta['mukey'].values

    # merge back so that we do not lose data
    assert('mukey' in df.keys())
    assert('mukey' in df_rosetta.keys())
    assert('mukey' in vgm.keys())
    merged = pandas.merge(vgm, df, how='outer', left_on='mukey', right_on='mukey')
    assert(len(merged) == len(df))
    return merged

def to_ATS(df):
    """Converts units from aggregated, Rosetta standard-parameters to ATS."""

    df_new = pandas.DataFrame()
    for k in df.keys():
        if k == 'Rosetta log Ksat [um s^-1]':
            knew = 'Rosetta permeability [m^2]'
            vals = 10**df[k] * 1.e-13
            df_new[knew] = vals
        elif k == 'Rosetta log van Genuchten n [-]':
            knew = 'van Genuchten n [-]'
            vals = 10**df[k]
            df_new[knew] = vals
        elif k == 'Rosetta log van Genuchten alpha [cm^-1]':
            knew = 'van Genuchten alpha [Pa^-1]'
            vals = 10**df[k]*100/1000/10
            df_new[knew] = vals
        elif k == 'Rosetta residual volumetric water content [cm^3 cm^-3]':
            knew = 'residual saturation [-]'
            vals = df[k] / df['Rosetta saturated volumetric water content [cm^3 cm^-3]']
            df_new[knew] = vals
        elif k == 'Rosetta saturated volumetric water content [cm^3 cm^-3]':
            knew = 'Rosetta porosity [-]'
            df_new[knew] = df[k]
        elif k == 'log Ksat [um s^-1]':
            knew = 'permeability [m^2]'
            vals = 10**df[k] * 1.e-13
            df_new[knew] = vals
        else:
            df_new[k] = df[k]
    return df_new
        
        
        
