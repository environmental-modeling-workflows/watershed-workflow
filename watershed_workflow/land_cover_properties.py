import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import scipy

import rasterio
import rasterio.transform
import rasterio.features

import shapely

import watershed_workflow.config
import watershed_workflow.triangulation
import watershed_workflow.warp
import watershed_workflow.plot
import watershed_workflow.river_tree
import watershed_workflow.split_hucs
import watershed_workflow.hydrography
import watershed_workflow.sources.utils
import watershed_workflow.sources.manager_shape
import watershed_workflow.utils


lc_type1_colors = {
        -1:  ('Unclassified', (0.00000000000,  0.00000000000,  0.00000000000)),
        0: ('Open Water', (0.27843137255,  0.41960784314,  0.62745098039)),
        1: ('Evergreen Needleleaf Forests', (0.10980392157,  0.38823529412,  0.18823529412)),
        2: ('Evergreen Broadleaf Forests', (0.10980392157,  0.38823529412,  0.18823529412)),
        3: ('Deciduous Needleleaf Forests', (0.40784313726,  0.66666666667,  0.38823529412)),
        4: ('Deciduous Broadleaf Forests', (0.40784313726,  0.66666666667,  0.38823529412)),
        5: ('Mixed Forests', (0.70980392157,  0.78823529412,  0.55686274510)),
        6: ('Closed Shrublands', (0.80000000000,  0.72941176471,  0.48627450980)),
        7: ('Open Shrublands', (0.80000000000,  0.72941176471,  0.48627450980)),
        8: ('Woody Savannas', (0.40784313726,  0.66666666667,  0.38823529412)),
        9: ('Savannas', (0.70980392157,  0.78823529412,  0.55686274510)),
        10: ('Grasslands', (0.88627450980,  0.88627450980,  0.75686274510)),
        11: ('Permanent Wetlands', (0.43921568628,  0.63921568628,  0.72941176471)),
        12: ('Croplands', (0.66666666667,  0.43921568628,  0.15686274510)),
        13: ('Urban and Built up lands', (0.86666666667,  0.78823529412,  0.78823529412)),
        14: ('Cropland Natural Vegetation Mosaics', (0.66666666667,  0.43921568628,  0.15686274510)),
        15: ('Permanent Snow and Ice', (0.81960784314,  0.86666666667,  0.97647058824)),
        16: ('Barren Land', (0.69803921569,  0.67843137255,  0.63921568628)),
        17: ('Water Bodies', (0.27843137255,  0.41960784314,  0.62745098039)),
    }


class FileManagerMODIS:
    """Land cover as well as LAI data rasters are obtained from MODIS [NLCD]_.

    .. note:: MODIS data are downloaded from the APPEEARS server.

    """

##lc_colors = lc_type1_colors
#for i in ids:
#    lc_colors = lc_type1_colors[i][1]

def compute_time_series(lai_raster, lc_raster, times=None):
    """Computes a time-series of LAI for each land cover type that appears
    in the raster.

    Parameters
    ----------
    lai_raster : numpy.ma.MaskedArray[NX,NY,NTIMES]
       Array of LAI.
    lc_raster : numpy.ma.MaskedArray[NX,NY]
       Array of land cover.
    times : list(datetime)[NTIMES]
       Array of times, in [datetime], of the lai_raster.
       
    Returns
    -------
    lai timeseries : pandas datafame
       LAI time series dataframe with rows as time and columns as land
       type.
    """
    
    unique_lc = [val for val in np.unique(lc_raster.ravel()) if val != lc_raster.fill_value]

    df = pd.DataFrame()
    if times is not None:
        df['time [datetime]'] = times

    for lc in unique_lc:
        time_series = []
        for i in range(lai_raster.shape[2]):
            lai_slice = lai_raster[:,:,i]
            time_series.append(lai_slice[np.where(lc_raster == lc)].mean())

        df = time_series

    return df

def average_time_series(df, smooth_filter=True, nyears=None):
    """Based on multi-year time_series, compute an average year and repeat it n-times.
    
    Parameters
    ----------
    df : pandas data frame
       Timeseries of LAI of each LULC types.
    smooth_filter, bool
        whether to use savgol_filter for smoothing
    nyears, int
        repreat the typical year for how many years

    Returns
    -------
    smoothed LAI time-series : pandas dataframe
       Time series dataframe with rows as time and columns as land
       type.
    """
    
    logging.info("averaging MODIS LAI by taking the average for each day across the actual years.")
    
    if nyears == None:
        nyears = raw[var_list[0]].shape[0]*8//365 ##this is an 8-day averaged product
    
    df = pd.DataFrame()
    df['time [s]'] = nyears*365*86400
    df['time [d]'] = df['time [s]']//86400 + 8214

    # interpolate this time series into a daily time series
    ts = np.arange(8214, 14969, 1)

    #df_interp = pandas.DataFrame()
    df['time [d]'] = ts
    
    df_interp = pd.DataFrame()
    for k in df.keys():
        if k != 'time [s]':
            f = scipy.interpolate.interp1d(df['time [d]'][:], df[k][:])
            df_interp[k] = f(ts)

    df = df_interp

    # smooth the data
    df_smooth = pd.DataFrame()
    df_smooth['time [d]'] = df['time [d]']
    for k in df.keys():
        if k != 'time [d]':
            df_smooth[k] = scipy.signal.savgol_filter(df[k], 101, 3)
    
    ##Convert this to typical years and repeat that
    df_yr = []
    for year in range(23, 39):
        yr = df.loc[df_interp['time [d]'] >= year*365]
        df_yr.append(yr.loc[yr['time [d]'] < (year+1)*365])

    # average across the years
    df_avg = pd.DataFrame()
    for yr in df_yr:
        for k in yr.keys():
            if not k.startswith('time'):
                if k in df_avg:
                    df_avg[k] = df_avg[k].array + yr[k].array
                else:
                    df_avg[k] = yr[k].copy()

    for k in df_avg.keys():
        df_avg[k] = df_avg[k][:] / len(df_yr)

    df_avg['time [d]'] = df['time [d]']
    
    # replicate 40 times to make 40 years (remem)
    # tile all data to repeat n_year times
    df_rep = pd.DataFrame()
    for key in df_avg:
        if not key.startswith('time'):
            df_rep[key] = np.tile(df_avg[key].array, 40)
            assert(len(df_rep) == 40*365)
    
    # time is simply daily data
    df_rep['time [s]'] = 86400. * np.arange(0., 40 * 365., 1.)
    df_rep['time [d]'] = np.arange(0., 40 * 365., 1.)
    
    return df_rep

#    outputs['modis_filename'] = f'../data-processed/{name}_MODIS_LAI_smoothed_2002_2020.h5'
#
#    df_smooth['time [s]'] = df_smooth['time [d]']*86400
#    with h5py.File(outputs['modis_filename'],'w') as fid:
#        for k in df_smooth:
#            fid.create_dataset(k, data=df_smooth[k][:])
#

def compute_crosswalk_correlation(modis_lc, modis_profile,
                                  nlcd_lc, nlcd_profile,
                                  plot=True, warp=True):
    """Compute a map from NLCD indices to MODIS indices using correlation
    of the two rasters.

    Parameters
    ----------
    modis_lc : np.ndarray[NX,NY]
      Raster of a single year of MODIS land cover type.
    modis_profile : dict
      Rasterio profile of the modis_lc
    nlcd_lc : np.ndarray[NX,NY]
      Raster of a single year of NLCD land cover type.
    nlcd_profile : dict
      Rasterio profile of the nlcd_lc
    warp : bool, optional=True
      Warp MODIS into NLCD raster profile.

    Returns
    -------
    crosswalk
       Maps keys of MODIS indices into NLCD indices.

    """
    if warp:
        assert(watershed_workflow.crs.equal(modis_profile['crs'], nlcd_profile['crs']))
        modis_profile, modis_lc = watershed_workflow.warp.raster(modis_profile, modis_lc,
                                                             nlcd_profile['crs'],
                                                             nlcd_profile['resolution'],
                                                             nlcd_profile['height'],
                                                             nlcd_profile['width'])

    unique_nlcd = [val for val in np.unique(nlcd_lc.ravel()) if val != nlcd_lc.fill_value]
    unique_modis = [val for val in np.unique(modis_lc.ravel()) if val != modis_lc.fill_value]


    #correlation_matrix = np.array((len(unique_nlcd), len(unique_modis)), 'd')
    correlation_matrix = np.zeros((len(unique_nlcd), len(unique_modis)), dtype=float, order='C')
    

    for i, nlcd in enumerate(unique_nlcd):
        where = np.where(nlcd_lc == nlcd)
        count_nlcd = len(where[0])

        for j, modis in enumerate(unique_modis):
            where_modis = np.where(np.bitwise_and(nlcd_lc == nlcd, modis_lc == modis))
            count_modis_and_nlcd = len(where_modis[0])
    
            correlation_matrix[i,j] =  count_modis_and_nlcd / count_nlcd

    if plot:
        plt.imshow(correlation_matrix)
        plt.show()

    crosswalk = dict()
    for i, nlcd in enumerate(unique_nlcd):
        crosswalk[nlcd] = unique_modis[np.argmax(correlation_matrix[i])]
    return crosswalk
                                  
def write_to_ats(lai_raster, crosswalk, smooth=False, smooth_filter=False, nyears=None):
    """Accepts a numpy array of LAI data and returns a dictionary ATS data.
    
    Parameters:
        lai_raster : dict
            MODIS LAI dictinary
        smooth : bool
            whether to smooth the data or not
        smooth_filter : bool
            whether to apply a savgol filter for smoothing. Default is False.
        nyears : int
            how many years to repeat the typical year. Default is to get the same length as the raw data.
        crosswalk : Maps keys of MODIS indices into NLCD indices.
    """

    outputs['modis_filename'] = f'../data-processed/{name}_MODIS_LAI_smoothed_2002_2020.h5'

    df_smooth['time [s]'] = df_smooth['time [d]']*86400
    with h5py.File(outputs['modis_filename'],'w') as fid:
        for k in df_smooth:
            fid.create_dataset(k, data=df_smooth[k][:])
    
    # for the entire 40 years data
    outputs['modis_typical_filename'] = f'../data-processed/{name}_MODIS_LAI_typical_1980_2020.h5'

    with h5py.File(outputs['modis_typical_filename'],'w') as fid:
        for k in df_rep:
            fid.create_dataset(k, data=df_rep[k][:])

#
#    nlcd_LAI['time [s]'] = (nlcd_LAI.index - nlcd_LAI.index[0]).total_seconds()
#
#    with h5.File(f'../data-raw/{name}_MODIS_LAI_072002_122020.h5', 'w') as fout:
#
#        for i in ['time [s]', f'NLCD {NLCDLULC1label1} LAI [-]']:
#            fout.create_dataset(i, data= nlcd_LAI[i].values)
            

def plot_histogram(modis_lc):
    """Plot histogram based on the MODIS LC pixels.

    Parameters
    ----------
#    nlcd_lc : np.ndarray[NX,NY]
#      Raster of a single year of NLCD land cover type.
    modis_lc : np.ndarray[NX,NY]
        Raster of a single year of MODIS land cover type.
    """

#    unique = ...
#    plt.figure()
#    plt.histogram()
#    plt.show()
    #colors = [lc_colors[i][1] for i in ids]
    #lc_colors = [lc_type1_colors[i][1] for i in ids]

    #labels = [lc_type1_colors[i][0] for i in ids]

    #labelsp1 = [lc_type1_colors[i][0] for i in lc_type1_colors]

    counts, bins = np.histogram(modis_lc, range=[-1,17], bins=18)
    plt.hist(bins[:-1], bins=18, range=[-1,17], weights=counts)
    plt.xlabel("MODIS LULC")
    plt.ylabel("Counts")
    plt.title("Histogram of number of pixels per LULC types")
    plt.xticks(bins,labelsp1,rotation=90)
    
def area_cutoff(modis_lc):
    
    ids, counts = np.unique(modis_lc.values[~np.isnan(modis_lc.values)], return_counts=True)

    sum1 = sum(counts)
    cutoff = sum1*0.05 #Separate LULC types with 5% of the pixel coverage cutoff
    


################ Implementation in notebook ###############
#1. modis and nlcd crosswalk comparison to enable users making their own judgement
#2. logics related to LC type selection (1, 2, 3, 4 etc.)
#3. plots of MODIS LC comparison with NLCD, MODIS LAI time-series
#4. filter LAI data to remove the erroneously high and low values (-ve)
