import numpy as np
import pandas as pd

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
    time_series : pandas.DataFrame
       Time series dataframe with rows as time and columns as land
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

        df[lc] = time_series

    return df


def average_time_series(time_series, repeat_nyears=1):
    """Based on multi-year time_series, compute an average year and repeat it n-times."""
    pass


def write_to_ats(time_series, filename):
    """Writes a time series to the HDF5 file format used by ATS."""
    pass


def to_nlcd(modis_df, crosswalk):
    """Maps MODIS LAI data (IN PLACE) to NLCD indices based on the mapping
    in crosswalk.

    Parameters
    ----------
    lai_df : pandas.DataFrame
       LAI returned from compute_time_series().
    crosswalk : dict
       Mapping from NLCD index to MODIS index.

    Returns
    -------
    pandas.DataFrame
       A new dataframe indexed by NLCD land type.

    """
    nlcd_df = pd.DataFrame()
    if 'time [datetime]' in modis_df:
        nlcd_df['time [datetime]'] = modis_df['time [datetime]']

    for nlcd, modis in crosswalk.items():
        if modis in modis_df:
            nlcd_df[nlcd] = modis_df[modis]

    return nlcd_df
    


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
    dict 
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


    correlation_matrix = np.array((len(unique_nlcd), len(unique_modis)), 'd')

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
    for i,nlcd in enumerate(unique_nlcd):
        crosswalk[nlcd] = unique_modis[np.argmax(correlation_matrix[i])]
    return crosswalk
                                  



def plot_histogram(lc_raster):
    unique = ...
    plt.figure()
    plt.histogram()
    plt.show()
