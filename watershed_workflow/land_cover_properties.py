import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy

import watershed_workflow.warp
import watershed_workflow.plot


def compute_time_series(lai, lc, unique_lc=None, lc_idx=-1, **kwargs):
    """Computes a time-series of LAI for each land cover type that appears
    in the raster.

    Parameters
    ----------
    lai : datasets.Data
      The LAI data.
    lc : datasets.Data
      The LULC data.
    unique_lc : list
      List of unique land cover types.  If None, will be computed from raster.
    lc_idx : int
      Index of the land cover type to use for the time series. Default is -1 (lastest year).
    kwargs : dict
      Keyword arguments to pass to scipy.signal.savgol_filter such as 
      'window_length (default=101)' and 'polyorder (default=3)'.
    Returns
    -------
    lai timeseries : pandas datafame
       LAI time series dataframe with rows as time and columns as land
       type.
    """
    if unique_lc is None:
        unique_lc = list(np.unique(lc.data))
    try:
        unique_lc.remove(lc.profile['nodata'])
    except ValueError:
        pass

    df = pd.DataFrame()
    df['time [datetime]'] = lai.times

    for ilc in unique_lc:
        time_series = [
            lai.data[itime, :, :][np.where(lc.data[lc_idx, :, :] == ilc)].mean() for itime in range(len(lai.times))
        ]
        col_name = watershed_workflow.sources.manager_modis_appeears.colors[int(ilc)][0]
        df[f'MODIS {col_name} LAI [-]'] = time_series
    return df


def compute_crosswalk_correlation(modis_profile,
                                  modis_lc,
                                  nlcd_profile,
                                  nlcd_lc,
                                  plot=True,
                                  warp=True,
                                  unique_nlcd=None,
                                  unique_modis=None):
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
    plot : bool
      Image the correlation matrix.
    warp : bool
      Warps MODIS to NLCD.  Should always be true except for tests.
    unique_nlcd : list
      List of unique NLCD indices.  If None, will be computed from raster.
    unique_modis : list
      List of unique MODIS indices.  If None, will be computed from raster.

    Returns
    -------
    crosswalk
       Maps keys of MODIS indices into NLCD indices.

    """
    assert (len(modis_lc.shape) == 2)
    assert (len(nlcd_lc.shape) == 2)
    if warp:
        modis_profile, modis_lc = watershed_workflow.warp.raster(modis_profile,
                                                                 modis_lc,
                                                                 dst_crs=nlcd_profile['crs'],
                                                                 dst_height=nlcd_profile['height'],
                                                                 dst_width=nlcd_profile['width'])
    if unique_nlcd is None:
        unique_nlcd = list(np.unique(nlcd_lc))

    if 'nodata' in nlcd_profile:
        try:
            unique_nlcd.remove(nlcd_profile['nodata'])
        except ValueError:
            pass

    if unique_modis is None:
        unique_modis = list(np.unique(modis_lc))

    if 'nodata' in modis_profile:
        try:
            unique_modis.remove(modis_profile['nodata'])
        except ValueError:
            pass

    correlation_matrix = np.zeros((len(unique_nlcd), len(unique_modis)), 'd')
    for i, nlcd in enumerate(unique_nlcd):
        where = np.where(nlcd_lc == nlcd)
        count_nlcd = len(where[0])

        for j, modis in enumerate(unique_modis):
            where_modis = np.where(np.bitwise_and(nlcd_lc == nlcd, modis_lc == modis))
            count_modis_and_nlcd = len(where_modis[0])
            correlation_matrix[i, j] = count_modis_and_nlcd / count_nlcd

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cb = ax.imshow(correlation_matrix)

        ax.set_xticks(range(len(unique_modis)))
        ax.set_xticklabels(unique_modis)
        ax.set_yticks(range(len(unique_nlcd)))
        ax.set_yticklabels(unique_nlcd)
        ax.set_xlabel('MODIS labels')
        ax.set_ylabel('NLCD labels')
        plt.colorbar(cb, label='correlation')
        plt.show()

    crosswalk = dict()
    for i, nlcd in enumerate(unique_nlcd):
        crosswalk[nlcd] = unique_modis[np.argmax(correlation_matrix[i])]
    return crosswalk
