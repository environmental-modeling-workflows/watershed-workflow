import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy
import rasterio
import datetime

import watershed_workflow.warp
import watershed_workflow.plot

from watershed_workflow.sources.manager_nlcd import colors as nlcd_colors
from watershed_workflow.sources.manager_modis_appeears import colors as modis_colors

def computeTimeSeries(lai, lc, unique_lc=None, lc_idx=-1, polygon=None, polygon_crs=None, **kwargs):
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
    polygon : shapely.Polygon, optional
      If provided, restricts the lai and lc data to the polygon.
    polygon_crs : CRS, optional
      Coordinate system of the polygon.
    kwargs : dict
      Keyword arguments to pass to scipy.signal.savgol_filter such as 
      'window_length (default=101)' and 'polyorder (default=3)'.
    Returns
    -------
    lai timeseries : pandas datafame
       LAI time series dataframe with rows as time and columns as land
       type.
    """
    # compute a mask
    if polygon:
        if polygon_crs:
            polygon = watershed_workflow.warp.shply(polygon, polygon_crs, lai.profile['crs'])
        mask = rasterio.features.geometry_mask([polygon,], lc.data.shape[-2:], lc.profile['transform'], invert=True)
    else:
        mask = np.ones(lc.data.shape[-2:], 'i')

    # find the unique land cover types
    if unique_lc is None:
        unique_lc = list(np.unique(lc.data[lc_idx][mask]))
    try:
        unique_lc.remove(lc.profile['nodata'])
    except ValueError:
        pass

    # average lai for all pixels in the mask and of the lc type
    df = pd.DataFrame()
    df['time [datetime]'] = lai.times

    for ilc in unique_lc:
        time_series = [
            lai.data[itime][mask][np.where(lc.data[lc_idx][mask] == ilc)].mean()
            for itime in range(len(lai.times))
        ]
        col_name = watershed_workflow.sources.manager_modis_appeears.colors[int(ilc)][0]
        df[f'{col_name} LAI [-]'] = time_series
    return df


def computeCrosswalkCorrelation(modis_profile,
                                modis_lc,
                                nlcd_profile,
                                nlcd_lc,
                                method='fractional area',
                                plot=True,
                                warp=True,
                                unique_nlcd=None,
                                unique_modis=None):
    """Compute a map from NLCD indices to MODIS indices using correlation
    of the two rasters.

    Parameters
    ----------
    modis_profile : dict
      Rasterio profile of the modis_lc
    modis_lc : np.ndarray[NX,NY]
      Raster of a single year of MODIS land cover type.
    nlcd_profile : dict
      Rasterio profile of the nlcd_lc
    nlcd_lc : np.ndarray[NX,NY]
      Raster of a single year of NLCD land cover type.
    method : string, optional
      Two available methods:
      - 'fractional area' (default) returns weights
        based on the area fraction of the correlation
      - 'maximal area' returns a weight of 1 for the largest
        fractional area and 0 for all others.
    plot : bool, optional
      Plot the correlation matrix, default is True.
    warp : bool, optional
      Warps MODIS to NLCD.  Should always be true except for tests.
    unique_nlcd : list
      List of unique NLCD indices.  If None, will be computed from raster.
    unique_modis : list
      List of unique MODIS indices.  If None, will be computed from raster.

    Returns
    -------
    dict
       Dictionary whose keys are the unique_nlcd ids and whose values are 
       a list of tuples of (modis_id, weight).  For each value, the
       sum over the list of weights will always sum to 1.

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
        where = np.where(np.bitwise_and(nlcd_lc == nlcd, modis_lc != modis_profile['nodata']))[0]
        count_nlcd = len(where)

        if count_nlcd > 0:
            for j, modis in enumerate(unique_modis):
                where_modis = np.where(np.bitwise_and(nlcd_lc == nlcd, modis_lc == modis))[0]
                count_modis_and_nlcd = len(where_modis)
                correlation_matrix[i, j] = count_modis_and_nlcd / count_nlcd
            assert(abs(correlation_matrix[i,:].sum() - 1) < 1.e-8)
                         
    if plot:
        fig = plt.figure()
        ax = fig.add_axes(rect=[0.2,0.3,0.7,0.6])
        cb = ax.imshow(correlation_matrix, cmap='magma')

        ax.set_xticks(range(len(unique_modis)))
        ax.set_xticklabels([f'{id} : {modis_colors[id][0]}' for id in unique_modis], rotation=90)
        ax.set_yticks(range(len(unique_nlcd)))
        ax.set_yticklabels([f'{id} : {nlcd_colors[id][0]}' for id in unique_nlcd])
        ax.set_xlabel('MODIS labels')
        ax.set_ylabel('NLCD labels')
        plt.colorbar(cb, label='NLCD --> MODIS fractional area')
        plt.show()

    crosswalk = dict()
    for i, nlcd in enumerate(unique_nlcd):
        if method == 'maximal area':
            crosswalk[nlcd] = [(unique_modis[np.argmax(correlation_matrix[i])],1.0),]
        elif method == 'fractional area':
            crosswalk[nlcd] = [(unique_modis[j], correlation_matrix[i,j])
                               for j in range(len(unique_modis)) if correlation_matrix[i,j] > 0]
        else:
            raise ValueError(f'Unknown method: {method}, valid are "fractional area" and "maximal area"')


        rowsum = sum(v[1] for v in crosswalk[nlcd])
        if rowsum > 0:
            assert abs(sum(v[1] for v in crosswalk[nlcd]) - 1.) < 1.e-10
    return crosswalk


def computeMaximalCrosswalkCorrelation(*args, **kwargs):
    """Calls connputeCrosswalkCorrelation, then takes the maximum correlation to just return a map from one to the other."""
    cw = computeCrosswalkCorrelation(*args, **kwargs)
    return dict((k, ((max(v, key=lambda a : a[1])[0]) if len(v) > 0 else None)) for (k, v) in cw.items())


def plotLAI(df, indices='NLCD', ax=None):
    """Plots time series of land cover data."""
    # are we using NLCD or MODIS?
    if indices == 'NLCD':
        info = watershed_workflow.source_list.FileManagerNLCD
    elif indices == 'MODIS':
        info = watershed_workflow.source_list.FileManagerMODISAppEEARS
    else:
        raise ValueError(f'Unknown land cover indices kind "{indices}"')

    # plot the dynamic data
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    lai_date = np.array([datetime.datetime(t.year, t.month, t.day) for t in df['time [datetime]']])
    for column in df:
        if column != 'time [datetime]':
            name = column.strip(' LAI [-]')
            index = info.indices[name]
            color = info.colors[index][1]
            ax.plot(lai_date, df[column], color=color, label=column)
    ax.set_ylabel('Leaf Area Index [-]')
    ax.set_xlabel('time')
    ax.legend()
    plt.show()

        

