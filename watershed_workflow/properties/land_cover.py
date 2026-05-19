from typing import Optional, Iterable, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import logging
import rasterio
import datetime
import xarray
import shapely
import importlib

import watershed_workflow.utils.warp
import watershed_workflow.plot.plot
from watershed_workflow.crs import CRS

from watershed_workflow.sources.manager_nlcd import colors as nlcd_colors
from watershed_workflow.sources.manager_modis_appeears import colors as modis_colors

__all__ = [
    'computeTimeSeries',
    'computeCrosswalkCorrelation',
    'computeCrosswalk',
    'applyCrosswalk',
    'removeNullLAI',
    'plotLAI',
    'getDefaultProperties',
    'mapNLCDToPFT',
    'ELM_PFT_NAMES',
]

# ELM natural PFT index → name (indices 0-16 in natpft; pftvarcon.F90 expected_pftnames).
ELM_PFT_NAMES = {
    0:  'not_vegetated',
    1:  'needleleaf_evergreen_temperate_tree',
    2:  'needleleaf_evergreen_boreal_tree',
    3:  'needleleaf_deciduous_boreal_tree',
    4:  'broadleaf_evergreen_tropical_tree',
    5:  'broadleaf_evergreen_temperate_tree',
    6:  'broadleaf_deciduous_tropical_tree',
    7:  'broadleaf_deciduous_temperate_tree',
    8:  'broadleaf_deciduous_boreal_tree',
    9:  'broadleaf_evergreen_shrub',
    10: 'broadleaf_deciduous_temperate_shrub',
    11: 'broadleaf_deciduous_boreal_shrub',
    12: 'c3_arctic_grass',
    13: 'c3_non-arctic_grass',
    14: 'c4_grass',
    15: 'c3_crop',
    16: 'c3_irrigated',
}

# NLCD code → ELM natural PFT index, keyed by climate zone.
#
# 'temperate' covers the majority of US watersheds.  'boreal' shifts
# evergreen/deciduous forests and shrubs to their boreal counterparts
# and is appropriate for Alaska or high-latitude northern sites.
#
# Notes:
#  - NLCD 82 (Cultivated Crops): PFT 15/16 belong in the crop land unit;
#    using c3_non-arctic_grass (13) is the best approximation within the
#    natural-vegetation column required by ATS-ELM (PCT_NATVEG=100).
#  - NLCD 11/12/31/73/74: mapped to not_vegetated (0); ELM has no PFT
#    for open water, ice, lichens, or mosses.
#  - NLCD 21/22 (developed open/low): treated as managed C3 grass because
#    PCT_URBAN=0 in ATS-ELM; impervious fraction is unrepresented.
#  - NLCD 23/24 (developed medium/high): predominantly impervious → bare.
_NLCD_TO_ELM_PFT = {
    'temperate': {
        11: 0,   # Open Water         → not_vegetated
        12: 0,   # Perennial Ice/Snow → not_vegetated
        21: 13,  # Developed, Open Space    → c3_non-arctic_grass (managed lawn)
        22: 13,  # Developed, Low Intensity → c3_non-arctic_grass
        23: 0,   # Developed, Medium Intensity → not_vegetated (impervious)
        24: 0,   # Developed, High Intensity   → not_vegetated (impervious)
        31: 0,   # Barren Land        → not_vegetated
        41: 7,   # Deciduous Forest   → broadleaf_deciduous_temperate_tree
        42: 1,   # Evergreen Forest   → needleleaf_evergreen_temperate_tree
        43: 7,   # Mixed Forest       → broadleaf_deciduous_temperate_tree
        51: 10,  # Dwarf Scrub        → broadleaf_deciduous_temperate_shrub
        52: 10,  # Shrub/Scrub        → broadleaf_deciduous_temperate_shrub
        71: 13,  # Grassland/Herbaceous        → c3_non-arctic_grass
        72: 13,  # Sedge/Herbaceous (wetland)  → c3_non-arctic_grass
        73: 0,   # Lichens            → not_vegetated
        74: 0,   # Moss               → not_vegetated
        81: 13,  # Pasture/Hay        → c3_non-arctic_grass
        82: 13,  # Cultivated Crops   → c3_non-arctic_grass (see note above)
        90: 7,   # Woody Wetlands              → broadleaf_deciduous_temperate_tree
        95: 13,  # Emergent Herbaceous Wetlands → c3_non-arctic_grass
    },
    'boreal': {
        11: 0,
        12: 0,
        21: 13,
        22: 13,
        23: 0,
        24: 0,
        31: 0,
        41: 8,   # Deciduous Forest   → broadleaf_deciduous_boreal_tree
        42: 2,   # Evergreen Forest   → needleleaf_evergreen_boreal_tree
        43: 8,   # Mixed Forest       → broadleaf_deciduous_boreal_tree
        51: 11,  # Dwarf Scrub        → broadleaf_deciduous_boreal_shrub
        52: 11,  # Shrub/Scrub        → broadleaf_deciduous_boreal_shrub
        71: 13,
        72: 13,
        73: 0,
        74: 0,
        81: 13,
        82: 13,
        90: 8,   # Woody Wetlands     → broadleaf_deciduous_boreal_tree
        95: 13,
    },
}


def mapNLCDToPFT(nlcd_index, climate_zone: str = 'temperate') -> np.ndarray:
    """Map NLCD land cover codes to ELM natural PFT indices.

    Returns the dominant ELM PFT index for each NLCD code.  The mapping
    assumes ATS-ELM conventions: PCT_NATVEG=100, PCT_URBAN/CROP/LAKE/
    WETLAND/GLACIER=0, and a single dominant PFT per gridcell.

    Parameters
    ----------
    nlcd_index : int or array-like of int
        NLCD land cover code(s).  Valid values are the standard NLCD codes:
        11, 12, 21-24, 31, 41-43, 51-52, 71-74, 81-82, 90, 95.
    climate_zone : {'temperate', 'boreal'}, optional
        Controls the forest and shrub PFT selection.  'temperate' (default)
        covers most US watersheds.  'boreal' shifts evergreen/deciduous
        tree and shrub types to their boreal counterparts, appropriate
        for Alaska or high-latitude northern sites.

    Returns
    -------
    np.ndarray of int
        ELM PFT index (0-16) for each input code.  Unknown NLCD codes
        raise a KeyError.

    Raises
    ------
    ValueError
        If climate_zone is not 'temperate' or 'boreal'.
    KeyError
        If an NLCD code is not in the mapping table.
    """
    if climate_zone not in _NLCD_TO_ELM_PFT:
        raise ValueError(
            f"climate_zone must be 'temperate' or 'boreal', got {climate_zone!r}")

    table = _NLCD_TO_ELM_PFT[climate_zone]
    arr = np.asarray(nlcd_index, dtype=int)
    scalar_input = arr.ndim == 0
    codes = arr.ravel()
    result = np.empty(len(codes), dtype=int)
    for i, code in enumerate(codes):
        if code not in table:
            raise KeyError(f"NLCD code {code} not in mapNLCDToPFT mapping table")
        result[i] = table[code]
    if scalar_input:
        return int(result[0])
    return result.reshape(arr.shape)


def computeTimeSeries(lai: xarray.DataArray,
                      lc: xarray.DataArray,
                      unique_lc: Optional[Iterable[int]] = None,
                      polygon: Optional[shapely.geometry.Polygon] = None,
                      polygon_crs: Optional[CRS] = None,
                      **kwargs) -> pd.DataFrame:
    """Computes a time-series of LAI for each land cover type that appears
    in the raster.

    Parameters
    ----------
    lai : xarray.DataArray
      The LAI data.
    lc : xarray.DataArray
      The LULC data.
    unique_lc : list
      List of unique land cover types.  If None, will be computed from raster.
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
    lai_crs = watershed_workflow.crs.from_xarray(lai)

    # compute a mask
    if polygon is not None:
        lai = lai.rio.clip([polygon, ], polygon_crs, drop=True, invert=False)
        lc = lc.rio.clip([polygon, ], polygon_crs, drop=True, invert=False)

    # find the class-based means
    space_dims = lc.dims
    stacked_space_dims = 'stacked_'+'_'.join(space_dims)
    
    lai_with_class = lai.assign_coords(class_=lc)
    lai_class_mean = lai_with_class.groupby('class_').mean(dim=stacked_space_dims)

    # average lai for all pixels in the mask and of the lc type
    df = pd.DataFrame({watershed_workflow.sources.manager_modis_appeears.colors[int(ilc)][0] + ' LAI [-]' :
                       lai_class_mean[:,i] for (i,ilc) in enumerate(lai_class_mean['class_'])})
    df['time'] = lai['time']
    return df


def computeCrosswalkCorrelation(modis_lc : xarray.DataArray,
                                nlcd_lc : xarray.DataArray,
                                plot : bool = True,
                                warp : bool = True,
                                unique_nlcd : Optional[List[int]] = None,
                                unique_modis : Optional[List[int]] = None) -> \
                                Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """Compute a map from NLCD indices to MODIS indices using correlation
    of the two rasters.

    Parameters
    ----------
    modis_lc : xarray.DataArray
      Raster of a single year of MODIS land cover type.
    nlcd_lc : xarray.DataArray
      Raster of a single year of NLCD land cover type.
    plot : bool, optional
      Plot the correlation matrix, default is True.
    warp : bool, optional
      Warps MODIS to NLCD CRS before correlating.  Should always be True
      except for tests where the rasters are already aligned.
    unique_nlcd : list of int, optional
      List of unique NLCD indices.  If None, will be computed from raster.
    unique_modis : list of int, optional
      List of unique MODIS indices.  If None, will be computed from raster.

    Returns
    -------
    dict
       Dictionary whose keys are the unique_nlcd ids and whose values are 
       a list of tuples of (modis_id, weight).  For each value, the
       sum over the list of weights will always sum to 1.

    """
    if warp:
        modis_lc = modis_lc.rio.reproject_match(nlcd_lc)

    # flatten for correlations
    assert modis_lc.shape[:-2] == nlcd_lc.shape[:-2]
    modis_flat = modis_lc.values.ravel()
    nlcd_flat = nlcd_lc.values.ravel()

    mask_modis = ~np.isnan(modis_flat) & ~(modis_flat == modis_lc.rio.nodata)
    mask_nlcd = ~np.isnan(nlcd_flat) & ~(nlcd_flat == nlcd_lc.rio.nodata)
    mask = mask_modis & mask_nlcd
    modis_valid = modis_flat[mask]
    nlcd_valid = nlcd_flat[mask]

    if unique_nlcd is None:
        unique_nlcd = list(np.unique(nlcd_valid))

    if unique_modis is None:
        unique_modis = list(np.unique(modis_valid))

    logging.info('Compute the crosswalk between MODIS and NLCD:')
    logging.info(f'  unique MODIS: {unique_modis}')
    logging.info(f'  unique NLCD: {unique_nlcd}')

    corr = np.zeros((len(unique_nlcd), len(unique_modis)), 'd')
    nlcd_counts = np.zeros((len(unique_nlcd), ), 'i')

    for i, nlcd in enumerate(unique_nlcd):
        where_nlcd = nlcd_valid == nlcd
        nlcd_counts[i] = np.sum(where_nlcd)

        if len(where_nlcd) > 0:
            for j, modis in enumerate(unique_modis):
                where_modis = modis_valid == modis

                if len(where_modis) > 0:
                    corr[i, j] = float(np.sum(where_nlcd & where_modis)) / nlcd_counts[i]
                else:
                    corr[i, j] = np.nan
        else:
            corr[i, :] = np.nan

    if plot:
        fig = plt.figure()
        ax = fig.add_axes(rect=(0.2, 0.3, 0.7, 0.6))
        cb = ax.imshow(corr, cmap='magma')

        ax.set_xticks(range(len(unique_modis)))
        ax.set_xticklabels([f'{id} : {modis_colors[id][0]}' for id in unique_modis], rotation=90)
        ax.set_yticks(range(len(unique_nlcd)))
        ax.set_yticklabels([
            f'{id} : {nlcd_colors[id][0]} ({nlcd_counts[i]} px)'
            for (i, id) in enumerate(unique_nlcd)
        ])
        ax.set_xlabel('MODIS labels')
        ax.set_ylabel('NLCD labels')
        plt.colorbar(cb, label='NLCD --> MODIS fractional area')
        plt.show()

    return unique_nlcd, unique_modis, corr, nlcd_counts


def computeCrosswalk(modis_lc: xarray.DataArray,
                     nlcd_lc: xarray.DataArray,
                     method: str = 'maximal area',
                     **kwargs) -> Dict[int, List[Tuple[int, float]]]:
    """Uses a correlation matrix to compute a crosswalk.

    Given MODIS and NLCD LU/LC arrays, computes the correlations of
    the two.

    Returns, for each unique ID in NLCD, a list of pairs of MODIS ID
    and fractional areas.  If method == 'maximal area', only the
    largest weighted correlation is used.  If method == 'fractional
    area', then all correlations are used.

    """
    unique_nlcd, unique_modis, corr, counts = computeCrosswalkCorrelation(
        modis_lc, nlcd_lc, **kwargs)
    crosswalk = dict()

    for i, nlcd in enumerate(unique_nlcd):
        if method == 'maximal area':
            crosswalk[nlcd] = [(unique_modis[np.argmax(corr[i])], 1.0), ]
        elif method == 'fractional area':
            crosswalk[nlcd] = [(unique_modis[j], corr[i, j]) for j in range(len(unique_modis))
                               if corr[i, j] > 0]
        else:
            raise ValueError(
                f'Unknown method: {method}, valid are "fractional area" and "maximal area"')

    return crosswalk


def applyCrosswalk(crosswalk: Dict[int, List[Tuple[int, float]]],
                   modis_lai: pd.DataFrame,
                   unique_nlcd: Optional[List[int]] = None) -> pd.DataFrame:
    """Given a crosswalk from unique NLCD IDs to MODIS, computes time series data for each NLCD ID based on the MODIS LAI data.

    Parameters
    ----------
    crosswalk : Dict[int, List[Tuple[int, float]]]
        Dictionary mapping NLCD IDs to lists of (MODIS_ID, weight) tuples.
    modis_lai : pd.DataFrame
        DataFrame containing MODIS LAI time series data.
    unique_nlcd : List[int], optional
        List of unique NLCD indices. If None, uses all keys from crosswalk.

    Returns
    -------
    pd.DataFrame
        DataFrame containing NLCD LAI time series computed from MODIS data.

    """
    modis_names = watershed_workflow.sources.manager_modis_appeears.colors

    def _modisName(modis_id):
        return f'{modis_names[modis_id][0]} LAI [-]'

    nlcd_names = watershed_workflow.sources.manager_nlcd.colors

    def _nlcdName(nlcd_id):
        return f'{nlcd_names[nlcd_id][0]} LAI [-]'

    nlcd_lai = pd.DataFrame()
    nlcd_lai['time'] = modis_lai['time']

    if unique_nlcd is None:
        unique_nlcd = list(crosswalk.keys())

    for nlcd_id in unique_nlcd:
        cw = crosswalk[nlcd_id]
        nlcd_lai[_nlcdName(nlcd_id)] = sum(part[1] * modis_lai[_modisName(part[0])].to_numpy()
                                           for part in cw)

    return nlcd_lai


def removeNullLAI(nlcd_lai: pd.DataFrame,
                  null_list: Optional[List[int]] = None,
                  names: Optional[Dict[int, Tuple[str, Any]]] = None) -> None:
    """In place, sets entries in the null list to 0 LAI.

    The null_list defaults to a common list for NLCD.
    """
    if null_list is None:
        null_list = [-1, 11, 12, 23, 24, 31]

    if names is None:
        names = watershed_workflow.sources.manager_nlcd.colors

    def _name(id):
        return f"{names[id][0]} LAI [-]"

    for lc_id in null_list:
        name = _name(lc_id)
        print(name, name in nlcd_lai.columns)
        if name in nlcd_lai.columns:
            nlcd_lai[name] = np.zeros((len(nlcd_lai), ), 'd')


def plotLAI(df: pd.DataFrame,
            indices: str = 'NLCD',
            ax: Optional[matplotlib.axes.Axes] = None,
            **kwargs) -> None:
    """Plots time series of land cover data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series LAI data.
    indices : str, optional
        Type of land cover indices ('NLCD' or 'MODIS'). Default is 'NLCD'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.

    """
    # are we using NLCD or MODIS?
    info : type[watershed_workflow.sources.manager_nlcd.ManagerNLCD] | \
        type[watershed_workflow.sources.manager_modis_appeears.ManagerMODISAppEEARS] | \
        None = None
    if indices == 'NLCD':
        info = watershed_workflow.sources.manager_nlcd.ManagerNLCD
    elif indices == 'MODIS':
        info = watershed_workflow.sources.manager_modis_appeears.ManagerMODISAppEEARS
    else:
        raise ValueError(f'Unknown land cover indices kind "{indices}"')

    # plot the dynamic data
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    lai_date = np.array([datetime.datetime(t.year, t.month, t.day) for t in df["time"]])
    for column in df:
        if column != 'time':
            name = column.strip(' LAI [-]')
            index = info.indices[name]
            color = info.colors[index][1]
            ax.plot(lai_date, df[column], color=color, label=column, **kwargs)
    ax.set_ylabel('Leaf Area Index [-]')
    ax.set_xlabel('time')
    ax.legend()
    plt.show()


nlcd_default_properties = pd.read_csv(importlib.resources.files("watershed_workflow") / "data" / "nlcd_land_cover_properties.csv").set_index('indices')
    
def getDefaultProperties(indices, model='NLCD'):
    """Sets default properties for each index, label in nlcd_dict"""
    if model == 'NLCD':
        return nlcd_default_properties[nlcd_default_properties.index.isin(indices)].copy()
    elif model == 'MODIS':
        raise ValueError('MODIS default properties are not yet implemented.')
    else:
        raise ValueError(f'{model} default properties are not implemented.')


mannings_n_by_stream_order = pd.read_csv(importlib.resources.files("watershed_workflow") / "data" / "manning_n_by_stream_order.csv").set_index('order')
