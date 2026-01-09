"""Functions for manipulating soil properties.

Computes soil properties such as permeability, porosity, and van Genutchen
parameters given texture properties using the Rosetta model.

Also provides functions for gap filling soil data via clustering,
dataframe manipulations to merge soil type regions with shared values,
etc.

"""
from typing import Tuple, List, Optional

import numpy as np
import logging
import pandas as pd
import geopandas as gpd
import rosetta

import watershed_workflow.config
import watershed_workflow.sources.standard_names as names


def computeVanGenuchtenModel_Rosetta(data: np.ndarray) -> pd.DataFrame:
    """Return van Genuchten model parameters using Rosetta v3 model.

    (Zhang and Schaap, 2017 WRR)
    
    Parameters 
    ----------
    data : numpy.ndarray(nvar, nsamples)
      Input data.

    Returns
    -------
    pd.DataFrame
      van Genuchten model parameters

    """
    logging.info(f'Running Rosetta for van Genutchen parameters')

    #convert data from 1d array to 2d matrix if necessary
    #
    # tranpose for backward compatibility!
    if data.ndim == 1:
        data_l = [list(data), ]
    else:
        data_l = [list(entry) for entry in data.transpose()]

    soildata = rosetta.SoilData.from_array(data_l)
    result_mean, result_std, codes = rosetta.rosetta(3, soildata)
    logging.info(f'  ... done')
    result_mean = np.array(result_mean)

    # check results
    #   output log10 of VG-alpha,VG-n, and Ks
    df = pd.DataFrame(columns=[
        'Rosetta residual volumetric water content [cm^3 cm^-3]',
        'Rosetta saturated volumetric water content [cm^3 cm^-3]',
        'Rosetta log van Genuchten alpha [cm^-1]', 'Rosetta log van Genuchten n [-]',
        'Rosetta log Ksat [um s^-1]'
    ],
                      dtype=float)
    df['Rosetta residual volumetric water content [cm^3 cm^-3]'] = result_mean[:, 0]
    df['Rosetta saturated volumetric water content [cm^3 cm^-3]'] = result_mean[:, 1]
    df['Rosetta log van Genuchten alpha [cm^-1]'] = result_mean[:, 2]
    df['Rosetta log van Genuchten n [-]'] = result_mean[:, 3]
    df['Rosetta log Ksat [um s^-1]'] = np.log10(
        (10**result_mean[:, 4]) / 86400 * 1e4)  # log cm/d --> log um/s
    return df


def computeVanGenuchtenModelFromSSURGO(df: pd.DataFrame) -> pd.DataFrame:
    """Get van Genutchen model parameters using Rosetta v3.
    
    Parameters
    ----------
    df : pd.DataFrame
      SSURGO properties dataframe, from manager_nrcs.FileManagerNRCS().get_properties()
    
    Returns
    -------
    pd.DataFrame
      df with new properties defining the van Genuchten model.  Note
      that this may be smaller than df as entries in df that have NaN
      values in soil composition (and therefore cannot calculate a
      VGM) will be dropped.

    """
    rosetta_input_header = [
        'total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]', 'bulk density [g/cm^3]',
    ]
    df_rosetta = df.dropna(subset=rosetta_input_header)

    # need to transpose the data so that the array have the shape (nvar, nsample)
    data = df_rosetta[rosetta_input_header].values.T
    vgm = computeVanGenuchtenModel_Rosetta(data)

    n_shapes = len(df_rosetta)
    n_resp = len(vgm["Rosetta residual volumetric water content [cm^3 cm^-3]"])
    logging.info(f'  requested {n_shapes} values')
    logging.info(f'  got {n_resp} responses')
    assert (n_shapes == n_resp)

    vgm['mukey'] = df_rosetta['mukey'].values

    # merge back so that we do not lose data
    assert ('mukey' in df.keys())
    assert ('mukey' in df_rosetta.keys())
    assert ('mukey' in vgm.keys())
    merged = pd.merge(vgm, df, how='outer', left_on='mukey', right_on='mukey')
    assert (len(merged) == len(df))
    return merged


def convertRosettaToATS(df: pd.DataFrame) -> pd.DataFrame:
    """Converts units from aggregated, Rosetta standard-parameters to ATS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Rosetta parameters to convert.

    Returns
    -------
    pd.DataFrame
        DataFrame with parameters converted to ATS units and naming conventions.

    """
    df_new = pd.DataFrame()
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
            vals = 10**df[k] * 100 / 1000 / 10
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


def _whiten(observations: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Returns whitened observations and statistics for use in unwhiten.

    Parameters
    ----------
    observations : np.ndarray
        Input observations to whiten.

    Returns
    -------
    whitened : np.ndarray
        Whitened observations (zero mean, unit variance).
    stats : tuple of np.ndarray
        Tuple containing (means, standard_deviations) for unwhitening.

    """
    means = np.mean(observations, axis=0)
    whitened = observations - np.expand_dims(means, 0)
    std = np.std(whitened, axis=0)
    std[std == 0] = 1
    whitened = whitened / np.expand_dims(std, 0)
    return whitened, (means, std)


def _unwhiten(observations: np.ndarray, dat: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Applies the inverse of whitening transformation.

    Parameters
    ----------
    observations : np.ndarray
        Whitened observations to transform back.
    dat : tuple of np.ndarray
        Statistics from _whiten: (means, standard_deviations).

    Returns
    -------
    np.ndarray
        Un-whitened observations with original scale and offset.

    """
    means, std = dat
    return observations * np.expand_dims(std, 0) + np.expand_dims(means, 0)


def cluster(rasters: np.ndarray,
            nbins: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, np.ndarray]]:
    """Given a bunch of raster bands, cluster into nbins.

    Returns the coloring map of the clusters.  This is used to fill in
    missing soil property data.

    Parameters
    ----------
    rasters : np.ndarray((nx,ny,nbands))
      nbands rasters providing spatial information on which to be clustered.
    nbins : int
      Number of bins to cluster into.

    Returns
    -------
    codebook : np.ndarray((nbins,nbands))
      The nbins centroids of the clusters.
    codes : np.ndarray((nx, ny), int)
      Which cluster each point belongs to.
    distortion : (float, np.ndarray((nx*ny))
      The distortion of the kmeans, and the distance between the
      observation and its nearest code.

    """
    import scipy.cluster.vq
    if len(rasters.shape) == 2:
        rasters = np.expand_dims(rasters, -1)
    assert (len(rasters.shape) == 3)
    in_shp = rasters.shape[0:2]
    total_shp = in_shp[0] * in_shp[1]

    obs = np.reshape(rasters, (-1, rasters.shape[-1]))
    obs_nonan = obs[~np.isnan(obs[:, 0]), :]

    whiten_obs, whiten_dat = _whiten(obs_nonan)
    codebook, dist1 = scipy.cluster.vq.kmeans(whiten_obs, nbins)
    code, dist2 = scipy.cluster.vq.vq(whiten_obs, codebook)
    codebook = _unwhiten(codebook, whiten_dat)

    codes_nan = -1 * np.ones((total_shp, ), 'i')
    codes_nan[~np.isnan(obs[:, 0])] = code
    return codebook, codes_nan.reshape(in_shp), (dist1, dist2)


def computeVGAlphaFromPermeability(perm: np.ndarray, poro: np.ndarray) -> np.ndarray:
    """Compute van Genuchten alpha from permeability and porosity.

    Uses the relationship from Guarracino WRR 2007.

    Parameters
    ----------
    perm : array(double)
      Permeability, in [m^2]
    poro : array(double)
      Porosity, [-]

    Returns
    -------
    alpha : array(double)
      van Genuchten alpha, in [Pa^-1]

    """
    # note all constants are as used in Guarracino paper to not
    # introduce biases in unit changes.
    K_m_per_s = perm * 998. * 9.8 / 1e-3
    K_cm_per_d = K_m_per_s * 100 * 86400.
    alpha_per_cm = np.sqrt(K_cm_per_d / 4.65e4 / poro)
    alpha_per_Pa = alpha_per_cm * 100 / 998. / 9.8
    return alpha_per_Pa


# make a bedrock dataframe
def getDefaultBedrockProperties() -> pd.DataFrame:
    """Simple helper function to get a one-row dataframe with bedrock properties.

    Returns
    -------
    pd.DataFrame
      Sane default bedrock soil properties.

    """
    poro = 0.05
    perm = 1.0e-16

    df = pd.DataFrame()
    df['ats_id'] = [999, ]
    df[names.ID] = [999, ]
    df[names.NAME] = ['bedrock', ]
    df['porosity [-]'] = [poro, ]
    df['permeability [m^2]'] = [perm, ]
    df['van Genuchten alpha [Pa^-1]'] = computeVGAlphaFromPermeability(np.array([perm,]),
                                                                       np.array([poro,]))
    df['van Genuchten n [-]'] = 3.0
    df['residual saturation [-]'] = 0.01
    df['source'] = 'n/a'
    df.set_index('ats_id', drop=True, inplace=True)
    return df


def mangleGLHYMPSProperties(shapes: gpd.GeoDataFrame,
                            min_porosity: float = 0.01,
                            max_permeability: float = np.inf,
                            max_vg_alpha: float = np.inf,
                            residual_saturation: float = 0.01,
                            van_genuchten_n: float = 1.5) -> gpd.GeoDataFrame:
    """GLHYMPs properties need their units changed and variables renamed.

    Parameters
    ----------
    shapes : gpd.GeoDataFrame
    min_porosity : float, optional
      Some GLHYMPS entries have 0 porosity; this sets a floor on that
      value.  Default is 0.01.
    max_permeability : float, optional
      If provided, sets a ceiling on the permeability.
    max_vg_alpha : float, optional
      If provided, sets a ceiling on the vG alpha.

    Returns
    -------
    pd.DataFrame
      The resulting properties in standard form, names, and units.
    """
    assert (len(shapes) > 0)

    ids = shapes['OBJECTID_1']
    shapes[names.ID] = shapes['OBJECTID_1']

    Ksat = shapes['logK_Ferr_'].to_numpy(dtype=float)
    Ksat = 10**(Ksat / 100)  # units = m^2, division by 100 is per GLHYMPS Readme file
    Ksat = np.minimum(Ksat, max_permeability)
    Ksat_std = shapes['K_stdev_x1'].to_numpy(dtype=float)
    Ksat_std = Ksat_std / 100  # division by 100 is per GLHYMPS readme
    poro = shapes['Porosity_x'].to_numpy(dtype=float)
    poro = poro / 100  # division by 100 is per GLHYMPS readme
    poro = np.maximum(poro, min_porosity)  # some values are 0?

    # derived properties
    # - this scaling law has trouble for really small porosity,
    # - especially high permeability low porosity
    vg_alpha = np.minimum(computeVGAlphaFromPermeability(Ksat, poro), max_vg_alpha)

    properties = gpd.GeoDataFrame(data={
        names.ID: ids,
        names.NAME: [f'GLHYMPS-{id}' for id in ids],
        'source': 'GLHYMPS',
        'permeability [m^2]': Ksat,
        'logk_stdev [-]': Ksat_std,
        'porosity [-]': poro,
        'van Genuchten alpha [Pa^-1]': vg_alpha,
        'van Genuchten n [-]': van_genuchten_n,
        'residual saturation [-]': residual_saturation,
        #'description' : descriptions,
    },
                                  geometry=shapes.geometry,
                                  crs=shapes.crs)
    return properties


def dropDuplicates(df: pd.DataFrame,
                   varying_columns: List[str]) -> pd.DataFrame:
    """Removes duplicate rows by merging rows with identical values in all columns except those specified.
    
    Rows are considered duplicates if they have identical values in all columns
    except those listed in varying_columns. For duplicate rows, the varying columns
    are aggregated into tuples containing all values from the merged rows.
    
    For GeoDataFrames, the geometry column is automatically unioned using unary_union
    and cannot be included in varying_columns.
    
    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        The input DataFrame to process.
    varying_columns : list of str
        Column names that are allowed to vary within duplicate groups. Values
        from these columns will be collected into tuples. Cannot include the
        geometry column for GeoDataFrames.
    
    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        A new DataFrame with duplicate rows merged. Returns the same type as
        the input (GeoDataFrame preserves CRS if applicable).
    
    Raises
    ------
    ValueError
        If varying_columns includes the geometry column for a GeoDataFrame.
    """
    # Check if it's a GeoDataFrame
    is_geo = hasattr(df, 'geometry') and df.geometry is not None
    
    if is_geo:
        geom_col = df.geometry.name
        # Geometry column cannot be in varying_columns for GeoDataFrame
        if geom_col in varying_columns:
            raise ValueError(f"Geometry column '{geom_col}' cannot be in varying_columns for GeoDataFrame")
    
    # Columns that must be identical for rows to be merged
    grouping_cols = [col for col in df.columns if col not in varying_columns]
    
    # Remove geometry from grouping cols if it's a GeoDataFrame
    if is_geo and geom_col in grouping_cols:
        grouping_cols.remove(geom_col)
    
    # Group by the identical columns
    grouped = df.groupby(grouping_cols, dropna=False, sort=False)
    
    # Build aggregation dictionary
    agg_dict = {}
    
    # Collect varying columns, creating a new string version
    for col in varying_columns:
        agg_dict[col] = lambda x: '_'.join([str(y) for y in x])
    
    # Union geometries if GeoDataFrame
    if is_geo:
        from shapely.ops import unary_union
        agg_dict[geom_col] = lambda x: unary_union(x.tolist())
    
    # Apply aggregation and reset index
    result = grouped.agg(agg_dict).reset_index()
    
    # Preserve GeoDataFrame type
    if is_geo:
        from geopandas import GeoDataFrame
        result = GeoDataFrame(result, geometry=geom_col, crs=df.crs)
    
    return result

