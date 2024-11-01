"""Functions for manipulating soil properties.

Computes soil properties such as permeability, porosity, and van Genutchen
parameters given texture properties using the Rosetta model.

Also provides functions for gap filling soil data via clustering,
dataframe manipulations to merge soil type regions with shared values,
etc.

"""

import numpy as np
import logging
import pandas
import rosetta

import watershed_workflow.config


def computeVanGenuchtenModel_Rosetta(data):
    """Return van Genuchten model parameters using Rosetta v3 model.

    (Zhang and Schaap, 2017 WRR)
    
    Parameters 
    ----------
    data : numpy.ndarray(nvar, nsamples)
      Input data.

    Returns
    -------
    pandas.DataFrame
      van Genuchten model parameters

    """
    logging.info(f'Running Rosetta for van Genutchen parameters')

    #convert data from 1d array to 2d matrix if necessary
    #
    # tranpose for backward compatibility!
    if data.ndim == 1:
        data = [list(data), ]
    else:
        data = [list(entry) for entry in data.transpose()]

    soildata = rosetta.SoilData.from_array(data)
    result_mean, result_std, codes = rosetta.rosetta(3, soildata)
    logging.info(f'  ... done')
    result_mean = np.array(result_mean)

    # check results
    #   output log10 of VG-alpha,VG-n, and Ks
    df = pandas.DataFrame(columns=[
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


def computeVanGenuchtenModelFromSSURGO(df):
    """Get van Genutchen model parameters using Rosetta v3.
    
    Parameters
    ----------
    df : pandas.DataFrame
      SSURGO properties dataframe, from manager_nrcs.FileManagerNRCS().get_properties()
    
    Returns
    -------
    pandas.DataFrame
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
    vgm =  computeVanGenuchtenModel_Rosetta(data)

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
    merged = pandas.merge(vgm, df, how='outer', left_on='mukey', right_on='mukey')
    assert (len(merged) == len(df))
    return merged


def convertRosettaToATS(df):
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


def _whiten(observations):
    """This returns the mean/std deviation for use in unwhiten."""
    means = np.mean(observations, axis=0)
    whitened = observations - np.expand_dims(means, 0)
    std = np.std(whitened, axis=0)
    std[std == 0] = 1
    whitened = whitened / np.expand_dims(std, 0)
    return whitened, (means, std)


def _unwhiten(observations, dat):
    """This does the inverse of _whiten"""
    means, std = dat
    return observations * np.expand_dims(std, 0) + np.expand_dims(means, 0)


def cluster(rasters, nbins):
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


def computeVGAlphaFromPermeability(perm, poro):
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
def getDefaultBedrockProperties():
    """Simple helper function to get a one-row dataframe with bedrock properties.

    Returns
    -------
    pandas.DataFrame
      Sane default bedrock soil properties.

    """
    poro = 0.05
    perm = 1.0e-16

    df = pandas.DataFrame()
    df['ats_id'] = [999, ]
    df['porosity [-]'] = [poro, ]
    df['permeability [m^2]'] = [perm, ]
    df['van Genuchten alpha [Pa^-1]'] = alpha_from_permeability(perm, poro)
    df['van Genuchten n [-]'] = 3.0
    df['residual saturation [-]'] = 0.01
    df['source'] = 'n/a'
    df.set_index('ats_id', drop=True, inplace=True)
    return df


def mangleGLHYMPSProperties(shapes,
                            min_porosity=0.01,
                            max_permeability=np.inf,
                            max_vg_alpha=np.inf):
    """GLHYMPs properties need their units changed and variables renamed.

    Parameters
    ----------
    shapes : list[dict] or list[shapely + properties]
      The raw result from FileManagerGLHYMPS.get_shapes()
    min_porosity : float, optional
      Some GLHYMPS entries have 0 porosity; this sets a floor on that
      value.  Default is 0.01.
    max_permeability : float, optional
      If provided, sets a ceiling on the permeability.
    max_vg_alpha : float, optional
      If provided, sets a ceiling on the vG alpha.

    Returns
    -------
    pandas.DataFrame
      The resulting properties in standard form, names, and units.
    """
    assert (len(shapes) > 0)
    if type(shapes[0]) is dict:
        shp_props = [shp['properties'] for shp in shapes]
    else:
        shp_props = [shp.properties for shp in shapes]

    ids = np.array([prop['OBJECTID_1'] for prop in shp_props], dtype=int)
    for prop in shp_props:
        prop['id'] = prop['OBJECTID_1']

    Ksat = np.array([prop['logK_Ferr_'] for prop in shp_props], dtype=float)
    Ksat = 10**(Ksat / 100)  # units = m^2, division by 100 is per GLHYMPS Readme file
    Ksat = np.minimum(Ksat, max_permeability)
    Ksat_std = np.array([prop['K_stdev_x1'] for prop in shp_props],
                        dtype=float)  # standard deviation
    Ksat_std = Ksat_std / 100  # division by 100 is per GLHYMPS readme
    poro = np.array([prop['Porosity_x'] for prop in shp_props], dtype=float)  # [-]
    poro = poro / 100  # division by 100 is per GLHYMPS readme
    poro = np.maximum(poro, min_porosity)  # some values are 0?

    #descriptions = [prop['Descriptio'] for prop in shp_props]
    # derived properties
    # - this scaling law has trouble for really small porosity, especially high permeability low porosity
    vg_alpha = np.minimum(watershed_workflow.soil_properties.computeVGAlphaFromPermeability(Ksat, poro),
                          max_vg_alpha)
    vg_n = 2.0  # arbitrarily chosen
    sr = 0.01  # arbitrarily chosen

    properties = pandas.DataFrame(
        data={
            'id': ids,
            'source': 'GLHYMPS',
            'permeability [m^2]': Ksat,
            'logk_stdev [-]': Ksat_std,
            'porosity [-]': poro,
            'van Genuchten alpha [Pa^-1]': vg_alpha,
            'van Genuchten n [-]': vg_n,
            'residual saturation [-]': sr,
            #'description' : descriptions,
        })
    return properties


def dropDuplicates(df):
    """Search for duplicate soils which differ only by ID, and rename them, returning a new df.

    Parameters
    ----------
    df : pandas.DataFrame
      A data frame that contains only properties (e.g. permeability,
      porosity, WRM) and is indexed by some native ID.

    Returns
    -------
    df_new : pandas.DataFrame
      After this is called, df_new will:

      1. have a new column, named by df's index name, containing a tuple of all
         of the original indices that had the same properties.
      2. be reduced in number of rows relative to df such that soil
         properties are now unique

    """
    df_new = df.copy()

    grouped = list(df_new.groupby(list(df_new)).apply(lambda x: tuple(x.index)))
    df_new.drop_duplicates(inplace=True)
    df_new.reset_index(inplace=True)

    grouped_reordered = [next(g for g in grouped if i in g) for i in df_new[df.index.name]]
    df_new[df.index.name] = grouped_reordered

    # error checkign!
    common_cols = [col for col in df if col in df_new]
    for ind in df.index:
        new_ind = next(e for e in df_new.index if ind in df_new.loc[e, df.index.name])
        for col in common_cols:
            assert (df.loc[ind, col] == df_new.loc[new_ind, col])
    return df_new
