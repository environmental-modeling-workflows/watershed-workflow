"""This module provides a dictionary of sources, broken out by data type, and a
dictionary of default sources.

These dictionaries are provided as module-local (singleton) variables.

* huc_sources : A dictionary of sources that provide USGS HUC boundaries.
* hydrography_sources : A dictionary of sources that provide river reaches by HUC.
* dem_sources : A dictionary of available digital elevation models.
* soil_sources : A dictionary of available sources for soil properties.
* land_cover_sources : A dictionary of available land cover datasets.

"""
import logging

from workflow.sources.manager_nhd import FileManagerNHD, FileManagerNHDPlus, FileManagerWBD
from workflow.sources.manager_nhd_accumulator import FileManagerNHDPlusAccumulator
from workflow.sources.manager_ned import FileManagerNED
from workflow.sources.manager_nrcs import FileManagerNRCS
from workflow.sources.manager_glhymps import FileManagerGLHYMPS
from workflow.sources.manager_nlcd import FileManagerNLCD
from workflow.sources.manager_daymet import FileManagerDaymet

from workflow.sources.manager_shape import FileManagerShape
from workflow.sources.manager_raster import FileManagerRaster


# available and default water boundary datasets
huc_sources = {'NHD Plus' : FileManagerNHDPlusAccumulator(),
               'NHD' : FileManagerNHD(),
               'WBD' : FileManagerWBD()
               }
default_huc_source = 'WBD'

# available and default hydrography datasets
hydrography_sources = {'NHD Plus' : FileManagerNHDPlus(),
                       'NHD' : FileManagerNHD(),
                       }
default_hydrography_source = 'NHD'

# available and default digital elevation maps
dem_sources = {'NED 1/3 arc-second' : FileManagerNED('1/3 arc-second'),
               'NED 1 arc-second' : FileManagerNED('1 arc-second'),
               }
default_dem_source = 'NED 1 arc-second'

# available and default soil survey datasets
structure_sources = {'NRCS SSURGO' : FileManagerNRCS(),
                'GLHYMPS' : FileManagerGLHYMPS(),
                }
default_structure_source = 'NRCS SSURGO'

# available and default land cover
land_cover_sources = {'NLCD (L48)' : FileManagerNLCD(layer='Land_Cover', location='L48'),
                      'NLCD (AK)' : FileManagerNLCD(layer='Land_Cover', location='AK')
                      }
default_land_cover = 'NLCD (L48)'


# available and default meteorology
met_sources = {'DayMet' : FileManagerDaymet()}
default_met = 'DayMet'

def get_default_sources():
    """Provides a default set of data sources.
    
    Returns a dictionary with default sources for each type.
    """
    sources = dict()
    sources['HUC'] = huc_sources[default_huc_source]
    sources['hydrography'] = hydrography_sources[default_hydrography_source]
    sources['DEM'] = dem_sources[default_dem_source]
    sources['soil structure'] = structure_sources['NRCS SSURGO']
    sources['geologic structure'] = structure_sources['GLHYMPS']
    sources['land cover'] = land_cover_sources[default_land_cover]
    sources['soil thickness'] = None
    sources['meteorology'] = met_sources[default_met]
    return sources


def get_sources(args):
    """Parsers the command line argument struct from argparse and provides an
    updated set of data sources.

    Parameter
    ---------
    args : struct
      A python struct generated from an argparse.ArgumentParser object with
      source options set by workflow.ui.*_source_options

    Returns
    -------
    sources : dict
      Diectionary of defaults for each of "HUC", "hydrography", "DEM", "soil
      type", and "land cover".
    """
    sources = get_default_sources()
    try:
        source_huc = args.source_huc
    except AttributeError:
        pass
    else:
        sources['HUC'] = huc_sources[source_huc]

    try:
        source_hydrography = args.source_hydro
    except AttributeError:
        pass
    else:
        sources['hydrography'] = hydrography_sources[source_hydrography]

    try:
        source_dem = args.source_dem
    except AttributeError:
        pass
    else:
        sources['DEM'] = dem_sources[source_dem]

    try:
        source_soil = args.source_soil
    except AttributeError:
        pass
    else:
        sources['soil type'] = soil_sources[source_soil]


    try:
        land_cover = args.land_cover
    except AttributeError:
        pass
    else:
        sources['land cover'] = land_cover_sources[land_cover]

    try:
        met = args.meteorology
    except AttributeError:
        pass
    else:
        sources['meteorology'] = met_sources[met]
        
    return sources

def log_sources(sources):
    """Pretty print source dictionary to log."""
    logging.info('Using sources:')
    logging.info('--------------')
    for stype, s in sources.items():
        if s is not None:
            logging.info('{}: {}'.format(stype, s.name))
        else:
            logging.info('{}: None'.format(stype))

            
