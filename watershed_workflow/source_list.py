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
from typing import Dict, Any

from watershed_workflow.sources.manager_wbd import ManagerWBD
from watershed_workflow.sources.manager_waterdata import ManagerWaterData
from watershed_workflow.sources.manager_3dep import Manager3DEP
# from watershed_workflow.sources.manager_nrcs import FileManagerNRCS
# from watershed_workflow.sources.manager_glhymps import FileManagerGLHYMPS
# from watershed_workflow.sources.manager_soilgrids_2017 import FileManagerSoilGrids2017
# from watershed_workflow.sources.manager_pelletier_dtb import FileManagerPelletierDTB
# from watershed_workflow.sources.manager_nlcd import FileManagerNLCD
# from watershed_workflow.sources.manager_daymet import FileManagerDaymet
# from watershed_workflow.sources.manager_modis_appeears import FileManagerMODISAppEEARS

from watershed_workflow.sources.manager_shapefile import ManagerShapefile
from watershed_workflow.sources.manager_raster import ManagerRaster

# available and default water boundary datasets
huc_sources = {
    # 'NHDPlus': FileManagerNHDPlusAccumulator(),
    # 'NHD': FileManagerNHD(),
    'WBD' : ManagerWBD(),
    'WaterData WBD' : ManagerWBD('WaterData'),
}
default_huc_source = 'WaterData WBD'

# available and default hydrography datasets
hydrography_sources = { 'NHDv2.1' : ManagerWaterData('nhdflowline_network'), }
#'NHDPlus': FileManagerNHDPlus(), 'NHD': FileManagerNHD(), }
#hydrography_sources['NHD Plus'] = hydrography_sources[
#    'NHDPlus']  # historical typo, kept for backward compatibility
default_hydrography_source = 'NHDv2.1'

# available and default digital elevation maps
dem_sources : Dict[str,Any] = {
    '3DEP 60m': Manager3DEP(60),
    '3DEP 30m': Manager3DEP(30),
    '3DEP 10m': Manager3DEP(10),
}
default_dem_source = '3DEP 60m'

# available and default soil survey datasets
structure_sources : Dict[str,Any] = {
    # 'NRCS SSURGO': FileManagerNRCS(),
    # 'GLHYMPS': FileManagerGLHYMPS(),
    # 'SoilGrids2017': FileManagerSoilGrids2017(),
    # 'Pelletier DTB': FileManagerPelletierDTB(),
}
default_structure_source = None#'NRCS SSURGO'

# available and default land cover
land_cover_sources : Dict[str,Any] = {
    # 'NLCD (L48)': FileManagerNLCD(layer='Land_Cover', location='L48'),
    # 'NLCD (AK)': FileManagerNLCD(layer='Land_Cover', location='AK'),
    # 'MODIS': FileManagerMODISAppEEARS()
}
default_land_cover = None #'NLCD (L48)'

lai_sources : Dict[str,Any] = {}
#    'MODIS': FileManagerMODISAppEEARS() }
default_lai = None #'MODIS'

# available and default meteorology
met_sources : Dict[str,Any] = {} # 'DayMet': FileManagerDaymet() }
default_met = None #'DayMet'


def get_default_sources():
    """Provides a default set of data sources.
    
    Returns a dictionary with default sources for each type.
    """
    sources = dict()
    sources['HUC'] = huc_sources[default_huc_source]
    sources['hydrography'] = hydrography_sources[default_hydrography_source]
    sources['DEM'] = dem_sources[default_dem_source]
    # sources['soil structure'] = structure_sources['NRCS SSURGO']
    # sources['geologic structure'] = structure_sources['GLHYMPS']
    # sources['land cover'] = land_cover_sources[default_land_cover]
    # sources['lai'] = lai_sources[default_lai]
    # sources['depth to bedrock'] = structure_sources['Pelletier DTB']
    # sources['meteorology'] = met_sources[default_met]
    return sources


def get_sources(args):
    """Parsers the command line argument struct from argparse and provides an
    updated set of data sources.

    Parameters
    ----------
    args : struct
      A python struct generated from an argparse.ArgumentParser object with
      source options set by watershed_workflow.ui.*_source_options

    Returns
    -------
    sources : dict
      Dictionary of defaults for each of "HUC", "hydrography", "DEM", "soil
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
