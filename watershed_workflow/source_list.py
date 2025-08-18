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
from watershed_workflow.sources.manager_nhd import ManagerNHD
from watershed_workflow.sources.manager_3dep import Manager3DEP
from watershed_workflow.sources.manager_nrcs import ManagerNRCS
from watershed_workflow.sources.manager_glhymps import ManagerGLHYMPS
# from watershed_workflow.sources.manager_soilgrids_2017 import ManagerSoilGrids2017
from watershed_workflow.sources.manager_pelletier_dtb import ManagerPelletierDTB
from watershed_workflow.sources.manager_nlcd import ManagerNLCD

# DayMet THREDDS API is disabled -- this only works for previously-downloaded files!
from watershed_workflow.sources.manager_daymet import ManagerDaymet

from watershed_workflow.sources.manager_aorc import ManagerAORC

from watershed_workflow.sources.manager_modis_appeears import ManagerMODISAppEEARS

from watershed_workflow.sources.manager_shapefile import ManagerShapefile
from watershed_workflow.sources.manager_raster import ManagerRaster

# available and default water boundary datasets
huc_sources = {
    'WBD' : ManagerWBD('WBD'),
    'WaterData WBD' : ManagerWBD('WaterData'),
}
default_huc_source = 'WBD'

# available and default hydrography datasets
hydrography_sources = { 'NHDPlus MR v2.1' : ManagerNHD('NHDPlus MR v2.1'),
                        'NHD MR' : ManagerNHD('NHD MR'),
                        'NHDPlus HR' : ManagerNHD('NHDPlus HR')
                       }
default_hydrography_source = 'NHDPlus MR v2.1'

# available and default digital elevation maps
dem_sources : Dict[str,Any] = {
    '3DEP 60m': Manager3DEP(60),
    '3DEP 30m': Manager3DEP(30),
    '3DEP 10m': Manager3DEP(10),
}
default_dem_source = '3DEP 60m'

# available and default soil survey datasets
structure_sources : Dict[str,Any] = {
    'NRCS SSURGO': ManagerNRCS(),
    'GLHYMPS': ManagerGLHYMPS(),
    # 'SoilGrids2017': ManagerSoilGrids2017(),
    'Pelletier DTB': ManagerPelletierDTB(),
}
default_structure_source = 'NRCS SSURGO'

# available and default land cover
land_cover_sources : Dict[str,Any] = {
    'NLCD (L48)': ManagerNLCD(layer='cover', location='L48'),
    'NLCD (AK)': ManagerNLCD(layer='cover', location='AK'),
    'MODIS': ManagerMODISAppEEARS()
}
default_land_cover = 'NLCD (L48)'

lai_sources : Dict[str,Any] = {
   'MODIS': ManagerMODISAppEEARS()
}
default_lai = 'MODIS'

# available and default meteorology
met_sources : Dict[str,Any] = {
    'AORC': ManagerAORC(),
    'DayMet': ManagerDaymet()
}
default_met = 'AORC'


def getDefaultSources() -> Dict[str, Any]:
    """Provides a default set of data sources.
    
    Returns a dictionary with default sources for each type.
    """
    sources : Dict[str,Any] = dict()
    sources['HUC'] = huc_sources[default_huc_source]
    sources['hydrography'] = hydrography_sources[default_hydrography_source]
    sources['DEM'] = dem_sources[default_dem_source]
    sources['soil structure'] = structure_sources['NRCS SSURGO']
    sources['geologic structure'] = structure_sources['GLHYMPS']
    sources['land cover'] = land_cover_sources[default_land_cover]
    sources['LAI'] = lai_sources[default_lai]
    sources['depth to bedrock'] = structure_sources['Pelletier DTB']
    sources['meteorology'] = met_sources[default_met]
    return sources


def getSources(args) -> Dict[str, Any]:
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
    sources = getDefaultSources()
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
        source_soil = args.soil_structure
    except AttributeError:
        pass
    else:
        sources['soil structure'] = structure_sources[source_soil]

    try:
        source_geo = args.geologic_structure
    except AttributeError:
        pass
    else:
        sources['geologic structure'] = structure_sources[source_geo]

    try:
        source_dtb = args.dtb_structure
    except AttributeError:
        pass
    else:
        sources['depth to bedrock'] = structure_sources[source_dtb]
        
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


def logSources(sources : Dict[str, Any]) -> None:
    """Pretty print source dictionary to log."""
    logging.info('Using sources:')
    logging.info('--------------')
    for stype, s in sources.items():
        if s is not None:
            logging.info('{}: {}'.format(stype, s.name))
        else:
            logging.info('{}: None'.format(stype))
