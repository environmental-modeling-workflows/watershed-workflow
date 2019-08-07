"""Default Sources:

Default HUCs and Hydrography data comes from the NHDPlus High Res
datasets.
See: "https://nhd.usgs.gov/"

Default DEMs come from the National Elevation Dataset (NED).
See: "https://lta.cr.usgs.gov/NED"
"""
import logging

from workflow.sources.manager_nhd import FileManagerNHD, FileManagerNHDPlus, FileManagerWBD
from workflow.sources.manager_ned import FileManagerNED
from workflow.sources.manager_shape import FileManagerShape
from workflow.sources.manager_nrcs import FileManagerNRCS
from workflow.sources.manager_nlcd import FileManagerNLCD


# available and default water boundary datasets
huc_sources = {'NHD Plus': FileManagerNHDPlus(),
               'NHD': FileManagerNHD(),
               'WBD': FileManagerWBD()
               }
default_huc_source = 'WBD'

# available and default hydrography datasets
hydrography_sources = {'NHD Plus': huc_sources['NHD Plus'],
                       'NHD': huc_sources['NHD'],
                       }
default_hydrography_source = 'NHD'

# available and default digital elevation maps
dem_sources = {'NED 1/3 arc-second': FileManagerNED('1/3 arc-second'),
               'NED 1 arc-second': FileManagerNED('1 arc-second'),
               }
default_dem_source = 'NED 1 arc-second'

# available and default soil survey datasets
soil_sources = {'NRCS SSURGO':FileManagerNRCS(),
                }
default_soil_source = 'NRCS SSURGO'

# available and default land cover
land_cover_sources = {'NLCD (L48)' : FileManagerNLCD(layer='Land_Cover', location='L48'),
                      'NLCD (AK)' : FileManagerNLCD(layer='Land_Cover', location='AK')
                      }
default_land_cover = 'NLCD (L48)'


def get_default_sources():
    """Provides a default set of data sources."""
    sources = dict()
    sources['HUC'] = huc_sources[default_huc_source]
    sources['hydrography'] = hydrography_sources[default_hydrography_source]
    sources['DEM'] = dem_sources[default_dem_source]
    sources['soil type'] = soil_sources[default_soil_source]
    sources['land cover'] = land_cover_sources[default_land_cover]
    sources['soil thickness'] = None
    return sources


def get_sources(args):
    """Parsers the arg dict and provides an updated set of data sources."""
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
        sources['land cover'] = soil_sources[land_cover]
        
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

            
