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


huc_sources = {'NHD Plus': FileManagerNHDPlus(),
               'NHD': FileManagerNHD(),
               'WBD': FileManagerWBD(),
               }

default_huc_source = 'WBD'

hydrography_sources = {'NHD Plus': huc_sources['NHD Plus'],
                       'NHD': huc_sources['NHD'],
                       }
default_hydrography_source = 'NHD'

dem_sources = {'NED 1/3 arc-second': FileManagerNED('1/3 arc-second'),
               'NED 1 arc-second': FileManagerNED('1 arc-second'),
               }
default_dem_source = 'NED 1 arc-second'


def get_default_sources():
    """Provides a default set of data sources."""
    sources = dict()
    sources['HUC'] = huc_sources[default_huc_source]
    sources['hydrography'] = hydrography_sources[default_hydrography_source]
    sources['DEM'] = dem_sources[default_dem_source]
    sources['land cover'] = None
    sources['soil thickness'] = None
    sources['soil type'] = None
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

            
