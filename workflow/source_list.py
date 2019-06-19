from workflow.sources.manager_nhdplus import FileManagerNHDPlus
from workflow.sources.manager_ned import FileManagerNED
from workflow.sources.manager_shape import FileManagerShape


huc_sources = {'NHD Plus H': FileManagerNHDPlus(),
               }
default_huc_source = 'NHD Plus H'

hydrography_sources = {'NHD Plus H': huc_sources['NHD Plus H'],
                       }
default_hydrography_source = 'NHD Plus H'

dem_sources = {'NED 1/3 arc-second': FileManagerNED('1/3 arc-second'),
               'NED 1 arc-second': FileManagerNED('1 arc-second'),
               }
default_dem_source = 'NED 1/3 arc-second'


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
