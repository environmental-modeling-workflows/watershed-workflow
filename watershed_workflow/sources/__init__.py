"""This module provides a unified dictionary of all data source managers,
hierarchically keyed by category and product, plus a dictionary of defaults.

``sources`` is a two-level dict::

    sources[category][product_short-source_short] = manager_instance

where the key is derived from each manager's own ``attrs.product_short`` and
``attrs.source_short``.  ``source_short`` is omitted for managers that do not
use the cache directory system.

``_default_sources`` maps a semantic role (e.g. ``'HUC'``, ``'DEM'``) to a
``(category, key)`` tuple.  ``getDefaultSources()`` resolves these to manager
instances.
"""
import logging
from typing import Dict, Any

from .manager_shapefile import ManagerShapefile
from .manager_raster import ManagerRaster

# geometry
from .manager_wbd import ManagerWBD
from .manager_nhd import ManagerNHD
from .manager_3dep import Manager3DEP

# soil structure
from .manager_nrcs import ManagerNRCS
from .manager_glhymps import ManagerGLHYMPS
from .manager_soilgrids_2017 import ManagerSoilGrids2017
from .manager_soilgrids import ManagerSoilGrids
from .manager_polaris import ManagerPOLARIS
from .manager_pelletier_dtb import ManagerPelletierDTB
from .manager_shangguan_dtb import ManagerShangguanDTB
from .manager_hf_hydrodata import ManagerHFHydrodata

# Met data
from .manager_daymet import ManagerDaymet
from .manager_aorc import ManagerAORC

# Land Cover data
from .manager_nlcd import ManagerNLCD
from .manager_modis_appeears import ManagerMODISAppEEARS
from .manager_modis_earthdata import ManagerMODISEarthdata

# Evapotranspiration
from .manager_ssebop import ManagerSSEBop

# Stream gage observations
from .manager_nwis import ManagerNWIS


def _key(manager) -> str:
    """Derive the sources dict key from a manager's attrs."""
    ps = manager.attrs.product_short or manager.attrs.product
    ss = manager.attrs.source_short
    return f'{ps}-{ss}' if ss else ps


def _register(*managers) -> Dict[str, Any]:
    """Build a category sub-dict from a list of manager instances."""
    return {_key(m): m for m in managers}


# ---------------------------------------------------------------------------
# Unified sources dictionary:  sources[category][key]
# ---------------------------------------------------------------------------
sources: Dict[str, Dict[str, Any]] = {
    'geometry': _register(
        ManagerWBD('WBD'),
        ManagerWBD('WaterData'),
        ManagerNHD('NHDPlus MR v2.1'),
        ManagerNHD('NHD MR'),
        ManagerNHD('NHDPlus HR'),
        Manager3DEP(60),
        Manager3DEP(30),
        Manager3DEP(10),
    ),
    'soil_structure': _register(
        ManagerNRCS(),
        ManagerGLHYMPS(),
        ManagerSoilGrids(),
        ManagerSoilGrids2017(),
        ManagerSoilGrids2017('US'),
        ManagerPOLARIS(),
        ManagerPelletierDTB(),
        ManagerShangguanDTB(),
        ManagerHFHydrodata(),
    ),
    'meteorology': _register(
        ManagerAORC(),
        ManagerDaymet(),
    ),
    'land_cover': _register(
        ManagerNLCD(location='L48'),
        ManagerNLCD(location='AK'),
        ManagerMODISAppEEARS(),
        ManagerMODISEarthdata(),
    ),
    'evapotranspiration': _register(
        ManagerSSEBop('daily'),   # key: SSEBop_daily-usgs_ssebop
        ManagerSSEBop('8day'),    # key: SSEBop_8day-usgs_ssebop
        ManagerSSEBop('monthly'), # key: SSEBop_monthly-usgs_ssebop
        ManagerSSEBop('yearly'),  # key: SSEBop_yearly-usgs_ssebop
    ),
    'observations': _register(
        ManagerNWIS(),
    ),
}


# ---------------------------------------------------------------------------
# Defaults: semantic role -> (category, key)
# Keys are product_short-source_short, derived from manager attrs.
# ---------------------------------------------------------------------------
_default_sources: Dict[str, tuple] = {
    'HUC':                ('geometry',           'WBD-pygeohydro_wbd'),
    'hydrography':        ('geometry',           'NHDPlusMR-pynhd_waterdata'),
    'DEM':                ('geometry',           '3DEP_60m-py3dep_tnm'),
    'soil structure':     ('soil_structure',     'SSURGO-nrcs'),
    'geologic structure': ('soil_structure',     'GLHYMPS-dataverse_https'),
    'depth to bedrock':   ('soil_structure',     'Pelletier_DTB-ornl_daac_https'),
    'land cover':         ('land_cover',         'NLCD_L48_2021-pygeohydro_mrlc'),
    'LAI':                ('land_cover',         'modis-nasa_earthdata_opendap'),
    'meteorology':        ('meteorology',        'aorc-ornl_daac_zarr'),
    'evapotranspiration': ('evapotranspiration', 'SSEBop_monthly-usgs_ssebop'),
    'observations':       ('observations',       'NWIS-nwis'),
}


def getDefaultSources() -> Dict[str, Any]:
    """Return a dictionary of default managers keyed by semantic role.

    Returns
    -------
    dict
        Maps semantic role (e.g. ``'HUC'``, ``'DEM'``) to a manager instance.
    """
    return {role: sources[cat][key] for role, (cat, key) in _default_sources.items()}


def getSources(args) -> Dict[str, Any]:
    """Parse argparse args and return an updated set of data sources.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with optional source override attributes.

    Returns
    -------
    dict
        Maps semantic role to a manager instance.
    """
    result = getDefaultSources()

    _arg_map = {
        'source_huc':         ('HUC',                'geometry'),
        'source_hydro':       ('hydrography',         'geometry'),
        'source_dem':         ('DEM',                 'geometry'),
        'soil_structure':     ('soil_structure',      'soil_structure'),
        'geologic_structure': ('geologic_structure',  'soil_structure'),
        'dtb_structure':      ('depth_to_bedrock',    'soil_structure'),
        'land_cover':         ('land_cover',          'land_cover'),
        'meteorology':        ('meteorology',         'meteorology'),
    }

    for arg_name, (role, category) in _arg_map.items():
        try:
            key = getattr(args, arg_name)
        except AttributeError:
            continue
        result[role] = sources[category][key]

    return result


def logSources(sources: Dict[str, Any]) -> None:
    """Pretty-print a sources dictionary to the log."""
    logging.info('Using sources:')
    logging.info('--------------')
    for stype, s in sources.items():
        if s is not None:
            logging.info('{}: {}'.format(stype, s.name))
        else:
            logging.info('{}: None'.format(stype))
