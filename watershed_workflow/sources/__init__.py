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

from .manager_shapefile import ManagerShapefile
from .manager_raster import ManagerRaster

from .manager_wbd import ManagerWBD
from .manager_nhd import ManagerNHD
from .manager_3dep import Manager3DEP
from .manager_nrcs import ManagerNRCS
from .manager_glhymps import ManagerGLHYMPS
from .manager_soilgrids_2017 import ManagerSoilGrids2017
from .manager_soilgrids import ManagerSoilGrids
from .manager_polaris import ManagerPOLARIS
from .manager_pelletier_dtb import ManagerPelletierDTB
from .manager_shangguan_dtb import ManagerShangguanDTB
from .manager_nlcd import ManagerNLCD

# DayMet THREDDS API is disabled -- this only works for previously-downloaded files!
from .manager_daymet import ManagerDaymet
from .manager_aorc import ManagerAORC

from .manager_modis_appeears import ManagerMODISAppEEARS
from .manager_modis_earthdata import ManagerMODISEarthdata
from .manager_hf_hydrodata import ManagerHFHydrodata
