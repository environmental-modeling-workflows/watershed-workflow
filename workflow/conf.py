"""Configuration"""

import sys,os
import numpy as np
import logging
import fiona
import rasterio
import math

try:
    rcParams = { 'data dir' : os.path.join(os.environ['ATS_MESHING_DIR'], 'data'),
                 'packages data dir' : 'packages',
                 'epsg' : 5070, # default Albers equal area conic
                 }
except KeyError:
    rcParams = { 'data dir' : os.path.join(os.getcwd(), 'data'),
                 'packages data dir' : 'packages',
                 'epsg' : 5070, # default Albers equal area conic
                 }
                 

def default_crs():
    """Returns the fiona coordniate system used by default for all output."""
    return fiona.crs.from_epsg(rcParams['epsg'])

def set_default_crs(epsg):
    """Set the default fiona coordinate system from an EPSG integer"""
    crs = fiona.crs.from_epsg(epsg) # make sure valid now
    global rcParams
    rcParams['epsg'] = epsg
        
def latlon_crs():
    """Returns the fiona coordinate system for Lat/Lon products."""
    return fiona.crs.from_epsg(4269)


def huc_str(huc):
    """Converts a huc int or string to a standard-format huc string."""
    if type(huc) is str:
        if len(huc)%2 == 1:
            huc = "0"+huc
    elif type(huc) is int:
        digits = math.ceil(math.log10(huc))
        if digits % 2 == 1:
            digits += 1
        huc = ("%%0%ii"%digits)%huc
    else:
        raise RuntimeError("Cannot convert type %r to huc"%type(huc))
    return huc





