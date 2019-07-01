"""Configuration and global defaults."""

import sys,os
import fiona

rcParams = {'packages data dir' : 'packages',
            'epsg' : 5070, # default Albers equal area conic
            'digits' : 7, # roundoff precision
            'national_map_api_url' : 'https://viewer.nationalmap.gov/tnmaccess/api/products',
            }
try:
    rcParams['data dir'] = os.path.join(os.environ['ATS_MESHING_DIR'], 'data')
except KeyError:
    rcParams['data dir'] = os.path.join(os.getcwd(), 'data')
                 

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





