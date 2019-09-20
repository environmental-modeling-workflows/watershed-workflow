"""Configuration and global defaults."""

import sys,os
import subprocess
import fiona

rcParams = {'packages data dir' : 'packages',
            'epsg' : 5070, # default Albers equal area conic
            'digits' : 7, # roundoff precision
            }
try:
    rcParams['data dir'] = os.path.join(os.environ['ATS_MESHING_DIR'], 'data')
except KeyError:
    rcParams['data dir'] = os.path.join(os.getcwd(), 'data')

_projections = dict()
def get_projection(crs):
    """Returns the cartopy transform for the given crs"""
    try:
        return _projections[crs['init']]
    except KeyError:
        try:
            import cartopy.crs
            _projections[crs['init']] = cartopy.crs.epsg(crs['init'][5:])
        except ValueError as err:
            if crs['init'] == latlon_crs()['init']:
                _projections[crs['init']] = cartopy.crs.PlateCarree()
            else:
                raise err
        return _projections[crs['init']]
    
def get_crs(epsg):
    """Returns the fiona coordinate system from epsg number"""
    if epsg is None:
        return None
    return fiona.crs.from_epsg(epsg)
    
    
def default_crs():
    """Returns the fiona coordinate system used by default for all output."""
    return get_crs(rcParams['epsg'])

def default_crs_alaska():
    """Returns a sane default fiona coordinate system for Alaska work."""
    return get_crs(3338)

def set_default_crs(epsg):
    """Set the default fiona coordinate system from an EPSG integer"""
    crs = get_crs(epsg) # make sure valid now
    global rcParams
    rcParams['epsg'] = epsg
        
def latlon_crs():
    """Returns the fiona coordinate system for Lat/Lon products."""
    return get_crs(4269)


def get_git_revision_hash():
    """Returns the git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')





