"""Coordinate Reference System conversions.

Coordinate Reference Systems (CRSs) differ across datasets, and
standardizing and managing these across the workflow is a necessary
technical detail.  That said, rarely does the user care what
coordinate system is being used, as long as it is appropriate for the
watershed in question.  Watershed Workflow aims to make using datasets
in different CRSs as streamlined as possible.  Typically, a workflow
will pick a CRS based upon either a default for the region or by
simply using the CRS of the shapefile that specifies the watershed
boundary.  This CRS is the passed into each function that acquires
more data, and that data's coordinates are changed to the CRS
requested.

This process is made more difficult by the fact that most python GIS
packages provide their own class object to store the CRS.  This said,
nearly all of them are based, to some degree, on the `proj4` library
and its python wrapper, `pyproj` for coordinate transformations.
Watershed Workflow uses the `pyproj.Proj` class as its own internal
representation of coordinate system, and provides methods for mapping
`fiona` (shapefiles), `rasterio` (rasters), and `cartopy` (plotting)
CRS objects to and from this type.  While this is typically done by
calling functions from those libraries, standardizing the API makes
dealing with these packages in an integrated form much simpler.


.. note::
    We intend to use the pyproj.Proj object as our standard.  But for
    now we are trying to avoid hard-coding that, so internal code
    should avoid using that knowledge, and instead map to and from
    `pyproj.Proj` objects using the provided interface.

"""
import logging
import pyproj.crs
from pyproj.crs import CRS

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def is_native(crs):
    """Is this crs in the native format?"""
    return type(crs) == type(to_proj(crs))

def from_proj(crs):
    """Converts a Proj CRS to the workflow CRS standard.

    Parameters
    ----------
    crs : pyproj.crs.CRS
        Input proj CRS object.
    
    Returns
    -------
    out : crs-type
        Equivalent workflow CRS.
    """
    # try:
    #     # if a proj.6 CRS object or a proj.4 Proj object
    #     wkt_str = crs.to_wkt()
    # except AttributeError:
    #     # if a proj.6 Proj object
    #     wkt_str = crs.crs.to_wkt()
    # return CRS.from_wkt(wkt_str)
    return crs


def to_proj(crs):
    """Converts a workflow CRS standard to a Proj4 CRS.

    Parameters
    ----------
    crs : crs-type
        Workflow standard CRS.
    
    Returns
    -------
    out : pyproj.crs.CRS
        Equivalent object.
    """
    #return pyproj.crs.CRS.from_wkt(crs.to_wkt())
    return crs


def from_fiona(crs):
    """Converts a fiona CRS to the workflow CRS standard.

    Parameters
    ----------
    crs : fiona-crs-dict
        Input fiona CRS, which is a dictionary containing an EPSG
        code.

    Returns
    -------
    out : crs-type
        Equivalent workflow CRS.

    """
    # if 'datum' in crs and crs['datum'] == 'WGS84' and 'epsg' not in crs and 'ellps' not in crs:
    #     logging.warning('Old-style datum WGS84, moving to ellipse')
    #     crs['ellps'] = crs.pop('datum')
    return CRS.from_dict(crs)

def to_fiona(crs):
    """Converts a workflow CRS to a fiona CRS.

    Parameters
    ----------
    crs : crs-type
        A workflow CRS object.

    Returns
    -------
    out : fiona-crs-dict
        Equivalent fiona CRS.

    """
    return crs.to_dict()

def from_rasterio(crs):
    """Converts from rasterio CRS to the workflow CRS standard.

    Parameters
    ----------
    crs : rasterio-crs-object
        Input rasterio crs.

    Returns
    -------
    out : crs-type
        Equivalent workflow CRS.

    """
    try:
        # from authority seems to get better results with bounds?
        return CRS.from_authority(*crs.to_authority())
    except Exception:
        return CRS.from_user_input(crs)

def to_rasterio(crs):
    """Converts a workflow CRS to a fiona CRS.

    Parameters
    ----------
    crs : crs-type
        A workflow CRS object.

    Returns
    -------
    out : rasterio.CRS
        Equivalent rasterio object.

    """
    import rasterio.crs
    return rasterio.crs.CRS.from_user_input(crs)

def from_epsg(epsg):
    """Converts from an EPSG code to a workflow CRS.

    Parameters
    ----------
    epsg : int
        An EPSG code. (see `https://epsg.io`_)

    Returns
    -------
    out : crs-type
        Equivalent workflow CRS.

    """
    return CRS.from_epsg(epsg)

def to_epsg(crs):
    """Attempts to conver to an EPSG code.

    Parameters
    ----------
    crs : crs-type

    Returns
    -------
    epsg : int
      An EPSG code, if possible.

    If not, this throws.
    """
    auth, code = crs.to_authority()
    if auth == 'EPSG':
        return code
    else:
        raise ValueError('Cannot convert CRS to EPSG code.')
        
def from_cartopy(crs):
    """Converts a cartopy CRS to a workflow CRS.

    Parameters
    ----------
    epsg : int
        An EPSG code. (see `https://epsg.io`_)

    Returns
    -------
    out : crs-type
        Equivalent workflow CRS.

    """
    return CRS.from_dict(crs.proj4_params)

def to_cartopy(crs):
    """Converts a workflow CRS to a cartopy.crs.Projection.

    Parameters
    ----------
    crs : crs-type
        The CRS to convert.

    Returns
    -------
    A cartopy.crs.Projection object for plotting.

    Adopted from: https://pyproj4.github.io/pyproj/stable/crs_compatibility.html
    """
    import cartopy.crs as ccrs

    # this is more robust, as srs could be anything (espg, etc.)
    #s1 = osr.SpatialReference()
    #s1.ImportFromProj4(crs.to_proj4())
    #srs = s1.ExportToProj4()
    srs = crs.to_dict()

    km_proj = {'lon_0': 'central_longitude',
               'lat_0': 'central_latitude',
               'x_0': 'false_easting',
               'y_0': 'false_northing',
               'k': 'scale_factor',
               'zone': 'zone',
               }
    km_globe = {'a': 'semimajor_axis',
                'b': 'semiminor_axis',
                }
    km_std = {'lat_1': 'lat_1',
              'lat_2': 'lat_2',
              }
    kw_proj = dict()
    kw_globe = dict()
    kw_std = dict()

    for k, v in srs.items():
        try:
            v = float(v)
        except:
            pass
        if k == 'proj':
            if v == 'tmerc':
                cl = ccrs.TransverseMercator
            elif v == 'lcc':
                cl = ccrs.LambertConformal
            elif v == 'merc':
                cl = ccrs.Mercator
            elif v == 'utm':
                cl = ccrs.UTM
            elif v == 'aea':
                cl = ccrs.AlbersEqualArea
            elif v == 'laea':
                cl = ccrs.LambertAzimuthalEqualArea
            elif v == 'longlat':
                cl = ccrs.PlateCarree
            elif v == 'cea':
                cl = ccrs.LambertCylindrical
                kw_globe = None
            else:
                raise NotImplementedError('Proj4-to-Cartopy needs to be updated.')
        if k in km_proj:
            kw_proj[km_proj[k]] = v
        if k in km_globe:
            kw_globe[km_globe[k]] = v
        if k in km_std:
            kw_std[km_std[k]] = v

    globe = None
    if kw_globe:
        globe = ccrs.Globe(**kw_globe)
        kw_proj['globe'] = globe
    if kw_std:
        kw_proj['standard_parallels'] = (kw_std['lat_1'], kw_std['lat_2'])

    # mercatoooor
    if cl.__name__ == 'Mercator' or cl.__name__ == 'LambertCylindrical':
        kw_proj.pop('false_easting', None)
        kw_proj.pop('false_northing', None)

    return cl(**kw_proj)

def from_string(string):
    """Returns a CRS from a proj string"""
    return CRS.from_string(string)

def from_wkt(string):
    """Returns a CRS from a WKT string specification"""
    return CRS.from_wkt(string)

def to_wkt(crs):
    """Returns the WKT string of a CRS."""
    return crs.to_wkt()

def default_crs():
    """Returns a default CRS that is functionally useful for North America.

    Returns
    -------
    out : crs-type
        The default CRS.  The user should not care what this is (if
        you do, don't use the default!) but it is EPSG:5070.

    """
    return from_epsg(5070)


def default_alaska_crs():
    """Returns a default CRS that is functionally useful for Alaska.

    Returns
    -------
    out : crs-type
        The default CRS.  The user should not care what this is (if
        you do, don't use the default!) but it is EPSG:3338.

    """
    return from_epsg(3338)

def daymet_crs():
    """Returns the CRS used by DayMet files, but in m, not km.

    Returns
    -------
    out : crs-type
        The DayMet CRS.  The user should not care what this is.

    """
    # old proj: return from_string('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs ')
    # new proj...
    return from_string('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs')

def daymet_crs_native():
    """Returns the CRS used by DayMet files natively, in km, not in m.

    Returns
    -------
    out : crs-type
        The DayMet CRS.  The user should not care what this is.

    """
    return from_string('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=km +no_defs')

def latlon_crs():
    """Returns the default latitude-longitude CRS.

    Returns
    -------
    out : crs-type
        The default CRS.  The user should not care what this is (if
        you do, don't use the default!) but it is EPSG:4269.
    """
    return from_epsg(4269)
    
def equal(crs1, crs2):
    """Tries to guess at the equality of two CRS objects.

    Note this is not trivial, just checking strings or dicts results
    in false-negatives.  Furthermore, this implementation may not be
    perfect, but it works for all those currently tested.  Please
    report bugs!
    
    Parameters
    ----------
    crs1,crs2 : crs-type
        Input workflow CRS objects.

    Returns
    -------
    out : bool
       Are equal?

    """
    return crs1 == crs2
    
