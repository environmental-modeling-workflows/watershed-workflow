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
import pyproj

def from_proj(crs):
    """Converts a Proj4 CRS to the workflow CRS standard.

    Parameters
    ----------
    crs : pyproj.Proj
        Input proj CRS object.
    
    Returns
    -------
    out : crs-type
        Equivalent workflow CRS.
    """
    return crs # this is the standard!


def to_proj(crs):
    """Converts a workflow CRS standard to a Proj4 CRS.

    Parameters
    ----------
    crs : crs-type
        Workflow standard CRS.
    
    Returns
    -------
    out : pyproj.Proj
        Equivalent object.
    """
    return crs # this is the standard!


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
    return pyproj.Proj(crs)

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
    import fiona.crs
    return fiona.crs.from_string(crs.definition_string())

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
    return pyproj.Proj(crs)

def to_rasterio(crs):
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
    import rasterio.crs
    return rasterio.crs.CRS.from_string(crs.definition_string())

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
    try:
        return pyproj.Proj(init='EPSG:{}'.format(epsg))
    except RuntimeError as err:
        raise RuntimeError('Proj4 error: "{}", likely epsg is not recognized by pyproj.'.format(err))

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
    return pyproj.Proj(crs.proj4_init)

def to_cartopy(crs):
    """Converts a workflow CRS to a cartopy.crs.Projection.

    Parameters
    ----------
    crs : crs-type
        The CRS to convert.

    Returns
    -------
    A cartopy.crs.Projection object for plotting.

    Adapted from: https://github.com/fmaussion/salem/blob/d0aaefab96bd4099c280d5088d4e66b52d20b72b/salem/gis.py#L901-L983
    Better solution blocked by: https://github.com/SciTools/cartopy/issues/813
    """
    import cartopy.crs as ccrs
    import osr
    if crs.is_latlong():
        return ccrs.PlateCarree()

    srs = crs.srs

    # this is more robust, as srs could be anything (espg, etc.)
    s1 = osr.SpatialReference()
    s1.ImportFromProj4(crs.srs)
    srs = s1.ExportToProj4()

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
    for s in srs.split('+'):
        s = s.split('=')
        if len(s) != 2:
            continue
        k = s[0].strip()
        v = s[1].strip()
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
    if kw_std:
        kw_proj['standard_parallels'] = (kw_std['lat_1'], kw_std['lat_2'])

    # mercatoooor
    if cl.__name__ == 'Mercator':
        kw_proj.pop('false_easting', None)
        kw_proj.pop('false_northing', None)

    return cl(globe=globe, **kw_proj)

def from_string(string):
    """Returns a CRS from a Proj4 string specification."""
    return pyproj.Proj(string)

def to_string(crs):
    """Returns the proj4 string of a CRS."""
    return crs.definition_string()

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
    """Returns teh CRS used by DayMet files.

    Returns
    -------
    out : crs-type
        The DayMet CRS.  The user should not care what this is.

    """
    return from_string('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs ')


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
    def to_dict(crs):
        def to_item(a):
            item = a.strip('+').split('=')
            if len(item) == 1:
                return item[0], True
            elif len(item) == 2:
                return tuple(item)
            else:
                raise ValueError('Invalid proj string: "{}"'.format(a))

        return dict(map(to_item, crs.definition_string().split()))

    d1 = to_dict(crs1)
    d2 = to_dict(crs2)

    if (d1['ellps'] != d2['ellps']):
        logging.debug('CRS not equal: ', 'ellps:', d1['ellps'], d2['ellps'])
        return False
    if (('datum' in d1) and ('datum' in d2) and (d1['datum'] != d2['datum'])):
        print('datum:', d1['datum'], d2['datum'])
        return False
    if (d1['proj'] != d2['proj']):
        logging.debug('CRS not equal: ', 'proj:', d1['proj'], d2['proj'])
        return False
    if (d1['units'] != d2['units']):
        logging.debug('CRS not equal: ', 'units:', d1['units'], d2['units'])
        return False

    for k in ['lat_0', 'lat_1', 'lat_2', 'lon_0', 'lon_1', 'lon_2']:
        if (k in d1) != (k in d2):
            logging.debug('CRS not equal: ', k, k in d1, k in d2)
            return False
        if k in d1:
            if (d1[k] != d2[k]):
                logging.debug('CRS not equal: ', k, d1[k], d2[k])
                return False
    return True

    
