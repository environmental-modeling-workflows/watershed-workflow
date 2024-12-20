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
from __future__ import annotations

import logging
import pyproj.crs
from pyproj.crs import CRS

import typing
if typing.TYPE_CHECKING:
    import rasterio.crs
    import cartopy.crs
    import xarray

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def isNative(crs : CRS) -> bool:
    """Is this crs in the native format?"""
    return type(crs) == type(to_proj(crs))


def from_proj(crs : CRS) -> CRS:
    """Converts a Proj CRS to the workflow CRS standard."""
    return crs


def to_proj(crs : CRS) -> CRS:
    """Converts a workflow CRS standard to a Proj4 CRS."""
    return crs


def from_rasterio(crs : rasterio.crs.CRS) -> CRS:
    """Converts from rasterio CRS to the workflow CRS standard."""
    try:
        # from authority seems to get better results with bounds?
        return CRS.from_authority(*crs.to_authority())
    except Exception:
        return CRS.from_user_input(crs)


def to_rasterio(crs : CRS) -> rasterio.crs.CRS:
    """Converts a workflow CRS to a rasterio CRS."""
    import rasterio.crs
    return rasterio.crs.CRS.from_user_input(crs)


def from_epsg(epsg : int) -> CRS:
    """Converts from an EPSG code to a workflow CRS."""
    return CRS.from_epsg(epsg)


def to_epsg(crs : CRS) -> int:
    """Attempts to convert to an EPSG code."""
    auth, code = crs.to_authority()
    if auth == 'EPSG':
        return code
    else:
        raise ValueError('Cannot convert CRS to EPSG code.')


def from_cartopy(crs : cartopy.crs.CRS) -> CRS:
    """Converts a cartopy CRS to a workflow CRS."""
    return CRS.from_user_input(crs)


def to_cartopy(crs : CRS) -> cartopy.crs.CRS:
    """Converts a workflow CRS to a cartopy.crs.Projection.

    Adopted from: https://pyproj4.github.io/pyproj/stable/crs_compatibility.html
    """
    import packaging.version
    assert packaging.version.Version(cartopy.__version__) >= packaging.version.Version('0.20.0')
    import cartopy.crs as ccrs
    return ccrs.CRS(crs)
    

def from_string(string : str) -> CRS:
    """Returns a CRS from a proj string"""
    return CRS.from_string(string)


def from_wkt(string : str) -> CRS:
    """Returns a CRS from a WKT string specification"""
    return CRS.from_wkt(string)


def to_wkt(crs : CRS) -> str:
    """Returns the WKT string of a CRS."""
    return crs.to_wkt()


def from_xarray(array : xarray.DataArray) -> CRS | None:
    """Tries to find a CRS from the xarray DataSet or DataArray."""
    wkt = None
    try:
        wkt = array.spatial_ref.attrs['crs_wkt']
    except (KeyError, AttributeError):
        pass

    if wkt is not None:
        return from_wkt(wkt)
    return None


# a default UTM based CRS that is functionally useful for North America.
default_crs = from_epsg(5070)

# a default UTM based CRS that is functionally useful for Alaska
default_alaska_crs = from_epsg(3338)

# DayMet's CRS, in m
daymet_crs = from_string(
    '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
)

# DayMet's native CRS, which is in km, not m
daymet_crs_km = from_string(
    '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=km +no_defs'
)

# default lat-lon CRS
latlon_crs = from_epsg(4269)


def isEqual(crs1 : CRS, crs2 : CRS) -> bool:
    """Tries to guess at the equality of two CRS objects."""
    return crs1 == crs2
