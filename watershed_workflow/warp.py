"""Used to warp shapefiles and rasters into new coordinate systems."""

from typing import Tuple
import shutil
import numpy as np
import logging

import xarray as xr
import rasterio
import pyproj
import rasterio.warp
import shapely.geometry
import shapely.ops

import warnings
import watershed_workflow.crs
from watershed_workflow.crs import CRS

pyproj_version = int(pyproj.__version__[0])


def xy(x: np.ndarray, y: np.ndarray, old_crs: CRS, new_crs: CRS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp a set of points from old_crs to new_crs.
    
    Parameters
    ----------
    x : numpy.ndarray
        X coordinates in the old coordinate system.
    y : numpy.ndarray  
        Y coordinates in the old coordinate system.
    old_crs : CRS
        Source coordinate reference system.
    new_crs : CRS
        Target coordinate reference system.
        
    Returns
    -------
    x_transformed : numpy.ndarray
        X coordinates in the new coordinate system.
    y_transformed : numpy.ndarray
        Y coordinates in the new coordinate system.
        
    Notes
    -----
    If the coordinate systems are equal, returns the input coordinates unchanged.
    """
    if watershed_workflow.crs.isEqual(old_crs, new_crs):
        return x, y

    old_crs_proj = watershed_workflow.crs.to_proj(old_crs)
    new_crs_proj = watershed_workflow.crs.to_proj(new_crs)

    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    x1, y1 = transformer.transform(x, y)
    return x1, y1


def points(array: np.ndarray, old_crs: CRS, new_crs: CRS) -> np.ndarray:
    """
    Warp an array of points from old_crs to new_crs.
    
    Parameters
    ----------
    array : numpy.ndarray
        Array of shape (N, 2) containing x,y coordinates.
    old_crs : CRS
        Source coordinate reference system.
    new_crs : CRS
        Target coordinate reference system.
        
    Returns
    -------
    numpy.ndarray
        Transformed array of shape (N, 2) with warped coordinates.
    """
    x, y = xy(array[:, 0], array[:, 1], old_crs, new_crs)
    return np.array([x, y]).transpose()


def bounds(bounds: Tuple[float, float, float, float], old_crs: CRS,
           new_crs: CRS) -> Tuple[float, float, float, float]:
    """
    Warp a bounding box from old_crs to new_crs.
    
    Parameters
    ----------
    bounds : tuple of float
        Bounding box as (minx, miny, maxx, maxy).
    old_crs : CRS
        Source coordinate reference system.
    new_crs : CRS
        Target coordinate reference system.
        
    Returns
    -------
    tuple of float
        Transformed bounding box as (minx, miny, maxx, maxy).
        
    Notes
    -----
    Creates a box geometry from the bounds, transforms it, then returns the
    new bounds of the transformed box.
    """
    return shply(shapely.geometry.box(*bounds), old_crs, new_crs).bounds


def shply(shp: shapely.geometry.base.BaseGeometry, old_crs: CRS,
          new_crs: CRS) -> shapely.geometry.base.BaseGeometry:
    """
    Warp a shapely geometry object from old_crs to new_crs.
    
    Parameters
    ----------
    shp : shapely.geometry.base.BaseGeometry
        Shapely geometry object to transform.
    old_crs : CRS
        Source coordinate reference system.
    new_crs : CRS
        Target coordinate reference system.
        
    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Transformed shapely geometry object.
        
    Notes
    -----
    If the coordinate systems are equal, returns the input geometry unchanged.
    Preserves any 'properties' attribute on the input geometry.
    """
    if watershed_workflow.crs.isEqual(old_crs, new_crs):
        return shp
    old_crs_proj = watershed_workflow.crs.to_proj(old_crs)
    new_crs_proj = watershed_workflow.crs.to_proj(new_crs)
    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    shp_out = shapely.ops.transform(transformer.transform, shp)
    if hasattr(shp, 'properties'):
        shp_out.properties = shp.properties
    return shp_out


def shplys(shps: list, old_crs: CRS, new_crs: CRS) -> list:
    """
    Warp a collection of shapely geometry objects from old_crs to new_crs.
    
    Parameters
    ----------
    shps : list of shapely.geometry.base.BaseGeometry
        Collection of shapely geometry objects to transform.
    old_crs : CRS
        Source coordinate reference system.
    new_crs : CRS
        Target coordinate reference system.
        
    Returns
    -------
    list of shapely.geometry.base.BaseGeometry
        List of transformed shapely geometry objects.
        
    Notes
    -----
    If the coordinate systems are equal, returns the input geometries unchanged.
    Preserves any 'properties' attribute on each input geometry.
    """
    if watershed_workflow.crs.isEqual(old_crs, new_crs):
        return shps
    old_crs_proj = watershed_workflow.crs.to_proj(old_crs)
    new_crs_proj = watershed_workflow.crs.to_proj(new_crs)
    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    shps_out = [shapely.ops.transform(transformer.transform, shp) for shp in shps]
    for sout, sin in zip(shps_out, shps):
        if hasattr(sin, 'properties'):
            sout.properties = sin.properties
    return shps_out


def dataset(ds: xr.Dataset,
            target_crs: CRS,
            lat_dim: str = "latitude",
            lon_dim: str = "longitude",
            resampling_method: str = "nearest") -> xr.Dataset:
    """
    Reproject an xarray Dataset from its current CRS to a target CRS using rioxarray.
    Maintains the same width and height as the original dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with CRS information (ds.rio.crs must be set)
    target_crs : pyproj.CRS
        Target coordinate reference system as a pyproj.CRS object
    lat_dim : str, default "latitude"
        Name of the latitude dimension in the input dataset
    lon_dim : str, default "longitude"
        Name of the longitude dimension in the input dataset
    resampling_method : str, default "nearest"
        Resampling method for reprojection (nearest, bilinear, cubic, etc.)
        
    Returns
    -------
    xr.Dataset
        Reprojected dataset with x/y coordinates instead of lat/lon
        
    Examples
    --------
    >>> from pyproj import CRS
    >>> # Load a dataset with lat/lon coordinates  
    >>> ds = xr.open_dataset('data.nc')
    >>> ds = ds.rio.write_crs("EPSG:4326")  # Set CRS if not already set
    >>> 
    >>> # Reproject to Albers Equal Area
    >>> target_crs = CRS.from_epsg(5070)
    >>> ds_projected = dataset(ds, target_crs=target_crs)
    """

    # Get source CRS from the dataset
    source_crs = ds.rio.crs
    if source_crs is None:
        raise ValueError("Dataset does not have CRS information. Set it using ds.rio.write_crs()")

    # set the crs attribute to all datasets
    ds_copy = ds.copy()
    for var in ds_copy:
        ds_copy[var] = ds_copy[var].rio.write_crs(source_crs)

    # Reproject dataset
    ds_reprojected = ds_copy.rio.reproject(target_crs,
                                           resampling=getattr(rasterio.enums.Resampling,
                                                              resampling_method))
    return ds_reprojected
