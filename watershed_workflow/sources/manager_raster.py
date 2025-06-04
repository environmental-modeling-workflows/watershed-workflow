"""Basic manager for interacting with raster files.
"""

import numpy as np
import attr
import xarray
import shapely
import rioxarray
from typing import Tuple
import geopandas as gpd
from shapely.geometry import mapping
import watershed_workflow.crs
import logging


@attr.s
class ManagerRaster:
    """A simple class for reading rasters.

    Parameters
    ----------
    filename : str
      Path to the raster file.
    """
    _filename = attr.ib(type=str)
    name = 'raster'


    def getDataset(self,
                   geometry : shapely.geometry.Polygon | \
                              shapely.geometry.MultiPolygon | \
                              Tuple[float,float,float,float],
                   geometry_crs : watershed_workflow.crs.CRS,
                   band : int = 1) -> xarray.DataArray:
        """Read a raster as a dataset on this shape, clipping to the shape.
        
        Parameters
        ----------
        geometry : shapely.geometry.Polygon | \
                   shapely.geometry.MultiPolygon | \
                   Tuple[float,float,float,float]
          Shape to provide bounds of the raster.
        geometry_crs : watershed_workflow.crs.CRS
          CRS of the shape.
        band : int,optional
          Default is 1, the first band (1-indexed).

        Returns
        -------
        dataset : xarray.DataArray
          Dataset containing the raster.

        Note that the raster provided is in its native CRS (which is in the
        rasterio profile), not the shape's CRS.
        """
        
        dataset = rioxarray.open_rasterio(self._filename, masked=False)
        gdf = gpd.GeoDataFrame(geometry=[geometry,], crs=geometry_crs)
        if not watershed_workflow.crs.isEqual(geometry_crs, dataset.rio.crs):
            gdf = gdf.to_crs(dataset.rio.crs)
        return dataset.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
  