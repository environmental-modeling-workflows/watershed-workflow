"""Basic manager for interacting with raster files.
"""

import numpy as np
import attr
import xarray as xr
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
                   band : int = -1) -> xr.DataArray:
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
        dataset : xr.DataArray
          Dataset containing the raster.

        Note that the raster provided is in its native CRS (which is in the
        rasterio profile), not the shape's CRS.
        """
        if isinstance(geometry, shapely.geometry.base.BaseGeometry):
            bounds = geometry.bounds
        else:
            bounds = geometry

        if not self._filename.lower().endswith('.tif'):
            dataset = rioxarray.open_rasterio(self._filename, chunk='auto')
        else:
            dataset = rioxarray.open_rasterio(self._filename, cache=False)
        assert isinstance(dataset, xr.Dataset) or isinstance(dataset, xr.DataArray)
        dataset = dataset.rio.clip_box(*bounds, crs=watershed_workflow.crs.to_rasterio(geometry_crs))

        if len(dataset.shape) > 2:
            if band > 0:
                dataset_out = dataset[band-1,:,:]
                dataset_out.rio.write_crs(dataset.rio.crs)
                dataset = dataset_out

            elif dataset.shape[0] == 1:
                dataset_out = dataset[0,:,:]
                dataset_out.rio.write_crs(dataset.rio.crs)
                dataset = dataset_out
        
        return dataset
  
