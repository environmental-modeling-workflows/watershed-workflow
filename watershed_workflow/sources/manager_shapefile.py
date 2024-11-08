"""Basic manager for interacting with shapefiles.
"""
from __future__ import annotations
from typing import Optional, List

import attr
import pyogrio
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
import watershed_workflow.utils
from watershed_workflow.crs import CRS

@attr.s
class ManagerShapefile:
    """A simple class for reading shapefiles.

    Parameters
    ----------
    filename : str
      Path to the shapefile.
    """
    _filename = attr.ib(type=str)
    _id_name : Optional[str] = attr.ib(default=None)
    name = 'shapefile'

    
    def getShapes(self) -> gpd.GeoDataFrame:
        """Read the file and filter to get shapes"""
        return gpd.read_file(self._filename)

    
    def getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS) -> gpd.GeoDataFrame:
        """Read the file and filter to get shapes."""
        info = pyogrio.read_info(self._filename)
        file_crs = watershed_workflow.crs.from_string(info['crs'])
        geom = watershed_workflow.warp.shply(geom, geom_crs, file_crs)
        df = gpd.read_file(self._filename, geom.bounds)
        df['ID'] = range(len(df.index))
        df = df.set_index('ID', drop=True)
        df = watershed_workflow.utils.filterToShape(df, geom, geom_crs, 'non_point_intersection')
        return df

    
    def getShapesByID(self,
                      ids : List[str] | str) -> gpd.GeoDataFrame:
        """Read the file and filter to a list of IDs."""
        df = gpd.read_file(self._filename)
        df['ID'] = range(len(df.index))
        df = df.set_index('ID', drop=True)
        df = df[df[self._id_name].apply(lambda id : id in ids)]
        return df
    
        
