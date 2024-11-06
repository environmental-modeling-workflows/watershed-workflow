from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd

import watershed_workflow.utils

class ManagerHyRiver:
    def __init__(self,
                 protocol : str,
                 layer : Optional[str] = None,
                 id_name : Optional[str]= None):
        self._layer = layer
        if id_name is None:
            id_name = layer
        self._id_name = id_name

        if protocol == '3DHP':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.HP3D
        elif protocol == 'NHD':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.NHD
        elif protocol == 'NHDPlusHR':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.NHDPlusHR
        elif protocol == 'WBD':
            import pygeohydro.watershed
            self._protocol = pygeohydro.watershed.WBD
        elif protocol == 'WaterData':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.WaterData
        else:
            raise ValueError(f'Invalid HyRiver protocol "{protocol}"')

        
    def getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS) -> gpd.GeoDataFrame:
        """Finds all shapes in the given dataset that touch a given geometry."""

        df = self._protocol(self._layer).bygeom(geom, geom_crs)
        df['ID'] = df[self._id_name]
        df = df.set_index('ID', drop=True)
        df = watershed_workflow.utils.filterToShape(df, geom, geom_crs, 'non_point_intersection')
        return df

    
    def getShapesByID(self,
                      ids : List[str] | str) -> gpd.GeoDataFrame:
        """Finds all shapes in the given dataset of a listed set of IDs."""
        if isinstance(ids, str):
            ids = [ids,]
            
        df = self._protocol(self._layer).byid(self._id_name, ids)
        df['ID'] = df[self._id_name]
        df = df.set_index('ID', drop=True)
        return df
