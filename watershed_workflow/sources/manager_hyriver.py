from __future__ import annotations

from typing import List, Optional, Any
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd

import watershed_workflow.utils
import watershed_workflow.sources.standard_names as names


class ManagerHyRiver:
    """A generic base class for working with HyRiver"""

    def __init__(self,
                 protocol : str,
                 layer : str = '',
                 id_name : Optional[str]= None):
        self.name = protocol
        self._layer = layer
        if id_name is None:
            id_name = layer
        self._id_name = id_name
        self._protocol : Any = None

        if protocol == 'NHD':
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

    def set(self, **kwargs):
        pass
        
    def getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS) -> gpd.GeoDataFrame:
        """Finds all shapes in the given dataset that touch a given geometry."""

        df = self._protocol(self._layer).bygeom(geom, geom_crs)
        df[names.ID] = df[self._id_name].astype('string')
        df = df.set_index(names.ID, drop=True)
        df = watershed_workflow.utils.filterToShape(df, geom, geom_crs, 'non_point_intersection')
        return df

    
    def getShapesByID(self,
                      ids : List[str] | str) -> gpd.GeoDataFrame:
        """Finds all shapes in the given dataset of a listed set of IDs."""
        if isinstance(ids, str):
            ids = [ids,]

        protocol = self._protocol(self._layer)
        if hasattr(protocol, 'byid'):
            df = protocol.byid(self._id_name, ids)
        else:
            df = protocol.byids(self._id_name, ids)

        df[names.ID] = df[self._id_name].astype('string')
        df = df.set_index(names.ID, drop=True)
        return df
