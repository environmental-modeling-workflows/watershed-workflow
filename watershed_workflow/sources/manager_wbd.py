from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd
from watershed_workflow.sources.manager_hyriver import ManagerHyRiver

class ManagerWBD(ManagerHyRiver):
    """Leverages pygeohydro to download WBD data."""
    lowest_level = 12

    def __init__(self, protocol : str = 'WBD'):
        """Also valid is WaterData"""
        self._protocol_name : str = protocol
        self._level : Optional[int] = None
        self.name = 'WBD'
        if protocol != 'WBD':
            self.name = ' '.join(['WBD', protocol])
        super(ManagerWBD, self).__init__(protocol)

    def set(self, **kwargs):
        if 'level' in kwargs:
            self.setLevel(kwargs['level'])
        
    def setLevel(self, level : int) -> None:
        self._level = level
        if self._protocol_name == 'WBD':
            self._layer = f'huc{level}'
            self._id_name = self._layer
        else:
            self._layer = f'wbd{level:02d}'
            self._id_name = f'huc{level}'
        
    def getShapesByID(self,
                      hucs : List[str] | str) -> gpd.GeoDataFrame:
        """Finds all HUs in the WBD dataset of a given level contained in a list of HUCs.""" 
        if isinstance(hucs, str):
            hucs = [hucs,]

        req_levels = set(len(l) for l in hucs)
        if len(req_levels) != 1:
            raise ValueError("FileManagerWBD.getShapesByID can only be called with a list of HUCs of the same level")
        req_level = req_levels.pop()

        if self._level is not None and self._level != req_level:
            level = self._level
            self.setLevel(req_level)
            geom_df = self.getShapesByID(hucs)
            self.setLevel(level)
            df = self.getShapesByGeometry(geom_df.union_all(), geom_df.crs)
            return df.loc[df.index.to_series().apply(lambda l : any(l.startswith(huc) for huc in hucs))]
        else:
            self.setLevel(req_level)
            return super(ManagerWBD, self).getShapesByID(hucs)

