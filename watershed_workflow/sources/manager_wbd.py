from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd
from watershed_workflow.sources.manager_hyriver import ManagerHyRiver

class ManagerWBD(ManagerHyRiver):
    """Leverages pygeohydro to download WBD data."""
    lowest_level = 12

    def __init__(self, protocol='WBD'):
        """Also valid is WaterData"""
        self._protocol_name = protocol
        super(ManagerWBD, self).__init__(protocol)

    def setLayer(self, level : int):
        if self._protocol_name == 'WBD':
            self._layer = f'huc{level}'
            self._id_name = self._layer
        else:
            self._layer = f'wbd{level:02d}'
            self._id_name = f'huc{level}'
        
    def getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS,
                            level : int) -> gpd.GeoDataFrame:
        """Finds all HUs in the WBD dataset at a given level that touch a given geometry."""
        self.setLayer(level)
        return super(ManagerWBD, self).getShapesByGeometry(geom, geom_crs)
    
    def getShapesByID(self,
                      hucs : List[str] | str,
                      level : Optional[int] = None):
        """Finds all HUs in the WBD dataset of a given level contained in a list of HUCs.""" 
        if isinstance(hucs, str):
            hucs = [hucs,]

        req_levels = set(len(l) for l in hucs)
        if len(req_levels) != 1:
            raise ValueError("FileManagerWBD.getShapesByID can only be called with a list of HUCs of the same level")
        req_level = req_levels.pop()

        if level is not None and level != req_level:
            geom_df = self.getShapesByID(hucs)
            df = self.getShapesByGeometry(geom_df.union_all(), geom_df.crs, req_level)
            return df.loc[df['ID'].apply(lambda l : any(l.startswith(huc) for huc in hucs))]
        else:
            self.setLayer(req_level)
            return super(ManagerWBD, self).getShapesByID(hucs)

