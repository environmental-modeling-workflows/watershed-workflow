from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd
from watershed_workflow.sources.manager_hyriver import ManagerHyRiver
import watershed_workflow.sources.standard_names as names


class ManagerWBD(ManagerHyRiver):
    """Leverages pygeohydro to download WBD data."""
    lowest_level = 12

    def __init__(self, protocol : str = 'WBD'):
        """Also valid is WaterData"""
        self._protocol_name : str = protocol
        self._level : Optional[int] = None
        super(ManagerWBD, self).__init__(protocol)

        self.name = 'WBD'
        if protocol != 'WBD':
            self.name = ' '.join(['WBD', protocol])

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

            df[names.HUC] = df[f'huc{self._level}']
            df[names.AREA] = df[f'areasqkm']
            
            return df.loc[df.index.to_series().apply(lambda l : any(l.startswith(huc) for huc in hucs))]
        else:
            self.setLevel(req_level)
            df = super(ManagerWBD, self).getShapesByID(hucs)

            df[names.HUC] = df[f'huc{req_level}']
            df[names.AREA] = df[f'areasqkm']

            return df


    def getAll(self,
               level : int) -> gpd.GeoDataFrame:
        """Download all HUCs at a given level."""
        # this is a shortcut...
        import pygeohydro.watershed
        df = pygeohydro.watershed.huc_wb_full(level)
        df[names.HUC] = df[f'huc{level}']
        df[names.AREA] = df[f'areasqkm']
        return df
