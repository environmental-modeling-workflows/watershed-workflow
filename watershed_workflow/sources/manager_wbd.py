from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd

import watershed_workflow.crs
import watershed_workflow.sources.standard_names as names
from watershed_workflow.sources.manager_hyriver import ManagerHyRiver


class ManagerWBD(ManagerHyRiver):
    """Leverages pygeohydro to download WBD data."""
    lowest_level = 12

    def __init__(self, protocol_name : str = 'WBD'):
        """Also valid is WaterData"""
        self._level : Optional[int] = None
        
        # WBD data is typically in lat/lon coordinates
        native_crs_in = watershed_workflow.crs.from_epsg(4269)
        native_resolution = 0.001  # ~100m at mid-latitudes
        
        super().__init__(protocol_name, native_crs_in, native_resolution)
        self.name = 'WBD'

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

    def _addStandardNames(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert native column names to standard names.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame with native column names.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with standard column names added.
        """
        # Add ID column from current ID field (set by setLevel)
        if hasattr(self, '_id_name') and self._id_name in df.columns:
            df[names.ID] = df[self._id_name].astype('string')
        
        # Add WBD-specific standard name mappings
        if 'areasqkm' in df.columns:
            df[names.AREA] = df['areasqkm']
        
        # Add HUC field if level is set
        if self._level is not None:
            huc_field = f'huc{self._level}'
            if huc_field in df.columns:
                df[names.HUC] = df[huc_field]
        
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
