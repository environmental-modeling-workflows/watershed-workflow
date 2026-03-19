from typing import List, Optional
import logging
import geopandas as gpd

import watershed_workflow.crs

from . import standard_names as names
from . import manager_hyriver

class ManagerWBD(manager_hyriver.ManagerHyRiver):
    """Leverages pygeohydro to download WBD data."""
    lowest_level = 12

    def __init__(self, product_short : str = 'WBD'):
        """Initialize WBD manager.

        Parameters
        ----------
        product_short : str, optional
            HyRiver product: ``'WBD'`` (default) or ``'WaterData'``.
        """
        self._level : Optional[int] = None

        from .manager import ManagerAttributes
        attrs = ManagerAttributes(
            category='geometry',
            product='Watershed Boundary Dataset',
            product_short=product_short,
            source='hyriver',
            url='https://www.usgs.gov/national-hydrography/watershed-boundary-dataset',
            license='public domain',
            citation='USGS WBD',
            description='USGS Watershed Boundary Dataset (WBD) hydrologic unit polygons.',
            native_crs_in=watershed_workflow.crs.from_epsg(4269),
            native_resolution=0.001,
            valid_variables=['huc2'],
            default_variables=['huc2'],
        )
        super().__init__(attrs)

    def set(self, **kwargs):
        if 'level' in kwargs:
            self.setLevel(kwargs['level'])
        
    def setLevel(self, level : int) -> None:
        self._level = level
        if self.attrs.product_short == 'WBD':
            self.attrs.default_variables[0] = f'huc{level}'
            self.attrs.native_id_field = self.attrs.default_variables[0]
        else:
            self.attrs.default_variables[0] = f'wbd{level:02d}'
            self.attrs.native_id_field = f'huc{level}'
        
    def _getShapesByID(self,
                      hucs : List[str]) -> gpd.GeoDataFrame:
        """Finds all HUs in the WBD dataset of a given level contained in a list of HUCs.""" 
        req_levels = set(len(l) for l in hucs)
        if len(req_levels) != 1:
            raise ValueError("ManagerWBD.getShapesByID can only be called with a list of HUCs of the same level")
        req_level = req_levels.pop()

        if self._level is not None and self._level != req_level:
            level = self._level
            self.setLevel(req_level)
            geom_df = self._getShapesByID(hucs)
            self.setLevel(level)

            df = self.getShapesByGeometry(geom_df.union_all(), geom_df.crs, geom_df.crs)
            return df[df.ID.apply(lambda l : any(l.startswith(huc) for huc in hucs))]
        else:
            self.setLevel(req_level)
            df = super()._getShapesByID(hucs)
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
        if self.attrs.native_id_field is not None and self.attrs.native_id_field in df.columns:
            df[names.ID] = df[self.attrs.native_id_field].astype('string')
        
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
