from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd
from pygeohydro.watershed import WBD

class FileManagerWBD:
    """Leverages pygeohydro to download WBD data."""
    lowest_level = 12

    def getShapes(self,
                  hucs : Optional[List[str] | str] = None,
                  level : Optional[int]  = None,
                  geom : Optional[BaseGeometry] = None,
                  geom_crs : Optional[CRS] = None) -> gpd.GeoDataFrame:
        """Calls either getShapesInGeometry or getShapesByID depending upon arguments provided."""

        if hucs is not None:
            return self.getShapesByID(hucs, level)
        elif geom is not None:
            assert level is not None
            assert geom_crs is not None
            return self.getShapesInGeometry(level, geom, geom_crs)
    
    def getShapesInGeometry(self,
                            level : int,
                            geom : BaseGeometry,
                            geom_crs : CRS,
                            distance : Optional[int] = None) -> gpd.GeoDataFrame:
        """Finds all HUs in the WBD dataset at a given level that touch a given geometry."""

        layer = f'huc{level}'
        wbd = WBD(layer)
        df = wbd.bygeom(geom, geom_crs, distance=distance)
        df['ID'] = df[layer]
        df = df.set_index('ID', drop=True)
        return df
    
    def getShapesByID(self,
                      hucs : List[str] | str,
                      level : Optional[int] = None):
        """Finds all HUs in the WBD dataset of a given level contained in a list of HUCs.""" 

        if not isinstance(hucs, list):
            hucs = [hucs,]

        req_levels = set(len(l) for l in hucs)
        if len(req_levels) != 1:
            raise ValueError("FileManagerWBD.getShapesByID can only be called with a list of HUCs of the same level")
        req_level = req_levels.pop()

        if level is not None and level != req_level:
            geom_df = self.getShapesByID(hucs)
            df = self.getShapesInGeometry(level, geom_df.union_all(), geom_df.crs)
            layer = f'huc{level}'
            return df.loc[df[layer].apply(lambda l : any(l.startswith(huc) for huc in hucs))]
        else:
            layer = f'huc{req_level}'
            wbd = WBD(layer)
            df = wbd.byids(field=layer, fids=hucs)
            df['ID'] = df[layer]
            df = df.set_index('ID', drop=True)
            return df

