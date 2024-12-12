from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd
import pandas as pd

from watershed_workflow.sources.manager_hyriver import ManagerHyRiver
import watershed_workflow.sources.standard_names as names

water_data_ids = {'nhdflowline_network' : 'comid',
                  'nhdflowline_nonnetwork' : 'comid',
                  'catchmentsp' : 'featureid',
                  'nhdwaterbody' : 'comid'
                  }

class ManagerWaterData(ManagerHyRiver):
    """Leverages WaterData to download NHDv2.1? data and its supporting shapes."""
    lowest_level = 12

    def __init__(self, layer : str):
        super(ManagerWaterData, self).__init__('WaterData')
        self.setLayer(layer)
        

    def setLayer(self, layer : str) -> None:
        self._layer = layer
        if layer in water_data_ids:
            self._id_name = water_data_ids[layer]
        else:
            self._id_name = self._layer

    def getCatchments(self,
                      df : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if self._layer.startswith('nhd'):
            # also get catchments
            old_layer = self._layer
            self.setLayer('catchmentsp')
            ids = df.comid
            cas_raw = self.getShapesByID(ids)
            self.setLayer(old_layer)

            df = pd.merge(df, cas_raw, how='outer', left_on='ID', right_on='ID', suffixes=(None, '_ca'))
        return df

    def addStandardNames(self, df):
        try:
            df[names.LENGTH] = df['lengthkm']
            df[names.AREA] = df['areasqkm']
            df[names.ORDER] = df['streamorde']
            df[names.DRAINAGE_AREA] = df['totdasqkm']
        except KeyError:
            pass
        return df
            
    def getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS) -> gpd.GeoDataFrame:
        df = super(ManagerWaterData, self).getShapesByGeometry(geom, geom_crs)
        df = self.getCatchments(df)
        df = self.addStandardNames(df)        
        return df

    
    def getShapesByID(self,
                      ids : List[str] | str) -> gpd.GeoDataFrame:
        df = super(ManagerWaterData, self).getShapesByID(ids)
        df = self.getCatchments(df)
        df = self.addStandardNames(df)        
        return df
    
