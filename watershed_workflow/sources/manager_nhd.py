from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
import geopandas as gpd
import pandas as pd

from watershed_workflow.sources.manager_hyriver import ManagerHyRiver
import watershed_workflow.sources.standard_names as names

waterdata_ids = {'nhdflowline_network' : 'comid',
                  'catchmentsp' : 'featureid',
                  }

waterdata_renames = {'gnis_name' : names.NAME,
                     'lengthkm' : names.LENGTH,
                     'areasqkm' : names.CATCHMENT_AREA,
                     'streamorde' : names.ORDER,
                     'totdasqkm' : names.DRAINAGE_AREA,
                     'geometry_ca' : names.CATCHMENT,
                     'hydroseq' : names.HYDROSEQ,
                     'uphydroseq' : names.UPSTREAM_HYDROSEQ,
                     'dnhydroseq' : names.DOWNSTREAM_HYDROSEQ,
                     'divergence' : names.DIVERGENCE,
                     }

hr_ids = {'flowline' : 'nhdplusid',
          'catchment' : 'nhdplusid',
          }
hr_renames = { 'gnis_name' : names.NAME,
               'lengthkm' : names.LENGTH,
               'areasqkm' : names.CATCHMENT_AREA,
               'streamorde' : names.ORDER,
               'totdasqkm' : names.DRAINAGE_AREA,
               'geometry_ca' : names.CATCHMENT,
               'hydroseq' : names.HYDROSEQ,
               'uphydroseq' : names.UPSTREAM_HYDROSEQ,
               'dnhydroseq' : names.DOWNSTREAM_HYDROSEQ,
               'divergence' : names.DIVERGENCE,
              }

mr_ids = {'flowline_mr' : 'COMID', }
mr_renames = { 'GNIS_NAME' : names.NAME,
               'LENGTHKM' : names.LENGTH,
               'AreaSqKM' : names.CATCHMENT_AREA,
               'StreamOrde' : names.ORDER,
               'TotDASqKM' : names.DRAINAGE_AREA,
               'Hydroseq' : names.HYDROSEQ,
               'UpHydroseq' : names.UPSTREAM_HYDROSEQ,
               'DnHydroseq' : names.DOWNSTREAM_HYDROSEQ,
               'Divergence' : names.DIVERGENCE,
              }


def _tryRename(df, old, new):
    try:
        df[new] = df[old]
    except KeyError:
        pass

class ManagerNHD(ManagerHyRiver):
    """Leverages pynhd to download NHD data and its supporting shapes."""
    lowest_level = 12

    def __init__(self,
                 protocol : str,
                 layer : Optional[str] = None,
                 catchments : Optional[bool] = True):
        self._catchment_layer = None
        
        if protocol == 'NHDPlus MR v2.1':
            self._protocol = 'WaterData'
            self._ids = waterdata_ids
            self._renames = waterdata_renames
            if layer is None:
                layer = 'nhdflowline_network'
            if layer == 'nhdflowline_network':
                self._catchment_layer = 'catchmentsp'

        elif protocol == 'NHDPlus HR':
            self._protocol = 'NHDPlusHR'
            self._ids = hr_ids
            self._renames = hr_renames
            if layer is None:
                layer = 'flowline'
            if layer == 'flowline':
                self._catchment_layer = 'catchment'

        elif protocol == 'NHD MR':
            self._protocol = 'NHD'
            self._ids = mr_ids
            self._renames = mr_renames
            if layer is None:
                layer = 'flowline_mr'

        else:
            raise ValueError(f'Invalid ManagerNHD protocol {protocol}')

        super(ManagerNHD, self).__init__(self._protocol)
        self.setLayer(layer)
        self._catchments = catchments
        
    def setLayer(self, layer : str) -> None:
        self._layer = layer
        if layer in self._ids:
            self._id_name = self._ids[layer]
        else:
            self._id_name = self._layer

    def getCatchments(self,
                      df : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if self._catchment_layer is not None:
            # also get catchments
            old_layer = self._layer
            self.setLayer(self._catchment_layer)
            ids = df.index
            cas_raw = super(ManagerNHD, self).getShapesByID(ids)
            self.setLayer(old_layer)

            df = pd.merge(df, cas_raw, how='outer', left_on=names.ID,
                          right_on=names.ID, suffixes=(None, '_ca'))
        return df

    def addStandardNames(self, df):
        for k,v in self._renames.items():
            _tryRename(df, k, v)
        return df
            
    def getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS) -> gpd.GeoDataFrame:
        df = super(ManagerNHD, self).getShapesByGeometry(geom, geom_crs)
        if self._catchments:
            df = self.getCatchments(df)
        df = self.addStandardNames(df)        
        return df
    
    def getShapesByID(self,
                      ids : List[str] | str) -> gpd.GeoDataFrame:
        df = super(ManagerNHD, self).getShapesByID(ids)
        if self._catchments:
            df = self.getCatchments(df)
        df = self.addStandardNames(df)        
        return df
    


