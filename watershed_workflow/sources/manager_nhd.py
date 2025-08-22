from typing import List, Optional
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import pandas as pd

from watershed_workflow.crs import CRS
import watershed_workflow.crs
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
                 dataset_name: str,
                 layer: Optional[str] = None,
                 catchments: Optional[bool] = True):
        """Initialize NHD manager.
        
        Parameters
        ----------
        dataset_name : str
            NHD dataset name ('NHDPlus MR v2.1', 'NHDPlus HR', 'NHD MR').
        layer : str, optional
            Layer name, defaults to protocol-specific default.
        catchments : bool, optional
            Whether to fetch catchments with flowlines, defaults to True.
        """
        self._catchment_layer = None
        
        if dataset_name == 'NHDPlus MR v2.1':
            self._protocol_name = 'WaterData'
            self._ids = waterdata_ids
            self._renames = waterdata_renames
            if layer is None:
                layer = 'nhdflowline_network'
            if layer == 'nhdflowline_network':
                self._catchment_layer = 'catchmentsp'

        elif dataset_name == 'NHDPlus HR':
            self._protocol_name = 'NHDPlusHR'
            self._ids = hr_ids
            self._renames = hr_renames
            if layer is None:
                layer = 'flowline'
            if layer == 'flowline':
                self._catchment_layer = 'catchment'

        elif dataset_name == 'NHD MR':
            self._protocol_name = 'NHD'
            self._ids = mr_ids
            self._renames = mr_renames
            if layer is None:
                layer = 'flowline_mr'

        else:
            raise ValueError(f'Invalid ManagerNHD dataset_name {dataset_name}')

        # NHD data is typically in lat/lon coordinates
        native_crs_in = watershed_workflow.crs.latlon_crs
        # Rough resolution estimate for degree-based data
        native_resolution = 0.001  # ~100m at mid-latitudes

        # Get ID name for this layer
        if layer in self._ids:
            id_name = self._ids[layer]
        else:
            id_name = layer

        super().__init__(self._protocol_name, native_crs_in, native_resolution, layer, id_name)
        self.name = dataset_name
        self._catchments = catchments

    def getCatchments(self,
                      df : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add catchment data to flowline data.
        
        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame with flowline data and ID column.
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with catchment data merged in.
        """
        if self._catchment_layer is not None:
            # Save current layer and switch to catchment layer
            old_layer = self._layer
            old_id_name = self._id_name
            
            # Set catchment layer properties
            self._layer = self._catchment_layer
            if self._catchment_layer in self._ids:
                self._id_name = self._ids[self._catchment_layer]
            else:
                self._id_name = self._catchment_layer
            
            # Get catchments using HyRiver directly (no recursive catchment fetching)
            ids = df[old_id_name].tolist()
            cas_raw = ManagerHyRiver._getShapesByID(self, ids)
            
            # Apply standard names to catchments
            cas_raw = self._addStandardNames(cas_raw)

            # Merge catchments with flowlines
            df = pd.merge(df, cas_raw, how='outer', left_on=old_id_name,
                          right_on=self._id_name, suffixes=(None, '_ca'))

            # Restore original layer properties
            self._layer = old_layer
            self._id_name = old_id_name

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
        # Add ID column from native ID field
        if self.native_id_field in df.columns:
            df[names.ID] = df[self.native_id_field].astype('string')
        
        # Add other standard name mappings
        for k, v in self._renames.items():
            _tryRename(df, k, v)
        return df
            
    def _getShapesByGeometry(self, geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch NHD shapes by geometry, including catchments if requested.

        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and catchments if requested.
        """
        # Get base shapes from HyRiver
        df = super()._getShapesByGeometry(geometry_gdf)
        
        # Add catchments if requested
        if self._catchments:
            df = self.getCatchments(df)
        
        return df
    
    def _getShapesByID(self, ids) -> gpd.GeoDataFrame:
        """Fetch NHD shapes by ID, including catchments if requested.

        Parameters
        ----------
        ids : List[str]
            List of IDs to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and catchments if requested.
        """
        # Get base shapes from HyRiver
        df = super()._getShapesByID(ids)
        
        # Add catchments if requested
        if self._catchments:
            df = self.getCatchments(df)
        
        return df
    

