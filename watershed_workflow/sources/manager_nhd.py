from typing import List, Optional
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import pandas as pd
import importlib.resources

from watershed_workflow.crs import CRS
import watershed_workflow.crs

from . import standard_names as names
from . import manager_hyriver


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
                     'BANKFULL_WIDTH' : names.BANKFULL_WIDTH,
                     'BANKFULL_DEPTH' : names.BANKFULL_DEPTH,
                     'BANKFULL_XSEC_AREA' : names.BANKFULL_AREA,
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
               'BANKFULL_WIDTH' : names.BANKFULL_WIDTH,
               'BANKFULL_DEPTH' : names.BANKFULL_DEPTH,
               'BANKFULL_XSEC_AREA' : names.BANKFULL_AREA,
              }


def _tryRename(df, old, new):
    try:
        df[new] = df.pop(old)
        return new
    except KeyError:
        return None


class ManagerNHD(manager_hyriver.ManagerHyRiver):
    """Leverages pynhd to download NHD data and its supporting shapes."""
    lowest_level = 12

    # Maps product_short → (product, ids, renames, default_layer, catchment_layer)
    _PRODUCT_ATTRS = {
        'NHDPlusMR': ('NHDPlus MR v2.1', waterdata_ids, waterdata_renames,
                      'nhdflowline_network', 'catchmentsp'),
        'NHDPlusHR': ('NHDPlus HR',       hr_ids,        hr_renames,
                      'flowline',          'catchment'),
        'NHD':       ('NHD MR',           mr_ids,        mr_renames,
                      'flowline_mr',       None),
    }

    def __init__(self,
                 product: str,
                 variable: Optional[str] = None,
                 catchments: Optional[bool] = True,
                 fewer_columns: Optional[bool] = False,
                 ):
        """Initialize NHD manager.

        Parameters
        ----------
        product : str, optional
            NHD product name: ``'NHDPlus HR'`` (default), ``'NHDPlus MR v2.1'``,
            or ``'NHD MR'``.
        variable : str, optional
            Layer/variable name.  Defaults to the product's primary flowline layer.
        catchments : bool, optional
            Whether to fetch catchments alongside flowlines.  Default is ``True``.
        fewer_columns : bool, optional
            Whether to drop QA/QC columns from the returned GeoDataFrame.
            Default is ``False``.
        """
        # Find product_short from product name
        product_short = next(
            (k for k, v in self._PRODUCT_ATTRS.items() if v[0] == product), None
        )
        if product_short is None:
            raise ValueError(
                f'Invalid ManagerNHD product "{product}". '
                f'Valid options: {[v[0] for v in self._PRODUCT_ATTRS.values()]}'
            )

        _, self._ids, self._renames, default_variable, catchment_layer = \
            self._PRODUCT_ATTRS[product_short]

        if variable is None:
            variable = default_variable
        self._catchment_layer = catchment_layer if variable == default_variable else None
        self._fewer_columns = fewer_columns

        id_name = self._ids.get(variable, variable)

        from .manager import ManagerAttributes
        attrs = ManagerAttributes(
            category='geometry',
            product=product,
            product_short=product_short,
            source='hyriver',
            url='https://www.usgs.gov/national-hydrography',
            license='public domain',
            citation='USGS NHD',
            description='National Hydrography Dataset river reaches and catchments.',
            native_crs_in=watershed_workflow.crs.latlon_crs,
            native_resolution=0.001,
            native_id_field=id_name,
            valid_variables=[variable],
            default_variables=[variable],
        )
        super().__init__(attrs)
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
            old_layer = self.attrs.default_variables[0]
            old_id_name = self.attrs.native_id_field

            # Set catchment layer properties
            self.attrs.default_variables[0] = self._catchment_layer
            catchment_id_name = self._ids.get(self._catchment_layer, self._catchment_layer)
            self.attrs.native_id_field = catchment_id_name

            # Get catchments using HyRiver directly (no recursive catchment fetching)
            ids = df[old_id_name].tolist()
            cas_raw = manager_hyriver.ManagerHyRiver._getShapesByID(self, ids)

            # Apply standard names to catchments
            cas_raw = self._addStandardNames(cas_raw, False)

            # Merge catchments with flowlines
            df = pd.merge(df, cas_raw, how='outer', left_on=old_id_name,
                          right_on=catchment_id_name, suffixes=(None, '_ca'))

            # Restore original layer properties
            self.attrs.default_variables[0] = old_layer
            self.attrs.native_id_field = old_id_name

        return df

    def _addStandardNames(self,
                          df: gpd.GeoDataFrame,
                          fewer_columns : Optional[bool] = None,
                          ) -> gpd.GeoDataFrame:
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
        renames = []
        for k, v in self._renames.items():
            res = _tryRename(df, k, v)
            if res is not None:
                renames.append(res)

        # remove QA/QC codes
        if fewer_columns is None:
            fewer_columns = self._fewer_columns
        if fewer_columns:
            renames.extend(['geometry', names.ID, self.native_id_field])
            df = df[renames]

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

        # if NHDv2.1, get bankfull properties
        if self.attrs.product_short == 'NHDPlusMR':
            bankfull_df = pd.read_parquet(importlib.resources.files("watershed_workflow") / "data" / "nhd_v21_bankfull_properties.parquet")

            # merge on comid (lowercase in WaterData df, uppercase in bankfull file)
            df = pd.merge(df, bankfull_df, left_on='comid', right_on='COMID')
            df = df.drop(columns='COMID')
        
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
    

