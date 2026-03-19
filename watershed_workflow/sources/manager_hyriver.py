from __future__ import annotations

import abc
from typing import List, Optional, Any
from shapely.geometry.base import BaseGeometry
import geopandas as gpd

from watershed_workflow.sources.manager_shapes import ManagerShapes
from watershed_workflow.sources.manager import ManagerAttributes
import watershed_workflow.sources.standard_names as names


class ManagerHyRiver(ManagerShapes):
    """A generic base class for working with HyRiver"""

    # Maps product_short → (HyRiver protocol class loader, source, source_short)
    _PRODUCTS = {
        'NHD':        ('NHD',        'pynhd NHD',        'pynhd_nhd'),
        'NHDPlusHR':  ('NHDPlusHR',  'pynhd NHDPlusHR',  'pynhd_nhdplushr'),
        'NHDPlusMR':  ('WaterData',  'pynhd WaterData',  'pynhd_waterdata'),
        'WBD':        ('WBD',        'pygeohydro WBD',   'pygeohydro_wbd'),
        'WaterData':  ('WaterData',  'pynhd WaterData',  'pynhd_waterdata'),
    }

    def __init__(self, attrs: ManagerAttributes):
        """Initialize HyRiver manager.

        Parameters
        ----------
        attrs : ManagerAttributes
            Metadata object.  attrs.product_short must be one of: NHD,
            NHDPlusHR, NHDPlusMR, WBD, WaterData.  source and source_short
            are set automatically from product_short.  Must also include
            native_crs_in, native_resolution, native_id_field,
            valid_variables, and default_variables.
        """
        assert attrs.default_variables is not None and len(attrs.default_variables) == 1, \
            f'ManagerHyRiver requires exactly one default variable, got: {attrs.default_variables}'

        product_short = attrs.product_short
        if product_short not in self._PRODUCTS:
            raise ValueError(f'Invalid HyRiver product_short "{product_short}". '
                             f'Valid options: {list(self._PRODUCTS)}')

        protocol_name, source, source_short = self._PRODUCTS[product_short]
        attrs.source = source
        attrs.source_short = source_short

        self._protocol: Any = None

        if protocol_name == 'NHD':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.NHD
        elif protocol_name == 'NHDPlusHR':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.NHDPlusHR
        elif protocol_name == 'WBD':
            import pygeohydro.watershed
            self._protocol = pygeohydro.watershed.WBD
        elif protocol_name == 'WaterData':
            import pynhd.pynhd
            self._protocol = pynhd.pynhd.WaterData

        super().__init__(attrs)

    def _getShapes(self):
        """Fetch all shapes in a dataset.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        raise NotImplementedError(f'Manager source {self.source} does not support getting all shapes.')


    def _getShapesByGeometry(self, geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch shapes for the given geometry using HyRiver API.

        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        # HyRiver APIs take the union of geometries in the GeoDataFrame
        union_geometry = geometry_gdf.union_all()
        df = self._protocol(self.default_variables[0]).bygeom(union_geometry, self.native_crs_in)
        return df

    def _getShapesByID(self, ids: List[str]) -> gpd.GeoDataFrame:
        """Fetch shapes by ID list using HyRiver API.

        Parameters
        ----------
        ids : List[str]
            List of IDs to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        protocol = self._protocol(self.default_variables[0])
        if hasattr(protocol, 'byid'):
            df = protocol.byid(self.attrs.native_id_field, ids)
        else:
            df = protocol.byids(self.attrs.native_id_field, ids)
        return df
