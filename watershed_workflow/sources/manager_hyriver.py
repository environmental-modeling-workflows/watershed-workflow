from __future__ import annotations

import abc
from typing import List, Optional, Any
from shapely.geometry.base import BaseGeometry
import geopandas as gpd

from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_shapes import ManagerShapes
import watershed_workflow.sources.standard_names as names


class ManagerHyRiver(ManagerShapes):
    """A generic base class for working with HyRiver"""

    def __init__(self,
                 protocol_name: str,
                 native_crs_in: CRS,
                 native_resolution: float,
                 layer: str = '',
                 id_name: Optional[str] = None):
        """Initialize HyRiver manager.
        
        Parameters
        ----------
        protocol_name : str
            HyRiver protocol name (NHD, NHDPlusHR, WBD, WaterData).
        native_crs_in : CRS
            Expected CRS of incoming geometry for API queries.
        native_resolution : float
            Native resolution in native_crs_in units.
        layer : str, optional
            Layer name for the protocol.
        id_name : str, optional
            Name of the ID field, defaults to layer name.
        """
        self._layer = layer
        if id_name is None:
            id_name = layer
        self._id_name = id_name
        self._protocol_name = protocol_name
        self._protocol: Any = None

        # Set up protocol-specific API
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
        else:
            raise ValueError(f'Invalid HyRiver protocol "{protocol_name}"')

        # Create name and source for base class
        name = f'HyRiver {protocol_name}: {layer}' if layer else f'HyRiver {protocol_name}'
        source = f'HyRiver.{protocol_name}'

        # Initialize base class
        super().__init__(name, source, native_crs_in, native_resolution, id_name)

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
        df = self._protocol(self._layer).bygeom(union_geometry, self.native_crs_in)
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
        protocol = self._protocol(self._layer)
        if hasattr(protocol, 'byid'):
            df = protocol.byid(self._id_name, ids)
        else:
            df = protocol.byids(self._id_name, ids)
        return df
