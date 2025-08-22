"""Base class for managers that provide shapes.

Provides base classes for managers that fetch geospatial data as 
GeoDataFrames, ensuring consistent interfaces and standard naming
conventions across all shape-based data sources.

"""

import abc
from typing import Optional, List
import shapely.geometry
import geopandas as gpd
import logging

from watershed_workflow.crs import CRS
import watershed_workflow.crs
import watershed_workflow.warp
import watershed_workflow.sources.standard_names as names


class ManagerShapes(abc.ABC):
    """Managers that provide GeoDataFrames should inherit from this class.

    This class provides a consistent interface for fetching geospatial data
    from various sources, handling coordinate transformations, and ensuring
    standard column naming conventions.

    There are two main usage patterns:

    1. Get shapes by geometry:
       shapes = mgr.getShapesByGeometry(polygon, crs)
       shapes = mgr.getShapesByGeometry(geodataframe)

    2. Get shapes by ID:
       shapes = mgr.getShapesByID(['id1', 'id2', ...])

    Developer notes: derived classes must implement:

    - __init__() : Constructor that supplies native data properties as
      parameters by calling super().__init__()

    - gpd.GeoDataFrame _getShapesByGeometry(geometry) : 
      Abstract method that fetches shapes for the given geometry in native_crs_in

    - gpd.GeoDataFrame _getShapesByID(ids) : 
      Abstract method that fetches shapes by ID list

    - gpd.GeoDataFrame _addStandardNames(df) : 
      Abstract method that converts native column names to standard names
    """

    def __init__(self, 
                 name: str,
                 source: str,
                 native_crs_in: CRS,
                 native_resolution: float,
                 native_id_field: str):
        """Initialize shape manager with native data properties.
        
        Parameters
        ----------
        name : str
            Name of the shape manager.
        source : str
            Data source or API used to retrieve the shapes.
        native_crs_in : CRS
            Expected CRS of the incoming geometry for API queries.
        native_resolution : float
            Native resolution of the data in native_crs_in units, used for buffering.
        native_id_field : str
            Name of the ID field in the native data.
        """
        self.name = name
        self.source = source
        self.native_crs_in = native_crs_in
        self.native_resolution = native_resolution
        self.native_id_field = native_id_field

    def getShapesByGeometry(self,
                           geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
                           geometry_crs: Optional[CRS] = None) -> gpd.GeoDataFrame:
        """Get shapes that intersect with the given geometry.

        Parameters
        ----------
        geometry : shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame
            Input geometry to search for intersecting shapes. Can be a shapely
            geometry (requiring geometry_crs) or a GeoDataFrame.
        geometry_crs : CRS, optional
            Coordinate reference system of the input geometry. Required if
            geometry is a shapely geometry, ignored if geometry is GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing shapes that intersect the geometry,
            with standardized column names and ID indexing.
        """
        # Handle input geometry - create GeoDataFrame for derived class
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry_crs is not None:
                raise ValueError("geometry_crs should not be provided with GeoDataFrame")
            geometry_gdf = geometry
            filter_polygon = geometry.union_all()
            filter_crs = geometry.crs
        elif isinstance(geometry, shapely.geometry.base.BaseGeometry):
            if geometry_crs is None:
                raise ValueError("geometry_crs is required when geometry is a shapely geometry")
            geometry_gdf = gpd.GeoDataFrame([{}], geometry=[geometry], crs=geometry_crs)
            filter_polygon = geometry
            filter_crs = geometry_crs
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")
        
        # Transform to native CRS if needed
        if not watershed_workflow.crs.isEqual(geometry_gdf.crs, self.native_crs_in):
            geometry_gdf = geometry_gdf.to_crs(self.native_crs_in)
            filter_polygon = watershed_workflow.warp.shply(filter_polygon, filter_crs, self.native_crs_in)
            filter_crs = self.native_crs_in
        
        # Get raw shapes from derived class
        df = self._getShapesByGeometry(geometry_gdf)
        
        # Add standard names (derived class specific)
        df = self._addStandardNames(df)
        
        # Postprocess using filter geometry
        df = self._postprocessShapes(df, filter_polygon, filter_crs)
        
        return df

    def getShapesByID(self,
                     ids: List[str] | str) -> gpd.GeoDataFrame:
        """Get shapes by their ID values.

        Parameters
        ----------
        ids : List[str] | str
            Single ID or list of IDs to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the requested shapes,
            with standardized column names and ID indexing.
        """
        # Ensure ids is a list
        if isinstance(ids, str):
            ids = [ids]
        
        # Get raw shapes from derived class
        df = self._getShapesByID(ids)
        
        # Add standard names (derived class specific)
        df = self._addStandardNames(df)
        
        # Postprocess (base class standard operations)
        df = self._postprocessShapes(df)
        
        return df


    def _postprocessShapes(self,
                          df: gpd.GeoDataFrame,
                          filter_geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
                          filter_geometry_crs: Optional[CRS] = None) -> gpd.GeoDataFrame:
        """Apply standard postprocessing to shapes.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame from derived class with standard names already added.
        filter_geometry : shapely.geometry.base.BaseGeometry, optional
            Geometry to filter shapes by intersection. If provided,
            filter_geometry_crs must also be provided.
        filter_geometry_crs : CRS, optional
            CRS of filter_geometry.

        Returns
        -------
        gpd.GeoDataFrame
            Postprocessed GeoDataFrame with standard ID/name columns
            and proper indexing.
        """
        # Ensure CRS is defined (but don't coerce to specific CRS)
        if df.crs is None:
            raise ValueError(f"GeoDataFrame from {self.name} does not have CRS defined")
        
        # Assert that derived class provided standard ID column
        assert names.ID in df.columns, f"Derived class {self.__class__.__name__} must provide {names.ID} column in _addStandardNames()"
        
        # Ensure standard name column exists
        if names.NAME not in df.columns:
            # Generate names from ID if not provided
            df[names.NAME] = df[names.ID].astype(str)
        
        # Set ID as index
        df = df.set_index(names.ID, drop=False)
        
        # Filter by geometry intersection if requested (using unbuffered geometry)
        if filter_geometry is not None and filter_geometry_crs is not None:
            # Transform filter geometry to same CRS as shapes
            if not watershed_workflow.crs.isEqual(filter_geometry_crs, df.crs):
                filter_geometry = watershed_workflow.warp.shply(
                    filter_geometry, filter_geometry_crs, df.crs)
            
            # Filter to only intersecting shapes
            df = df[df.intersects(filter_geometry)]
        
        # Add metadata to attributes
        df.attrs['name'] = self.name
        df.attrs['source'] = self.source
        
        return df

    @abc.abstractmethod
    def _getShapesByGeometry(self,
                            geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch shapes for the given geometry.

        This method should be implemented by derived classes to fetch
        shapes from their specific data source.

        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes (already buffered).

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        pass

    @abc.abstractmethod
    def _getShapesByID(self,
                      ids: List[str]) -> gpd.GeoDataFrame:
        """Fetch shapes by ID list.

        This method should be implemented by derived classes to fetch
        shapes by their ID values.

        Parameters
        ----------
        ids : List[str]
            List of IDs to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        pass

    @abc.abstractmethod
    def _addStandardNames(self,
                         df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert native column names to standard names.

        This method should be implemented by derived classes to map
        their native column names to the standard names defined in
        watershed_workflow.sources.standard_names.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame with native column names.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with standard column names added.
        """
        pass
