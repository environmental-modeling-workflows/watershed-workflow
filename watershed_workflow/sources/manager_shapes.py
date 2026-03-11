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
import watershed_workflow.utils.warp

from . import standard_names as names
from . import manager


class ManagerShapes(manager.Manager):
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
                 native_id_field: str,
                 cache_category: str | None = None,
                 cache_extension: str = 'shp',
                 short_name: str | None = None):
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
        cache_category : str or None, optional
            Top-level cache folder group, e.g. ``'soil_structure'``.  Pass
            ``None`` (default) to opt out of the standard cache system.
        cache_extension : str, optional
            File extension for cache files.  Default ``'shp'``.
        short_name : str, optional
            Short, filesystem-safe name used as the leaf cache directory and
            filename prefix (e.g. ``'NRCS'``).
        """
        super().__init__(
            name=name,
            source=source,
            native_crs_in=native_crs_in,
            native_resolution=native_resolution,
            cache_category=cache_category,
            cache_extension=cache_extension,
            has_varname=False,   # shapes managers never split by variable
            is_temporal=False,   # shapes managers are never temporal
            short_name=short_name,
        )
        self.native_id_field = native_id_field


    def getShapes(self,
                  out_crs : Optional[CRS] = None,
                  digits : Optional[int] = -1,
                  remove_third_dimension : Optional[bool] = True,
                  ):
        """Get all shapes in a manager."""
        self._prerequestDataset()
        df = self._getShapes()
        df = self._addStandardNames(df)
        df = self._postprocessShapes(df, out_crs=out_crs, digits=digits,
                                     remove_third_dimension=remove_third_dimension)
        return df


    def getShapesByGeometry(self,
                           geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
                           geometry_crs: Optional[CRS] = None,
                           out_crs : Optional[CRS] = None,
                           digits : Optional[int] = -1,
                           remove_third_dimension : Optional[bool] = True,
                            ) -> gpd.GeoDataFrame:
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
        self._prerequestDataset()

        # Normalise input geometry
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
            filter_polygon = watershed_workflow.utils.warp.warpShply(filter_polygon, filter_crs, self.native_crs_in)
            filter_crs = self.native_crs_in

        # Buffer for cache stability; keep filter_polygon un-buffered for clipping
        buffered_polygon = geometry_gdf.union_all().buffer(3 * self.native_resolution)
        snapped_bounds = self._snapBounds(buffered_polygon.bounds)
        logging.info(f'  Shapes manager: buffered+snapped query box = {snapped_bounds}')

        # Check for a cached superset before downloading
        superset = self._checkCache(
            geometry_bounds=buffered_polygon.bounds,
            snapped_bounds=snapped_bounds)
        if superset is not None:
            logging.info(f'  Using superset cache: {superset}')
            df = gpd.read_file(superset)
            df = self._addStandardNames(df)
            df = self._postprocessShapes(df, filter_polygon, filter_crs, out_crs, digits,
                                         remove_third_dimension)
            return df

        # Build query GeoDataFrame from snapped bounds
        query_polygon = shapely.geometry.box(*snapped_bounds)
        query_gdf = gpd.GeoDataFrame([{}], geometry=[query_polygon], crs=self.native_crs_in)

        df = self._getShapesByGeometry(query_gdf)
        df = self._addStandardNames(df)
        df = self._postprocessShapes(df, filter_polygon, filter_crs, out_crs, digits,
                                     remove_third_dimension)
        return df


    def getShapesByID(self,
                      ids: List[str] | str,
                      out_crs : Optional[CRS] = None,
                      digits : Optional[int] = -1,
                      remove_third_dimension : Optional[bool] = True,
                      ) -> gpd.GeoDataFrame:
        """Get shapes by their ID values.

        Parameters
        ----------
        ids : list of str | str
            Single ID or list of IDs to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the requested shapes,
            with standardized column names and ID indexing.
        """
        self._prerequestDataset()

        if isinstance(ids, str):
            ids = [ids]

        df = self._getShapesByID(ids)
        df = self._addStandardNames(df)
        df = self._postprocessShapes(df, out_crs=out_crs, digits=digits,
                                     remove_third_dimension=remove_third_dimension)
        return df


    def _postprocessShapes(self,
                          df: gpd.GeoDataFrame,
                          filter_geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
                          filter_geometry_crs: Optional[CRS] = None,
                          out_crs : Optional[CRS] = None,
                          digits : Optional[int] = -1,
                          remove_third_dimension : Optional[bool] = True,
                           ) -> gpd.GeoDataFrame:
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
        if df.crs is None:
            raise ValueError(f"GeoDataFrame from {self.name} does not have CRS defined")

        assert names.ID in df.columns, (
            f"Derived class {self.__class__.__name__} must provide {names.ID} "
            f"column in _addStandardNames()")

        if names.NAME not in df.columns:
            df[names.NAME] = df[names.ID].astype('string')

        # Filter by geometry intersection (using unbuffered geometry)
        if filter_geometry is not None and filter_geometry_crs is not None:
            if not watershed_workflow.crs.isEqual(filter_geometry_crs, df.crs):
                filter_geometry = watershed_workflow.utils.warp.warpShply(
                    filter_geometry, filter_geometry_crs, df.crs)
            df = df[df.intersects(filter_geometry)]

        # do things to ALL geometry columns
        orig_geometry = df.geometry.name
        for col in df.select_dtypes('geometry'):
            logging.info(f'fixing column: {col}')
            df = df.set_geometry(col)

            def _combine(shp):
                if isinstance(shp, shapely.geometry.MultiLineString):
                    return shapely.line_merge(shp)
                elif isinstance(shp, shapely.geometry.MultiPolygon):
                    valid_geoms = []
                    for g in shp.geoms:
                        if not g.is_valid:
                            g = shapely.make_valid(g)
                        if isinstance(g, shapely.geometry.MultiPolygon):
                            valid_geoms.extend(g.geoms)
                        else:
                            valid_geoms.append(g)
                    return shapely.union_all(shapely.geometry.MultiPolygon(valid_geoms))
                return shp
            df[col] = df[col].apply(_combine)

            if remove_third_dimension:
                is_none_mask = df[col].notna()
                df.loc[is_none_mask, col] = df.loc[is_none_mask, col].apply(
                    watershed_workflow.utils.removeThirdDimension)

            if out_crs is not None:
                df = df.to_crs(out_crs)

            if digits >= 0:
                df = df.set_precision(10**-digits)

        df = df.set_geometry(orig_geometry)

        df.attrs['name'] = self.name
        df.attrs['source'] = self.source

        if hasattr(df.index, 'name') and df.index.name == names.ID:
            df = df.reset_index()

        return df


    @abc.abstractmethod
    def _getShapes(self):
        """Fetch all shapes in a dataset.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        pass


    @abc.abstractmethod
    def _getShapesByGeometry(self,
                            geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch shapes for the given geometry.

        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes
            (already buffered and snapped).

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

        Parameters
        ----------
        ids : list of str
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
