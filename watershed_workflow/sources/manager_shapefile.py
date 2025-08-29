"""Basic manager for interacting with shapefiles.
"""
from typing import Optional, List
import os

import pyogrio
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
import watershed_workflow.utils
import watershed_workflow.crs
import watershed_workflow.warp
from watershed_workflow.crs import CRS

from . import manager_shapes
from . import standard_names as names

class ManagerShapefile(manager_shapes.ManagerShapes):
    """A simple class for reading shapefiles.

    Parameters
    ----------
    filename : str
      Path to the shapefile.
    id_name : str, optional
      Name of the ID field in the shapefile.
    """
    
    def __init__(self,
                 filename: str,
                 url : Optional[str] = None,
                 id_name: Optional[str] = None
                 ):
        """Initialize shapefile manager.
        
        Parameters
        ----------
        filename : str
            Path to the shapefile.
        url : str, optional
            URL from which to download the file.
        id_name : str, optional
            Name of the ID field in the shapefile.
        """
        self.filename = filename
        self.url = url
        self.id_name = id_name

        # flag to indicate that we have the file and we have processed
        # it for metadata
        self._file_preprocessed = False
        
        # Use basename of file as name
        name = f'shapefile: "{os.path.basename(filename)}"'

        # Use id_name or 'ID' as native_id_field
        native_id_field = id_name if id_name is not None else 'ID'

        if url is not None:
            # url is the source
            source = url
        else:
            # Use absolute path as source for complete provenance  
            source = os.path.abspath(filename)

        # Initialize base class
        super().__init__(name, source, None, None, native_id_field)

    def _prerequestDataset(self):
        # first download -- this is done here and not in _request so
        # that we can set the resolution and CRS for input geometry
        # manipulation.
        if not os.path.isfile(self.filename) and self.url is not None:
            self._download()

        if not self._file_preprocessed:
            # Get file info to determine native CRS
            info = pyogrio.read_info(self.filename)
            self.native_crs_in = watershed_workflow.crs.from_string(info['crs'])
        
            # Estimate resolution from bounds (simple heuristic)
            # Use 1/1000th of the smallest dimension as resolution estimate
            bounds = info['total_bounds']
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            self.native_resolution = min(width, height) / 1000.0

            # only do this work once
            self._file_preprocessed = True
        
        
    def _getShapes(self) -> gpd.GeoDataFrame:
        """Read the file and get all shapes.
        
        Returns
        -------
        gpd.GeoDataFrame
            All shapes from the shapefile.
        """
        return gpd.read_file(self.filename)

    def _getShapesByGeometry(self, geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch shapes for the given geometry.

        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        # Use bbox filtering - full intersection handled by base class
        union_geometry = geometry_gdf.union_all()
        df = gpd.read_file(self.filename, bbox=union_geometry.bounds)
        return df

    def _getShapesByID(self, ids: List[str]) -> gpd.GeoDataFrame:
        """Fetch shapes by ID list.

        Parameters
        ----------
        ids : List[str]
            List of IDs to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        if self.id_name is not None:
            # Read full file and filter by specified ID field
            df = gpd.read_file(self.filename)
            if self.id_name not in df.columns:
                raise ValueError(f"ID field '{self.id_name}' not found in shapefile columns: {list(df.columns)}")
                
            id_column = df[self.id_name]
            if len(id_column) > 0:
                target_type = type(id_column.iloc[0])
                try:
                    converted_ids = [target_type(id_val) for id_val in ids]
                    df = df[df[self.id_name].isin(converted_ids)]
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert IDs {ids} to type {target_type} for field '{self.id_name}': {e}")
        else:
            # No ID field specified - use row indices
            try:
                int_ids = [int(id_val) for id_val in ids]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert IDs {ids} to integers for row-based access: {e}")
            
            # Validate indices first
            info = pyogrio.read_info(self.filename)
            total_rows = info['features']
            valid_indices = [i for i in int_ids if 0 <= i < total_rows]
            
            if len(valid_indices) != len(int_ids):
                invalid_indices = [i for i in int_ids if i < 0 or i >= total_rows]
                raise ValueError(f"Invalid row indices {invalid_indices}. File has {total_rows} rows (0-{total_rows-1})")
            
            # Optimize for single row case
            if len(valid_indices) == 1:
                # Read just the single row using slice
                index = valid_indices[0]
                df = gpd.read_file(self.filename, rows=slice(index, index + 1))
            else:
                # Read full file and select specific rows
                # Note: gpd.read_file(rows=list) is not supported, so we read all and filter
                df = gpd.read_file(self.filename)
                df = df.iloc[valid_indices]
            
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
        # Map ID field if it exists, otherwise create row-based IDs
        if self.id_name is not None and self.id_name in df.columns:
            df[names.ID] = df[self.id_name]
        else:
            # For row-based access or when ID field doesn't exist, use row indices
            df[names.ID] = range(len(df))
        
        # No other standard name mappings for generic shapefiles
        return df
    
        
