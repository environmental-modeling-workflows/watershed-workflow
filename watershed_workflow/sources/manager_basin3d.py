"""Manager for Basin3D monitoring features."""

from typing import List, Optional
import logging
import geopandas as gpd
import shapely.geometry
import pandas as pd
import numpy as np

import basin3d.synthesis as synthesis
from basin3d.core.schema.query import QueryMonitoringFeature
from basin3d.core.schema.enum import FeatureTypeEnum

from watershed_workflow.sources.manager_shapes import ManagerShapes
import watershed_workflow.crs
import watershed_workflow.sources.standard_names as names


class ManagerBasin3D(ManagerShapes):
    """Basin3D monitoring features manager for watershed workflow.
    
    Provides access to spatial features from Basin3D data synthesis framework.
    Supports querying by feature IDs and bounding box geometry. Feature types 
    and data sources configured at construction.
    
    Uses Basin3D's plugin system to access multiple environmental data sources
    including USGS, EPA, and ESS-DIVE.
    """

    def __init__(self, 
                 feature_types: Optional[List[str]] = None,
                 data_sources: Optional[List[str]] = None,
                 **plugin_configs):
        """Initialize Basin3D manager.
        
        Parameters
        ----------
        feature_types : List[str], optional
            Types of monitoring features to retrieve (e.g., ['point', 'site']).
            Default: ['point']
        data_sources : List[str], optional  
            Basin3D plugins to use (e.g., ['usgs', 'epa']).
            Default: ['usgs']
        **plugin_configs : dict
            Configuration parameters for Basin3D plugins.
            Currently only USGS plugin is supported (no auth required).
        """
        # Set defaults
        self.feature_types = feature_types or ['point']
        self.data_sources = data_sources or ['usgs']
        
        # Validate feature types
        valid_types = ['point', 'site', 'plot', 'region', 'subregion', 
                       'basin', 'subbasin', 'watershed', 'subwatershed']
        for ft in self.feature_types:
            if ft not in valid_types:
                raise ValueError(f"Invalid feature_type '{ft}'. Valid types: {valid_types}")
        
        # Build plugin class list
        self.plugin_classes = []
        if 'usgs' in self.data_sources:
            self.plugin_classes.append('basin3d.plugins.usgs.USGSDataSourcePlugin')
        if 'epa' in self.data_sources:
            self.plugin_classes.append('basin3d.plugins.epa.EPADataSourcePlugin')
        if 'essdive' in self.data_sources:
            # Note: ESS-DIVE plugin may require additional configuration
            self.plugin_classes.append('basin3d.plugins.essdive.EssdiveDataSourcePlugin')
        
        if len(self.plugin_classes) == 0:
            raise ValueError(f"No valid plugins found for data_sources: {self.data_sources}")
        
        # Register Basin3D synthesizer
        try:
            self.synthesizer = synthesis.register(self.plugin_classes)
            logging.info(f"Basin3D registered with plugins: {[ds for ds in self.data_sources]}")
        except Exception as e:
            raise RuntimeError(f"Failed to register Basin3D plugins: {e}")
        
        # Initialize parent class
        super().__init__(
            name=f"Basin3D-{'-'.join(self.data_sources)}",
            source=f"Basin3D with plugins: {', '.join(self.data_sources)}",
            native_crs_in=watershed_workflow.crs.from_epsg(4326),  # Basin3D uses WGS84
            native_resolution=0.001,  # Approximate degrees for point data
            native_id_field='id'
        )

        
    def _getShapesByGeometry(self, geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Query Basin3D monitoring features by bounding box.
        
        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes.
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing monitoring features within bounds.
        """
        # Convert GeoDataFrame to bounding box
        total_bounds = geometry_gdf.total_bounds  # (xmin, ymin, xmax, ymax)
        print(total_bounds)
        bbox_tuple = tuple(np.round(b, 4) for b in total_bounds)
        
        # Query each configured feature type
        all_features = []
        for feature_type in self.feature_types:
            # Execute query
            logging.debug(f"executing query for feature-type {feature_type} on box {bbox_tuple}")
            result = self.synthesizer.monitoring_features(feature_type=feature_type,
                                                          monitoring_feature=[bbox_tuple,])
                
            # Collect features
            all_features.extend(result)
                
            logging.info(f"Found {len(all_features)} {feature_type} features from Basin3D")
        
        # Convert to GeoDataFrame
        logging.info(f'Found {len(all_features)} features')
        return self._convertToGeoDataFrame(all_features)

    def _getShapesByID(self, ids: List[str]) -> gpd.GeoDataFrame:
        """Query Basin3D monitoring features by specific IDs.
        
        Note: Basin3D ID queries have complex behavior and may not work
        as expected for all plugins. This implementation attempts several
        approaches.
        
        Parameters
        ----------
        ids : List[str]
            Feature ID(s) to retrieve.
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the requested features.
        """
        
        all_features = []
        
        # Try different query approaches for ID-based access
        for feature_type in self.feature_types:
            result = self.synthesizer.monitoring_features(feature_type=feature_type,
                                                          monitoring_feature=ids)

            all_features.extend(result)
        
        # Convert to GeoDataFrame
        return self._convertToGeoDataFrame(all_features)

    
    def _addStandardNames(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert native Basin3D column names to standard names.
        
        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame with Basin3D native column names.
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with standard column names added.
        """
        # Map Basin3D names to standard names where they exist
        if 'id' in df.columns and names.ID not in df.columns:
            df[names.ID] = df['id']
        if 'name' in df.columns and names.NAME not in df.columns:
            df[names.NAME] = df['name']
        return df

    
    def _convertToGeoDataFrame(self, basin3d_features):
        """Convert Basin3D MonitoringFeature objects to GeoDataFrame.
        
        Parameters
        ----------
        basin3d_features : List[MonitoringFeature]
            List of Basin3D monitoring features.
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with Basin3D native column names and proper CRS.
        """
        if not basin3d_features:
            # Return empty GeoDataFrame with expected schema
            return gpd.GeoDataFrame(
                columns=['id', 'name', 'feature_type', 'description', 'data_source', 'elevation'],
                geometry=[], 
                crs='EPSG:4326'
            )
        

        records = []
        geometries = []
        
        for feature in basin3d_features:
            try:
                # Extract coordinates
                coords = self._extractCoordinates(feature)
                if coords is None:
                    logging.warning(f"Could not extract coordinates for feature {feature.id}")
                    continue
                
                lon, lat, elevation = coords
                
                # Create shapely geometry
                geom = shapely.geometry.Point(lon, lat)
                geometries.append(geom)
                
                # Extract attributes
                record = {
                    'id': feature.id,
                    'name': getattr(feature, 'name', ''),
                    'feature_type': getattr(feature, 'feature_type', ''),
                    'description': getattr(feature, 'description', ''),
                    'data_source': getattr(feature, 'datasource', ''),
                    'elevation': elevation
                }
                records.append(record)
                
            except Exception as e:
                logging.warning(f"Failed to convert feature {getattr(feature, 'id', 'unknown')}: {e}")
        
        if not records:
            # Return empty GeoDataFrame
            return gpd.GeoDataFrame(
                columns=['id', 'name', 'feature_type', 'description', 'data_source', 'elevation'],
                geometry=[], 
                crs='EPSG:4326'
            )
        
        # Create GeoDataFrame (base class will handle CRS transformations)
        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs='EPSG:4326')
        return gdf

    
    def _extractCoordinates(self, feature):
        """Extract longitude, latitude, and elevation from Basin3D feature.
        
        Parameters
        ----------
        feature : MonitoringFeature
            Basin3D monitoring feature object.
            
        Returns
        -------
        tuple or None
            (longitude, latitude, elevation) or None if extraction fails.
        """
        try:
            if not feature.coordinates:
                return None
            
            abs_coord = feature.coordinates.absolute
            if not abs_coord:
                return None
            
            # Extract horizontal position (longitude, latitude)
            if abs_coord.horizontal_position:
                h_pos = abs_coord.horizontal_position[0]  # Get first position
                longitude = h_pos.longitude
                latitude = h_pos.latitude
            else:
                return None
            
            # Extract elevation (optional)
            elevation = None
            if abs_coord.vertical_extent:
                v_ext = abs_coord.vertical_extent[0]  # Get first extent
                elevation = getattr(v_ext, 'value', None)
            
            return (longitude, latitude, elevation)
            
        except Exception as e:
            logging.warning(f"Failed to extract coordinates: {e}")
            return None
