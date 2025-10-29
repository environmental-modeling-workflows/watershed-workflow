"""Manager for downloading 3DEP data."""

from typing import Tuple, Optional, List
import cftime
import logging

import shapely.geometry
import xarray as xr
import py3dep

import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset

class Manager3DEP(manager_dataset.ManagerDataset):
    """3D Elevation Program (3DEP) data manager.
    
    Provides access to USGS 3DEP elevation and derived products through
    the py3dep library. Supports multiple resolution options and various
    topographic layers including DEM, slope, aspect, and hillshade products.
    """
    
    def __init__(self, resolution : int):
        """Downloads DEM data from the 3DEP.

        Parameters
        ----------
        resolution : int
            Resolution in meters. Valid resolutions are: 60, 30, or 10.
        """
        self._resolution = resolution
        resolution_in_degrees = 2 * resolution * 9e-6

        in_crs = CRS.from_epsg(4326)  # lat-long
        out_crs = CRS.from_epsg(5070)  # CONUS Albers Equal Area

        valid_variables = [
            'DEM', 'Hillshade Gray', 'Aspect Degrees', 'Aspect Map',
            'GreyHillshade_elevationFill', 'Hillshade Multidirectional', 
            'Slope Map', 'Slope Degrees', 'Hillshade Elevation Tinted',
            'Height Ellipsoidal', 'Contour 25', 'Contour Smoothed 25'
        ]
        default_variables = ['DEM']
        
        # Initialize base class with native properties
        super().__init__(f'3DEP {resolution}m', 'py3dep', resolution_in_degrees, in_crs, out_crs,
                         None, None, valid_variables, default_variables)

    def _requestDataset(self, request : manager_dataset.ManagerDataset.Request
                        ) -> manager_dataset.ManagerDataset.Request:
        """Request the data -- ready upon request."""
        request.is_ready = True
        return request


    def _fetchDataset(self, request : manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Implementation of abstract method to get 3DEP data."""

        # Base class ensures these for multi-variable, time independent class
        assert request.variables is not None
        assert request.start is None
        assert request.end is None
        
        # Use instance resolution and native CRS
        logging.info(f'Getting DEM with map of area = {request.geometry.area}')
        result = py3dep.get_map(request.variables, request.geometry, self._resolution, 
                               geo_crs=self.native_crs_in, crs=self.native_crs_out)
        
        # py3dep returns DataArray for single layer, Dataset for multiple layers
        if isinstance(result, xr.DataArray):
            # Convert DataArray to Dataset
            result = result.to_dataset(name=request.variables[0].lower().replace(' ', '_'))
            
        return result
        
