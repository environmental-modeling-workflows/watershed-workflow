"""Manager for downloading 3DEP data."""

from typing import Tuple, Optional, List
import cftime

import shapely.geometry
import xarray as xr
import py3dep

import watershed_workflow.crs
from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_dataset import ManagerDataset

class Manager3DEP(ManagerDataset):
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
        resolution_in_degrees = resolution * 9e-6

        out_crs = CRS.from_epsg(5070)  # CONUS Albers Equal Area
        in_crs = CRS.from_epsg(4326)  # lat-long

        valid_variables = [
            'DEM', 'Hillshade Gray', 'Aspect Degrees', 'Aspect Map',
            'GreyHillshade_elevationFill', 'Hillshade Multidirectional', 
            'Slope Map', 'Slope Degrees', 'Hillshade Elevation Tinted',
            'Height Ellipsoidal', 'Contour 25', 'Contour Smoothed 25'
        ]
        default_variables = ['DEM']
        
        # Initialize base class with native properties
        super().__init__('3DEP', 'py3dep', resolution_in_degrees, in_crs, out_crs,
                         None, None, valid_variables, default_variables)

    def _requestDataset(self, request : ManagerDataset.Request) -> ManagerDataset.Request:
        """Request the data -- ready upon request."""
        request.is_ready = True
        return request


    def _fetchDataset(self, request : ManagerDataset.Request) -> xr.Dataset:
        """Implementation of abstract method to get 3DEP data."""

        # Base class ensures these for multi-variable, time independent class
        assert request.variables is not None
        assert request.start is None
        assert request.end is None
        
        # Use instance resolution and native CRS
        result = py3dep.get_map(request.variables, request.geometry, self._resolution, 
                               geo_crs=self.native_crs_in, crs=self.native_crs_out)
        
        # py3dep returns DataArray for single layer, Dataset for multiple layers
        if isinstance(result, xr.DataArray):
            # Convert DataArray to Dataset
            result = result.to_dataset(name=request.variables[0].lower().replace(' ', '_'))
            
        return result
        
