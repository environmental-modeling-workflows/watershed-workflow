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
from .manager_dataset_cached import in_memory_cached_manager
from .cache_info import CacheInfo


@in_memory_cached_manager(CacheInfo(
    category='elevation',
    subcategory='3dep',
    name='3dep',          # overridden per-instance in __init__ to encode resolution
    snap_resolution=0.001,
))
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

        # Override class-level CacheInfo to encode resolution in the cache dir name
        self._cache_info = CacheInfo(
            category='elevation',
            subcategory='3dep',
            name=f'3dep_{resolution}m',
            snap_resolution=0.001,
        )

    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request
                        ) -> manager_dataset.ManagerDataset.Request:
        """Return the request unchanged — no async step."""
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — py3dep is synchronous."""
        return True

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Fetch 3DEP data via py3dep and store on ``request._dataset``.

        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request with preprocessed geometry and variables.
        """
        assert request.variables is not None
        assert request.start is None
        assert request.end is None

        logging.info(f'Getting DEM with map of area = {request.geometry.area}')
        bounds = request.geometry.bounds
        bbox = shapely.geometry.box(*bounds)
        result = py3dep.get_map(request.variables, bbox, self._resolution,
                                geo_crs=self.native_crs_in, crs=self.native_crs_out)

        # py3dep returns DataArray for single layer, Dataset for multiple layers
        if isinstance(result, xr.DataArray):
            result = result.to_dataset(name=request.variables[0].lower().replace(' ', '_'))

        request._dataset = result

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Return the dataset stored on the request by ``_downloadDataset``."""
        return request._dataset
        
