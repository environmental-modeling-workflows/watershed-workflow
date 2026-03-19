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
from .manager import ManagerAttributes
from .manager_dataset_cached import in_memory_cached_manager


@in_memory_cached_manager
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

        attrs = ManagerAttributes(
            category='elevation',
            product=f'3DEP {resolution}m DEM',
            source='py3dep TNM',
            description='USGS 3D Elevation Program digital elevation model and derived products.',
            product_short=f'3DEP_{resolution}m',
            source_short='py3dep_tnm',
            url='https://www.usgs.gov/3d-elevation-program',
            license='public domain',
            citation='USGS 3DEP',
            native_crs_in=in_crs,
            native_crs_out=out_crs,
            native_resolution=resolution_in_degrees,
            valid_variables=valid_variables,
            default_variables=default_variables,
        )
        super().__init__(attrs)

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
