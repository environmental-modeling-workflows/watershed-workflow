"""Basic manager for interacting with raster files.
"""
from typing import Tuple, List, Optional, Iterable

import os
import xarray as xr
import shapely
import rioxarray
import cftime
import logging

import watershed_workflow.crs
from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_dataset import ManagerDataset


class ManagerRaster(ManagerDataset):
    """A simple class for reading rasters."""

    def __init__(self,
                 filename : str,
                 url : Optional[str] = None,
                 native_resolution : Optional[float] = None,
                 native_crs : Optional[CRS] = None,
                 bands : Optional[Iterable[str] | int] = None,
                 ):
        """Initialize raster manager.
        
        Parameters
        ----------
        filename : str
            Path to the raster file.
        """
        self.filename = filename
        self.url = url
        
        # Use basename of file as name
        name = f'raster: "{os.path.basename(filename)}"'
        
        # Use absolute path as source for complete provenance
        source = os.path.abspath(filename)

        if bands is None:
            valid_variables = None
            default_variables = None
        elif isinstance(bands, int):
            valid_variables = [f'band {i+1}' for i in range(num_bands)]
            default_variables = [valid_variables[0],]
        else:
            valid_variables = bands
            default_variables = [valid_variables[0],]

        # Initialize base class
        super().__init__(
            name, source,
            native_resolution, native_crs, native_crs,
            None, None, valid_variables, default_variables
        )


    def _prerequestDataset(self) -> None:
        # first download -- this is done here and not in _request so
        # that we can set the resolution and CRS for input geometry
        # manipulation.
        if not os.path.isfile(self.filename) and self.url is not None:
            self._download()

        # Inspect raster to get native properties
        with rioxarray.open_rasterio(self.filename) as temp_ds:
            # Get native CRS
            self.native_crs_in = temp_ds.rio.crs
            self.native_crs_out = temp_ds.rio.crs
            
            # Get native resolution (approximate from first pixel)
            if len(temp_ds.coords['x']) > 1 and len(temp_ds.coords['y']) > 1:
                x_res = abs(float(temp_ds.coords['x'][1] - temp_ds.coords['x'][0]))
                y_res = abs(float(temp_ds.coords['y'][1] - temp_ds.coords['y'][0]))
                self.native_resolution = max(x_res, y_res)
            else:
                self.native_resolution = 1.0  # fallback
            
            # Create variable names for each band
            if self.valid_variables is None:
                if hasattr(temp_ds, 'band'):
                    # pull from bands
                    self.valid_variables = [f'band_{i}' for i in temp_ds.band.values]

                    # First band as default
                    self.default_variables = [self.valid_variables[0],]
                elif len(d.values.shape) == 3:
                    num_bands = d.values.shape[0]
                    self.valid_variables = [f'band_{i}' for i in range(num_bands)]
                    self.default_variables = [self.valid_variables[0],]


    def _requestDataset(self, request : ManagerDataset.Request) -> ManagerDataset.Request:
        """Request the data -- ready upon request."""
        request.is_ready = True
        return request


    def _fetchDataset(self, request : ManagerDataset.Request) -> xr.Dataset:
        """Fetch the data."""
        bounds = request.geometry.bounds
        
        # Open raster and clip to bounds
        if not self.filename.lower().endswith('.tif'):
            dataset = rioxarray.open_rasterio(self.filename, chunk='auto')
        else:
            dataset = rioxarray.open_rasterio(self.filename, cache=False)
            
        # Clip to bounds
        dataset = dataset.rio.clip_box(*bounds, crs=watershed_workflow.crs.to_rasterio(self.native_crs_out))
        
        # Convert to Dataset with band variables
        result_dataset = xr.Dataset()
        
        if request.variables is None:
            # single-variable case
            if len(dataset.shape) > 2:
                result_dataset['raster'] = dataset[0, :, :]  # Take first band
            else:
                result_dataset['raster'] = dataset

        else:
            for var in request.variables:
                assert var.startswith('band_')
                band_idx = int(var.split('_')[1]) - 1  # Convert to 0-indexed
                if len(dataset.shape) > 2 and band_idx < dataset.shape[0]:
                    band_data = dataset[band_idx, :, :]
                    band_data = band_data.drop_vars('band', errors='ignore')
                    result_dataset[var] = band_data
                elif len(dataset.shape) == 2:  # Single band raster
                    if band_idx == 0:
                        result_dataset[var] = dataset
                else:
                    raise ValueError(f"Band {band_idx + 1} not available in raster")

        return result_dataset

  
    def _download(self, force : bool = False):
        """A default download implementation."""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if not os.path.exists(self.filename) or force:
            source_utils.download(self.url, self.filename, force)
        
        
