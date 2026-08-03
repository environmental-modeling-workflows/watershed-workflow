"""Basic manager for interacting with raster files."""
from typing import Optional, Iterable

import os
import xarray as xr
import rioxarray
import logging

import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset
from .manager import ManagerAttributes
from . import utils as source_utils


class ManagerRaster(manager_dataset.ManagerDataset):
    """A simple manager for reading local raster files.

    The file must exist on disk at construction time.  Subclasses may override
    ``_downloadDataset`` to fetch or validate the file before it is read.

    Parameters
    ----------
    filename : str
        Path to the raster file.  Must exist on disk.
    native_resolution : float, optional
        Resolution of the raster. Detected automatically if not provided.
    native_crs : CRS, optional
        CRS of the raster. Detected automatically if not provided.
    bands : iterable of str or int, optional
        Band names or number of bands. Detected automatically if not provided.
    """

    def __init__(self,
                 filename: str,
                 native_resolution: Optional[float] = None,
                 native_crs: Optional[CRS] = None,
                 bands: Optional[Iterable[str] | int] = None,
                 attrs: Optional['ManagerAttributes'] = None,
                 ):
        self.filename = filename

        # If any native properties were not provided, inspect the file now.
        # The file must exist for auto-detection; if it doesn't and all
        # properties were provided explicitly this is still fine.
        if native_crs is None or native_resolution is None or bands is None:
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f'Raster file not found: {filename}\n'
                    f'Either provide native_crs, native_resolution, and bands '
                    f'explicitly, or ensure the file exists at construction time.'
                )
            with rioxarray.open_rasterio(self.filename) as ds:
                if native_crs is None:
                    native_crs = watershed_workflow.crs.from_rasterio(ds.rio.crs)

                if native_resolution is None:
                    if len(ds.coords['x']) > 1 and len(ds.coords['y']) > 1:
                        x_res = abs(float(ds.coords['x'][1] - ds.coords['x'][0]))
                        y_res = abs(float(ds.coords['y'][1] - ds.coords['y'][0]))
                        native_resolution = max(x_res, y_res)
                    else:
                        native_resolution = 1.0

                if bands is None:
                    if hasattr(ds, 'band'):
                        bands = [f'band_{i}' for i in ds.band.values]
                    elif len(ds.values.shape) == 3:
                        bands = [f'band_{i}' for i in range(ds.values.shape[0])]

        if bands is None:
            valid_variables = default_variables = None
        elif isinstance(bands, int):
            valid_variables = [f'band_{i}' for i in range(bands)]
            default_variables = [valid_variables[0]]
        else:
            valid_variables = list(bands)
            default_variables = [valid_variables[0]]

        if attrs is None:
            attrs = ManagerAttributes(
                category='undefined',
                product='undefined',
                source=os.path.abspath(filename),
                description=f'Local raster file: {os.path.basename(filename)}',
                native_crs_in=native_crs,
                native_crs_out=native_crs,
                native_resolution=native_resolution,
                valid_variables=valid_variables,
                default_variables=default_variables,
            )
        else:
            # Fill in native properties derived from the file inspection.
            attrs.native_crs_in = native_crs
            attrs.native_crs_out = native_crs
            attrs.native_resolution = native_resolution
            attrs.valid_variables = valid_variables
            attrs.default_variables = default_variables
        super().__init__(attrs)

    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request
                        ) -> manager_dataset.ManagerDataset.Request:
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        return True

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        pass

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Open the raster and return as a Dataset."""
        if not self.filename.lower().endswith('.tif'):
            dataset = rioxarray.open_rasterio(self.filename, chunk='auto')
        else:
            dataset = rioxarray.open_rasterio(self.filename, cache=False)

        result_dataset = xr.Dataset()

        if request.variables is None:
            result_dataset['raster'] = dataset[0, :, :] if len(dataset.shape) > 2 else dataset
        else:
            for var in request.variables:
                assert var.startswith('band_')
                band_idx = int(var.split('_')[1]) - 1
                if len(dataset.shape) > 2 and band_idx < dataset.shape[0]:
                    result_dataset[var] = dataset[band_idx, :, :].drop_vars('band', errors='ignore')
                elif len(dataset.shape) == 2 and band_idx == 0:
                    result_dataset[var] = dataset
                else:
                    raise ValueError(f'Band {band_idx + 1} not available in raster')

        return result_dataset
