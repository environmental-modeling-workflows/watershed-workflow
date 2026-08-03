"""Manager for downloading SoilGrids v2.0 (250 m) products via the ISRIC WCS.

Provides access to the SoilGrids 2.0 gridded soil properties at 250 m resolution
for any bounding box via the ISRIC OGC Web Coverage Service (WCS).  Data are
returned as a 3-D ``xr.Dataset`` with a ``depth`` coordinate whose values are the
centre depths (in metres) of the six GlobalSoilMap standard layers.

No authentication is required.  Data are licensed CC-BY 4.0.

References: [SoilGrids2]_ [Poggio2021]_

.. [SoilGrids2] https://www.isric.org/explore/soilgrids
.. [Poggio2021] Poggio, L., et al. (2021). SoilGrids 2.0: producing soil information
   for the globe with quantified spatial uncertainty. *SOIL*, 7, 217–240.
   https://doi.org/10.5194/soil-7-217-2021
"""
from typing import List, Optional

import io
import os
import logging
import numpy as np
import xarray as xr
import rasterio

import watershed_workflow.crs
import watershed_workflow.properties.soil
from watershed_workflow.crs import CRS

from . import manager_dataset
from .manager import ManagerAttributes
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import snapBounds


# SoilGrids 2.0 is delivered on the Homolosine grid (EPSG:152160 = IGH).
# We request data in EPSG:4326 from the WCS so coordinates come back in
# familiar lat/lon, consistent with native_crs_in.
_WCS_BASE = 'https://maps.isric.org/mapserv?map=/map/{variable}.map'

# Six GlobalSoilMap depth intervals and their representative centre depths [m].
_DEPTH_LABELS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
_DEPTH_CENTRES = [0.025, 0.10, 0.225, 0.45, 0.80, 1.50]

# Scale divisors to convert the WCS INT16 values to conventional units.
_SCALE = {
    'bdod':     100.0,   # cg/cm³  → kg/dm³  (= g/cm³)
    'clay':      10.0,   # g/kg    → %
    'sand':      10.0,
    'silt':      10.0,
    'cfvo':      10.0,   # cm³/dm³ → %
    'soc':       10.0,   # dg/kg   → g/kg
    'nitrogen': 100.0,   # cg/kg   → g/kg
    'phh2o':    10.0,    # pH×10   → pH
    'ocd':      10.0,    # hg/m³   → kg/m³
}

_UNITS = {
    'bdod':     'kg/dm³',
    'clay':     '%',
    'sand':     '%',
    'silt':     '%',
    'cfvo':     '%',
    'soc':      'g/kg',
    'nitrogen': 'g/kg',
    'phh2o':    'pH',
    'ocd':      'kg/m³',
}

_LONG_NAMES = {
    'bdod':     'Bulk density of the fine earth fraction',
    'clay':     'Clay content',
    'sand':     'Sand content',
    'silt':     'Silt content',
    'cfvo':     'Coarse fragments volumetric',
    'soc':      'Soil organic carbon concentration',
    'nitrogen': 'Total nitrogen',
    'phh2o':    'Soil pH in water',
    'ocd':      'Organic carbon density',
}


@cached_dataset_manager
class ManagerSoilGrids(manager_dataset.ManagerDataset):
    """SoilGrids 2.0 (250 m) soil property manager.

    Downloads soil properties from the ISRIC SoilGrids v2.0 product [SoilGrids2]_
    via the OGC WCS endpoint [Poggio2021]_.  Each requested variable is returned
    as a 3-D array with dimensions ``(depth, y, x)`` where ``depth`` holds the
    centre depth in metres of each of the six GlobalSoilMap standard layers
    (0–5, 5–15, 15–30, 30–60, 60–100, 100–200 cm).

    Available variables
    -------------------
    bdod, clay, sand, silt, cfvo, soc, nitrogen, phh2o, ocd

    Default variables: ``clay``, ``sand``, ``silt``, ``bdod``
    (the four needed to run Rosetta for van Genuchten parameters).

    .. [SoilGrids2] https://www.isric.org/explore/soilgrids
    .. [Poggio2021] Poggio, L., et al. (2021). SoilGrids 2.0: producing soil
       information for the globe with quantified spatial uncertainty. *SOIL*,
       7, 217–240. https://doi.org/10.5194/soil-7-217-2021
    """

    VALID_VARIABLES = list(_LONG_NAMES.keys())
    DEFAULT_VARIABLES = ['clay', 'sand', 'silt', 'bdod']

    def __init__(self, force_download: bool = False):
        """Initialize the SoilGrids 2.0 manager.

        Parameters
        ----------
        force_download : bool, optional
            Re-download data even when a valid cached file already exists.
        """
        attrs = ManagerAttributes(
            category='soil_structure',
            product='SoilGrids 2.0',
            source='ISRIC WCS',
            description='SoilGrids 2.0 gridded soil properties at 250 m resolution via ISRIC WCS.',
            product_short='soilgrids',
            source_short='isric_wcs',
            url='https://maps.isric.org',
            license='CC BY 4.0',
            citation='Poggio et al. 2021',
            native_crs_in=CRS.from_epsg(4326),
            native_crs_out=CRS.from_epsg(4326),
            native_resolution=0.002,            # ~250 m in degrees
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
        )
        super().__init__(attrs)
        self.force_download = force_download

    def isComplete(self, dir: str, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True if all per-variable NetCDF files exist in the cache directory.

        Parameters
        ----------
        dir : str
            Absolute path to a candidate cache directory.
        request : ManagerDataset.Request
            The request being fulfilled.

        Returns
        -------
        bool
            True if ``{var}.nc`` exists for every requested variable.
        """
        for var in request.variables:
            if not os.path.isfile(os.path.join(dir, f'{var}.nc')):
                return False
        return True

    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        ) -> manager_dataset.ManagerDataset.Request:
        """Return the request unchanged — no async step.

        Parameters
        ----------
        request : ManagerDataset.Request
            Pre-processed request with geometry and variables.

        Returns
        -------
        ManagerDataset.Request
            The same request, unchanged.
        """
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — SoilGrids WCS is synchronous."""
        return True

    def _downloadDataset(self,
                         request: manager_dataset.ManagerDataset.Request,
                         ) -> None:
        """Download each requested variable to the cache directory.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set. Files are written to
            ``request._download_path/{var}.nc`` for each variable.
        """
        snapped_bounds = snapBounds(request.geometry.bounds, self.native_resolution)
        for var in request.variables:
            fname = os.path.join(request._download_path, f'{var}.nc')
            self._download(var, snapped_bounds, fname)

    def _download(self, var: str, snapped_bounds: tuple, filename: str) -> None:
        """Download all depth layers for one variable and save as NetCDF.

        Parameters
        ----------
        var : str
            Variable name (e.g. ``'clay'``).
        snapped_bounds : tuple of float
            ``(xmin, ymin, xmax, ymax)`` in WGS84 (EPSG:4326).
        filename : str
            Destination NetCDF path.
        """
        from owslib.wcs import WebCoverageService

        if os.path.exists(filename) and not self.force_download:
            logging.info(f'  Using existing: {filename}')
            return

        xmin, ymin, xmax, ymax = snapped_bounds
        url = _WCS_BASE.format(variable=var)
        logging.info(f'  Connecting to WCS: {url}')
        wcs = WebCoverageService(url, version='1.0.0')

        layers = []
        for depth_label in _DEPTH_LABELS:
            coverage_id = f'{var}_{depth_label}_mean'
            logging.info(f'    Downloading {coverage_id}')
            resp = wcs.getCoverage(
                identifier=coverage_id,
                crs='urn:ogc:def:crs:EPSG::4326',
                bbox=(xmin, ymin, xmax, ymax),
                resx=self.native_resolution,
                resy=self.native_resolution,
                format='GEOTIFF_INT16',
            )
            raw = resp.read()
            with rasterio.open(io.BytesIO(raw)) as src:
                arr = src.read(1).astype(np.float32)
                nodata = src.nodata
                transform = src.transform
                height, width = arr.shape

            # Replace nodata with NaN
            if nodata is not None:
                arr[arr == nodata] = np.nan

            # Apply scale factor
            arr /= _SCALE[var]

            layers.append(arr)

        # Build coordinate arrays from the affine transform of the last tile
        # (all depths share the same grid for a given variable + bounds).
        xs = transform.c + transform.a * (np.arange(width) + 0.5)
        ys = transform.f + transform.e * (np.arange(height) + 0.5)

        depth_coords = np.array(_DEPTH_CENTRES, dtype=np.float32)
        data = np.stack(layers, axis=0)   # (depth, y, x)

        ds = xr.Dataset({
            var: xr.DataArray(
                data,
                dims=['depth', 'y', 'x'],
                coords={'depth': depth_coords, 'y': ys, 'x': xs},
                attrs={
                    'long_name': _LONG_NAMES[var],
                    'units': _UNITS[var],
                    'depth_labels': ', '.join(_DEPTH_LABELS),
                },
            )
        })
        ds = ds.rio.write_crs(self.native_crs_out)
        ds.attrs['source'] = self.source
        ds.attrs['variable'] = var

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ds.to_netcdf(filename)
        logging.info(f'    Written to: {filename}')

    def _loadDataset(self,
                     request: manager_dataset.ManagerDataset.Request,
                     ) -> xr.Dataset:
        """Open cached NetCDF files, merge, and apply Rosetta if texture vars are present.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request with ``_download_path`` set.

        Returns
        -------
        xr.Dataset
            Merged dataset with all requested variables, each shaped
            ``(depth, y, x)``.  If ``clay``, ``sand``, ``silt``, and ``bdod``
            are all present, five additional van Genuchten variables are appended
            via Rosetta v3: ``residual saturation [-]``, ``porosity [-]``,
            ``van Genuchten alpha [Pa^-1]``, ``van Genuchten n [-]``,
            ``permeability [m^2]``.
        """
        datasets = []
        for var in request.variables:
            path = os.path.join(request._download_path, f'{var}.nc')
            datasets.append(xr.open_dataset(path))
        ds = xr.merge(datasets, compat='override')

        rosetta_inputs = {'clay', 'sand', 'silt', 'bdod'}
        if rosetta_inputs.issubset(set(ds.data_vars)):
            logging.info('  Running Rosetta on SoilGrids texture rasters')
            ds = watershed_workflow.properties.soil.computeVanGenuchtenModelFromRasters(ds)

        return ds
