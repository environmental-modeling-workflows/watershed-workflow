"""Manager for interacting with AORC datasets."""
from typing import List, Optional

import os
import xarray as xr
import cftime
import logging
import s3fs

import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset
from .manager import ManagerAttributes
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import snapBounds


@cached_dataset_manager
class ManagerAORC(manager_dataset.ManagerDataset):
    """AORC dataset.

    Explore the Analysis Of Record for Calibration (AORC) version 1.1 data

    https://registry.opendata.aws/noaa-nws-aorc/

    Using Xarray, Dask and hvPlot to explore the AORC version 1.1
    data. We read from a cloud-optimized Zarr dataset that is part of
    the NOAA Open Data Dissemination (NODD) program and we use a Dask
    cluster to parallelize the computation and reading of data chunks.

    AORC variables available to use:
    - APCP_surface
    - DLWRF_surface
    - DSWRF_surface
    - PRES_surface
    - SPFH_2maboveground
    - TMP_2maboveground
    - UGRD_10maboveground
    - VGRD_10maboveground

    There are eight variables representing the meteorological conditions

    - Total Precipitaion (APCP_surface): Hourly total precipitation
      (kgm-2 or mm) for Calibration (AORC) dataset
    - Air Temperature (TMP_2maboveground): Temperature (at 2 m
      above-ground-level (AGL)) (K)
    - Specific Humidity (SPFH_2maboveground): Specific humidity (at 2
      m AGL) (g g-1)
    - Downward Long-Wave Radiation Flux (DLWRF_surface): (1) longwave
      (infrared) and (2) radiation flux (at the surface) (W m-2)
    - Downward Short-Wave Radiation Flux (DSWRF_surface): (1) Downward
      shortwave (solar) and (2) radiation flux (at the surface) (W
      m-2)
    - Pressure (PRES_surface): Air pressure (at the surface) (Pa)
    - U-Component of Wind (UGRD_10maboveground): U (west-east) -
      components of the wind (at 10 m AGL) (m s-1)
    - V-Component of Wind (VGRD_10maboveground): V (south-north) -
      components of the wind (at 10 m AGL) (m s-1)

    **Precipitation and Temperature**

    The gridded AORC precipitation dataset contains one-hour
    Accumulated Surface Precipitation (APCP) ending at the "top" of
    each hour, in liquid water-equivalent units (kg m-2 to the nearest
    0.1 kg m-2), while the gridded AORC temperature dataset is
    comprised of instantaneous, 2 m above-ground-level (AGL)
    temperatures at the top of each hour (in Kelvin, to the nearest
    0.1).

    **Specific Humidity, Pressure, Downward Radiation, Wind**

    The development process for the six additional dataset components
    of the Conus AORC [i.e., specific humidity at 2m above ground (kg
    kg-1); downward longwave and shortwave radiation fluxes at the
    surface (W m-2); terrain-level pressure (Pa); and west-east and
    south-north wind components at 10 m above ground (m s-1)] has two
    distinct periods, based on datasets and methodology applied:
    1979-2015 and 2016-present.

    """

    # AORC constants
    VALID_VARIABLES = ['APCP_surface', 'DLWRF_surface',
                       'DSWRF_surface', 'PRES_surface', 'SPFH_2maboveground',
                       'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    DEFAULT_VARIABLES = ['APCP_surface', 'DLWRF_surface',
                         'DSWRF_surface', 'PRES_surface', 'SPFH_2maboveground',
                         'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    URL = 's3://noaa-nws-aorc-v1-1-1km'

    def __init__(self):
        native_start = cftime.datetime(1980, 1, 1, calendar='standard')
        native_end = cftime.datetime(2024, 12, 31, calendar='standard')
        native_crs = CRS.from_epsg(4326)
        native_resolution = 0.00833333  # 30 arc-second resolution

        attrs = ManagerAttributes(
            category='meteorology',
            product='AORC',
            source='ORNL DAAC Zarr',
            description='Analysis Of Record for Calibration (AORC) v1.1 hourly gridded meteorology.',
            product_short='aorc',
            source_short='ornl_daac_zarr',
            url='https://doi.org/10.25923/w6n8-qs02',
            license='public domain',
            citation='Fall et al. 2023',
            native_crs_in=native_crs,
            native_crs_out=native_crs,
            native_resolution=native_resolution,
            native_start=native_start,
            native_end=native_end,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
            is_temporal=True,
            is_resampled=True,
        )
        super().__init__(attrs)

    def isComplete(self, dir: str, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True if the cache directory contains a complete AORC download.

        Parameters
        ----------
        dir : str
            Absolute path to a candidate cache directory.
        request : ManagerDataset.Request
            The request being fulfilled.

        Returns
        -------
        bool
            True if ``aorc.nc`` exists in ``dir``.
        """
        return os.path.isfile(os.path.join(dir, 'aorc.nc'))

    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        ) -> manager_dataset.ManagerDataset.Request:
        """No-op, request not required."""
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — AORC S3 Zarr is always immediately available."""
        return True

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Download AORC data for the request from S3 Zarr.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
            The download is written to ``request._download_path/aorc.nc``.
        """
        output_path = os.path.join(request._download_path, 'aorc.nc')
        if os.path.isfile(output_path):
            logging.info(f'  Using existing: {output_path}')
            return

        start_year = request.start.year
        end_year = request.end.year

        # Snap bounds for the spatial sel() call for cache-directory consistency.
        xmin, ymin, xmax, ymax = snapBounds(request.geometry.bounds, self.native_resolution)

        dataset_years = list(range(start_year, end_year + 1))
        s3_out = s3fs.S3FileSystem(anon=True)
        fileset = [s3fs.S3Map(root=f"{self.URL}/{y}.zarr", s3=s3_out, check=False)
                   for y in dataset_years]

        ds_multi_year = xr.open_mfdataset(fileset, engine='zarr')
        logging.info(f'Full dataset size: {ds_multi_year.nbytes/1e12:.1f} TB')

        logging.info(f'Subsetting: lon {xmin, xmax}  lat {ymin, ymax}')
        ds_subset = ds_multi_year.sel(longitude=slice(xmin, xmax),
                                      latitude=slice(ymin, ymax))
        logging.info(f'Spatial subset size: {ds_subset.nbytes/1e9:.3f} GB')

        temporal_resampling = request.temporal_resampling
        if temporal_resampling is not None:
            ds_temporal = ds_subset.resample(time=temporal_resampling).mean()
            logging.info(f'Resampling in time: {temporal_resampling}')
        else:
            ds_temporal = ds_subset

        ds_temporal.to_netcdf(output_path)
        logging.info(f"Write to file: {output_path}")

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request,
                     chunk_time=None) -> xr.Dataset:
        """Open the cached AORC NetCDF file.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object with ``_download_path`` set.
        chunk_time : int, optional
            If provided, chunk the time dimension for Dask-lazy loading.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested AORC data.
        """
        path = os.path.join(request._download_path, 'aorc.nc')
        if chunk_time:
            dataset = xr.open_dataset(path, chunks={"time": chunk_time})
        else:
            dataset = xr.open_dataset(path)

        if request.variables != self.valid_variables:
            dataset = dataset[request.variables]

        return dataset
