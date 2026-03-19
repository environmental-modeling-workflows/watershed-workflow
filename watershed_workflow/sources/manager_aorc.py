"""Manager for interacting with AORC datasets."""
from typing import Tuple, List, Optional

import os
import numpy as np
import xarray as xr
import shapely
import cftime, datetime
import logging
import s3fs
import attr

import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import manager_dataset


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

    class Request(manager_dataset.ManagerDataset.Request):
        """AORC-specific request that includes filename for cached data."""
        def __init__(self,
                     request: manager_dataset.ManagerDataset.Request,
                     filename: str = ''):
            super().copyFromExisting(request)
            self.filename = filename

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
        _today = datetime.date.today()
        native_end = cftime.datetime(_today.year, _today.month, _today.day, calendar='standard')
        native_crs = CRS.from_epsg(4326)
        native_resolution = 0.00833333  # 30 arc-second resolution

        super().__init__(
            name='AORC v1.1',
            source='NOAA AWS S3 Zarr',
            native_resolution=native_resolution,
            native_crs_in=native_crs,
            native_crs_out=native_crs,
            native_start=native_start,
            native_end=native_end,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
            cache_category='meteorology',
            cache_extension='nc',
            has_varname=False,      # all variables in one file
            has_resampling=True,    # filename encodes temporal resampling rate
            short_name='AORC',
        )

        os.makedirs(self._cacheFolder(), exist_ok=True)


    def _download(self,
                  snapped_bounds: tuple,
                  start_year: int,
                  end_year: int,
                  temporal_resampling: Optional[str] = None,
                  force: bool = False,
                  geometry_bounds: Optional[tuple] = None) -> str:
        """Download AORC data for geometry and time range from S3 Zarr.

        Parameters
        ----------
        snapped_bounds : tuple of float
            (xmin, ymin, xmax, ymax) snapped, from request.snapped_bounds.
        start_year : int
            Starting year for data download.
        end_year : int
            Ending year for data download.
        temporal_resampling : str, optional
            Resample in time according to this time string, e.g. ``'1D'``.
        force : bool, optional
            If true, re-download even if a file already exists.
        geometry_bounds : tuple of float, optional
            Buffered un-snapped bounds for superset cache detection.

        Returns
        -------
        str
            The filename of the cached dataset.
        """
        os.makedirs(self._cacheFolder(), exist_ok=True)

        filename = self._cacheFilename(snapped_bounds,
                                       start_year=start_year,
                                       end_year=end_year,
                                       temporal_resampling=temporal_resampling)

        # Superset check before downloading
        if not os.path.exists(filename) and not force:
            if geometry_bounds is not None:
                superset = self._checkCache(geometry_bounds, snapped_bounds,
                                            start_year=start_year, end_year=end_year,
                                            temporal_resampling=temporal_resampling)
                if superset is not None:
                    logging.info(f'  Using superset cache: {superset}')
                    return superset

        if (not os.path.exists(filename)) or force:
            dataset_years = list(range(start_year, end_year + 1))
            s3_out = s3fs.S3FileSystem(anon=True)
            fileset = [s3fs.S3Map(root=f"{self.URL}/{y}.zarr", s3=s3_out, check=False)
                       for y in dataset_years]

            ds_multi_year = xr.open_mfdataset(fileset, engine='zarr')
            logging.info(f'Full dataset size: {ds_multi_year.nbytes/1e12:.1f} TB')

            xmin, ymin, xmax, ymax = snapped_bounds
            logging.info(f'Subsetting: lon {xmin, xmax}  lat {ymin, ymax}')
            ds_subset = ds_multi_year.sel(longitude=slice(xmin, xmax),
                                          latitude=slice(ymin, ymax))
            logging.info(f'Spatial subset size: {ds_subset.nbytes/1e9:.3f} GB')

            if temporal_resampling is not None:
                ds_temporal = ds_subset.resample(time=temporal_resampling).mean()
                logging.info(f'Resampling in time: {temporal_resampling}')
            else:
                ds_temporal = ds_subset

            ds_temporal.to_netcdf(filename)
            logging.info(f"Write to file: {filename}")

        else:
            logging.info(f"  Using existing: {filename}")

        return filename


    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        temporal_resampling: Optional[str] = None,
                        ) -> manager_dataset.ManagerDataset.Request:
        """Request AORC data - ready upon download completion.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
        temporal_resampling : str, optional
            Resample in time according to this time string, e.g. ``'1D'``,
            taking the mean.

        Returns
        -------
        ManagerDataset.Request
            New AORC request object with filename and is_ready flag set.
        """
        assert request.start is not None
        assert request.end is not None
        assert request.variables is not None

        start_year = request.start.year
        end_year = request.end.year
        if start_year > end_year:
            raise RuntimeError(
                f"Provided start year {start_year} is after provided end year {end_year}")

        filename = self._download(
            request.snapped_bounds, start_year, end_year,
            temporal_resampling=temporal_resampling,
            force=False,
            geometry_bounds=request.geometry.bounds)

        aorc_request = self.Request(request, filename)
        aorc_request.is_ready = True
        return aorc_request


    def _fetchDataset(self, request: manager_dataset.ManagerDataset.Request,
                      chunk_time=None) -> xr.Dataset:
        """Fetch AORC data.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing cached data reference.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested AORC data.
        """
        if chunk_time:
            dataset = xr.open_dataset(request.filename, chunks={"time": chunk_time})
        else:
            dataset = xr.open_dataset(request.filename)

        if request.variables != self.valid_variables:
            dataset = dataset[request.variables]

        return dataset
