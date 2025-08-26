"""Manager for interacting with AORC datasets."""
from typing import Tuple, List, Optional

import os
import numpy as np
import xarray as xr
import shapely
import cftime, datetime

import s3fs
# import zarr
# import dask

import watershed_workflow.sources.manager_raster
import watershed_workflow.sources.names
import watershed_workflow.crs
from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_dataset import ManagerDataset
import attr


class ManagerAORC(ManagerDataset):
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
    Accumulated Surface Precipitation (APCP) ending at the “top” of
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
    1979–2015 and 2016–present.

    """

    class Request(ManagerDataset.Request):
        """AORC-specific request that includes filename for cached data."""
        def __init__(self,
                     request : ManagerDataset.Request,
                     filename : str = ''):
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
        # AORC native data properties
        native_start = cftime.datetime(2007, 1, 1, calendar='standard')
        native_end = cftime.datetime(2024, 12, 31, calendar='standard')
        native_crs = CRS.from_epsg(4326)  # WGS84 Geographic
        native_resolution = 0.00833333  # ~1km in degrees (approximately 1km at mid-latitudes)
        
        # Initialize base class with correct parameter order
        super().__init__(
            name='AORC v1.1',
            source='NOAA AWS S3 Zarr',
            native_resolution=native_resolution,
            native_crs_in=native_crs,
            native_crs_out=native_crs,
            native_start=native_start,
            native_end=native_end,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES
        )
        
        # File naming for cached downloads
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'aorc', 'aorc_{start_year}-{end_year}_{north}x{west}_{south}x{east}.nc')

        # Check directory structure
        print(f'making directory: {self.names.folder_name()}')
        os.makedirs(self.names.folder_name(), exist_ok=True)


    def _cleanBounds(self, geometry : shapely.geometry.Polygon) -> list[float]:
        """Extract bounds from geometry already in native CRS and buffered by base class."""
        bounds = geometry.bounds
        return [np.round(b, 4) for b in bounds]
        

    def _download(self,
                  geometry : shapely.geometry.Polygon,
                  start_year : int,
                  end_year : int,
                  force : bool = False) -> str:
        """Download AORC data for geometry and time range from S3 Zarr.

        Parameters
        ----------
        geometry : shapely.geometry.Polygon
            Geometry in native CRS already transformed and buffered by base class.
        start_year : int
            Starting year for data download.
        end_year : int
            Ending year for data download.
        force : bool, optional
            If true, re-download even if a file already exists.

        Returns
        -------
        str
            The filename of the cached dataset.
        """
        # check directory structure
        os.makedirs(self.names.folder_name(), exist_ok=True)

        dataset_years = list(range(start_year, end_year+1))

        # Get bounds from geometry (already in native CRS and buffered)
        bounds = self._cleanBounds(geometry)
        
        # get the subset filename
        filename = self.names.file_name(start_year=start_year,
                                        end_year=end_year,
                                        east=bounds[0],
                                        south=bounds[1],
                                        west=bounds[2],
                                        north=bounds[3])


        if (not os.path.exists(filename)) or force:
            s3_out = s3fs.S3FileSystem(anon=True)
            fileset = [s3fs.S3Map(
                root=f"{self.URL}/{dataset_year}.zarr", s3=s3_out, check=False
            ) for dataset_year in dataset_years]
            

            # dask.config.set(temporary_directory=self.names.folder_name())
            # client = dask.distributed.Client()
            ds_multi_year = xr.open_mfdataset(fileset, engine='zarr')

            print(f'Variable size: {ds_multi_year.nbytes/1e12:.1f} TB')
            print(ds_multi_year)

            # Subset the dataset to the bounding box
            print('Subsetting:')
            print(f'  lon: {bounds[0], bounds[2]}')
            print(f'  lat: {bounds[1], bounds[3]}')
            ds_subset = ds_multi_year.sel(longitude=slice(bounds[0], bounds[2]),
                                          latitude=slice(bounds[1], bounds[3]))
            print(f'Variable size: {ds_subset.nbytes/1e9:.3f} GB')

            ds_subset = ds_subset.compute()
            ds_subset.to_netcdf(filename)
            # client.close()
        
        else:
            print(f"  Using existing: {filename}")

        return filename

    
    def _requestDataset(self, request: ManagerDataset.Request) -> ManagerDataset.Request:
        """Request AORC data - ready upon download completion.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
            
        Returns
        -------
        ManagerDataset.Request
            New AORC request object with filename and is_ready flag set.
        """
        assert request.start is not None
        assert request.end is not None
        assert request.variables is not None

        # Convert dates to years
        start_year = request.start.year
        end_year = request.end.year
        if start_year > end_year:
            raise RuntimeError(
                f"Provided start year {start_year} is after provided end year {end_year}")

        # Download data to cache (this may take time)
        filename = self._download(request.geometry, start_year, end_year, force=False)
        
        # Create new AORC-specific request with filename
        aorc_request = self.Request(request, filename)
        aorc_request.is_ready = True
        return aorc_request


    def _fetchDataset(self, request: ManagerDataset.Request) -> xr.Dataset:
        """Implementation of abstract method to fetch AORC data.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing cached data reference.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested AORC data.
        """
        # Open cached dataset
        dataset = xr.open_dataset(request.filename)
        
        # Filter to requested variables only
        if request.variables != self.valid_variables:
            # Only keep requested variables
            dataset = dataset[request.variables]
        
        # Ensure CRS is properly set
        if not hasattr(dataset, 'rio') or dataset.rio.crs is None:
            dataset = dataset.rio.write_crs(self.native_crs_out)
            
        return dataset
