"""Manager for interacting with AORC datasets."""

import os, sys
import logging
import numpy as np
import pandas
import xarray as xr
import geopandas as gpd
from typing import Tuple
import shapely
from shapely.geometry import Polygon

import fsspec
import s3fs
import zarr
import dask
from dask.distributed import Client

import watershed_workflow.sources.manager_raster
import watershed_workflow.sources.names


class FileManagerAORC:
    """AORC dataset.
    # Explore the Analysis Of Record for Calibration (AORC) version 1.1 data

    https://registry.opendata.aws/noaa-nws-aorc/

    Using Xarray, Dask and hvPlot to explore the AORC version 1.1 data. We read from a cloud-optimized Zarr dataset that is part of the NOAA Open Data Dissemination (NODD) program and we use a Dask cluster to parallelize the computation and reading of data chunks.
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
    - Total Precipitaion (APCP_surface): Hourly total precipitation (kgm-2 or mm) for Calibration (AORC) dataset
    - Air Temperature (TMP_2maboveground): Temperature (at 2 m above-ground-level (AGL)) (K)
    - Specific Humidity (SPFH_2maboveground): Specific humidity (at 2 m AGL) (g g-1)
    - Downward Long-Wave Radiation Flux (DLWRF_surface): (1) longwave (infrared) and (2) radiation flux (at the surface) (W m-2)
    - Downward Short-Wave Radiation Flux (DSWRF_surface): (1) Downward shortwave (solar) and (2) radiation flux (at the surface) (W m-2)
    - Pressure (PRES_surface): Air pressure (at the surface) (Pa)
    - U-Component of Wind (UGRD_10maboveground): U (west-east) - components of the wind (at 10 m AGL) (m s-1)
    - V-Component of Wind (VGRD_10maboveground): V (south-north) - components of the wind (at 10 m AGL) (m s-1)

    **Precipitation and Temperature**

    The gridded AORC precipitation dataset contains one-hour Accumulated Surface Precipitation (APCP) ending at the “top” of each hour, in liquid water-equivalent units (kg m-2 to the nearest 0.1 kg m-2), while the gridded AORC temperature dataset is comprised of instantaneous, 2 m above-ground-level (AGL) temperatures at the top of each hour (in Kelvin, to the nearest 0.1).

    **Specific Humidity, Pressure, Downward Radiation, Wind**

    The development process for the six additional dataset components of the Conus AORC [i.e., specific humidity at 2m above ground (kg kg-1); downward longwave and shortwave radiation fluxes at the surface (W m-2); terrain-level pressure (Pa); and west-east and south-north wind components at 10 m above ground (m s-1)] has two distinct periods, based on datasets and methodology applied: 1979–2015 and 2016–present.

    """

    _START_YEAR = 2007
    _END_YEAR = 2024
    VALID_VARIABLES = ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface', 'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    DEFAULT_VARIABLES = ['APCP_surface', 'DLWRF_surface', 'DSWRF_surface', 'PRES_surface', 'SPFH_2maboveground', 'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    URL = f's3://noaa-nws-aorc-v1-1-1km'

    

    def __init__(self):
        self.name = 'AORC'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'aorc', 'aorc_{start_year}-{end_year}_{north}x{west}_{south}x{east}.nc')

    def _download(self,
                geometry: gpd.GeoDataFrame | gpd.GeoSeries | Tuple[float, float, float, float],
                geometry_crs: str = None,
                start_year : int = None,
                end_year : int = None,
                buffer : float = 0.05,
                force : bool = False) -> str:
        
        """
        This method downloads AORC data for a specified geometry and time range.

        Parameters
        ----------
        geometry : gpd.GeoDataFrame | gpd.GeoSeries | Tuple[float, float, float, float]
            The geometry for which the dataset is to be retrieved. It can be a GeoDataFrame,
            GeoSeries, or a tuple representing the bounding box (minx, miny, maxx, maxy).
        geometry_crs : str, optional
            The coordinate reference system of the geometry. If not provided, it defaults
            to the CRS of the geometry if available, otherwise assumes 'epsg:4326'.
        start_year : int, optional
            The starting year for the data download. Defaults to the class-level _START_YEAR.
        end_year : int, optional
            The ending year for the data download. Defaults to the class-level _END_YEAR.
        buffer : float, optional
            Buffer the bounds by this amount, in degrees. The default is 0.05.
        force : bool, optional
            If true, re-download even if a file already exists.

        Returns
        -------
        str
            The filename of the downloaded dataset.

            
        This function starts a Dask cluster

        This is not required but it speeds up computations. Here we start a local cluster that uses the cores available on the computer running the notebook server. There are many other ways to set up Dask clusters that can scale larger than this.
        If you are running this on your local machine add this - dask.config.set(temporary_directory='/dask-worker-space') - under import dask
        
        """

        if start_year is None:
            start_year = self._START_YEAR

        if end_year is None:
            end_year = self._END_YEAR

        if start_year > end_year:
            raise RuntimeError(
                f"Provided start year {start_year} is after provided end year {end_year}")
        

        # check directory structure
        os.makedirs(self.names.folder_name(), exist_ok=True)

        dask.config.set(temporary_directory=self.names.folder_name())

        client = Client()
        print(client)


        dataset_years = list(range(start_year,end_year+1))

        s3_out = s3fs.S3FileSystem(anon=True)
        fileset = [s3fs.S3Map(
                    root=f"s3://{self.URL}/{dataset_year}.zarr", s3=s3_out, check=False
                ) for dataset_year in dataset_years]

        ds_multi_year = xr.open_mfdataset(fileset, engine='zarr')

        print(f'Variable size: {ds_multi_year.nbytes/1e12:.1f} TB')
        ds_multi_year

        # Read the AORC mesh file. This was previously created from the AORC dataset, saving time in the clipping step.
        # TODO: Find a good way to distribute this mesh file
        filename_mesh_aorc = '/Users/8n8/Downloads/aorc_data/aorc_meshpoints.feather'

        gdf_mesh = gpd.read_feather(filename_mesh_aorc) # This is in 'epsg:4326'

        # Clip mesh to watershed boundary

        if geometry_crs is None:
            if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
                geometry_crs = geometry.crs or "epsg:4326"
            elif isinstance(geometry, Tuple):
                geometry_crs = "epsg:4326"
                print("Warning: geometry_crs was not provided. Assuming geometry_crs = 'epsg:4326'")
        
        if isinstance(geometry, Tuple):
            # Create a GeoDataFrame from the tuple coordinates
            minx, miny, maxx, maxy = geometry
            geometry = gpd.GeoDataFrame(
                {'geometry': [shapely.geometry.box(minx, miny, maxx, maxy)]},
                crs=geometry_crs,
                index=['domain']
            )

        # Reproject the geometry to the AORC CRS
        geometry_projected = geometry.to_crs(gdf_mesh.crs)

        bounds = geometry_projected.bounds

        # Create a polygon from the rectangle
        clip_domain = gpd.GeoDataFrame(geometry=[Polygon([(bounds.minx[0] - buffer, bounds.miny[0] - buffer),
                        (bounds.minx[0] - buffer, bounds.maxy[0] + buffer),
                        (bounds.maxx[0] + buffer, bounds.maxy[0] + buffer),
                        (bounds.maxx[0] + buffer, bounds.miny[0] - buffer)])])

        # Define the bounding box coordinates
        lon_min, lon_max = clip_domain.total_bounds[0], clip_domain.total_bounds[2]
        lat_min, lat_max = clip_domain.total_bounds[1], clip_domain.total_bounds[3]

        # Subset the dataset to the bounding box
        ds_subset = ds_multi_year.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

        print(f'Variable size: {ds_subset.nbytes/1e9:.3f} GB')

        # Display the subset dataset
        ds_subset

        # # We can slice the dates to get a specific time range
        # ds_subset = ds_subset.sel(time=slice('2007-01-01', '2007-01-02'))

        # Save the subset dataset
        
        filename = self.names.file_name(
                                        start_year=start_year,
                                        end_year=end_year,
                                        north=bounds[3],
                                        east=bounds[2],
                                        south=bounds[1],
                                        west=bounds[0])
        
        if (not os.path.exists(filename)) or force:
            ds_subset = ds_subset.compute()
            ds_subset.to_netcdf(filename)
        else:
            print(f"  Using existing: {filename}")

        client.close()

        return filename

    def getDataset(self, 
                   geometry : gpd.GeoDataFrame | gpd.GeoSeries | Tuple[float, float, float, float],
                   geometry_crs : str = None,
                   start_year : int = None,
                   end_year : int = None,
                   variables : list[str] = None,
                   force_download : bool = False,
                   buffer : float = 0.05) -> dict:
        

        filename = self._download(geometry, geometry_crs, start_year, end_year, buffer, force_download)

        ds = xr.open_dataset(filename)

        return ds