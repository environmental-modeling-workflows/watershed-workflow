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
from watershed_workflow.crs import CRS


class ManagerAORC:
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
    _START = cftime.datetime(2007, 1, 1, calendar='noleap')
    _END = cftime.datetime(2025, 1, 1, calendar='noleap')

    VALID_VARIABLES = ['APCP_surface', 'DLWRF_surface',
                       'DSWRF_surface', 'PRES_surface', 'SPFH_2maboveground',
                       'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    DEFAULT_VARIABLES = ['APCP_surface', 'DLWRF_surface',
                         'DSWRF_surface', 'PRES_surface', 'SPFH_2maboveground',
                         'TMP_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground']
    URL = f's3://noaa-nws-aorc-v1-1-1km'

    def __init__(self):
        self.name = 'AORC'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'aorc', 'aorc_{start_year}-{end_year}_{north}x{west}_{south}x{east}.nc')

        # check directory structure
        os.makedirs(self.names.folder_name(), exist_ok=True)


    def _cleanDate(self, date : str | cftime.DatetimeNoLeap) -> cftime.DatetimeNoLeap:
        """Returns a string of the format needed for use in the filename and request."""
        if type(date) is str:
            date_split = date.split('-')
            date = cftime.datetime(int(date_split[0]),
                                   int(date_split[1]),
                                   int(date_split[2]),
                                   calendar='noleap')
        if date < self._START:
            raise ValueError(f"Invalid date {date}, must be after {self._START}.")
        if date > self._END:
            raise ValueError(f"Invalid date {date}, must be before {self._END}.")
        return date

    def _cleanBounds(self, 
                     geometry : shapely.geometry.base.BaseGeometry,
                     geometry_crs : CRS,
                     buffer : float) -> list[float]:
        """Compute bounds in the required CRS from a polygon or bounds in a given crs"""
        bounds_ll = watershed_workflow.warp.shply(geometry, geometry_crs,
                                                  watershed_workflow.crs.latlon_crs).bounds
        feather_bounds = list(bounds_ll[:])
        feather_bounds[0] = np.round(feather_bounds[0] - buffer, 4)
        feather_bounds[1] = np.round(feather_bounds[1] - buffer, 4)
        feather_bounds[2] = np.round(feather_bounds[2] + buffer, 4)
        feather_bounds[3] = np.round(feather_bounds[3] + buffer, 4)
        return feather_bounds
        

    def _download(self,
                  geometry : shapely.geometry.base.BaseGeometry,
                  geometry_crs: CRS,
                  start_year : int,
                  end_year : int,
                  buffer : float = 0.01,
                  force : bool = False) -> str:
        """This method downloads AORC data for a specified geometry and time range.

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

        This is not required but it speeds up computations. Here we
        start a local cluster that uses the cores available on the
        computer running the notebook server. There are many other
        ways to set up Dask clusters that can scale larger than this.
        If you are running this on your local machine add this -
        dask.config.set(temporary_directory='/dask-worker-space') -
        under import dask

        """
        dataset_years = list(range(start_year, end_year+1))

        # Reproject the geometry to the AORC CRS
        bounds = self._cleanBounds(geometry, geometry_crs, buffer)
        
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

    
    def getDataset(self, 
                   geometry : shapely.geometry.base.BaseGeometry,
                   geometry_crs : CRS,
                   start : Optional[str | cftime.DatetimeNoLeap] = None,
                   end : Optional[str | cftime.DatetimeNoLeap] = None,
                   force_download : bool = False,
                   buffer : float = 0.01) -> xr.Dataset:
        if start is None:
            start = self._START_YEAR
        start = self._cleanDate(start)
        assert not isinstance(start, str)
        start_year = start.year

        if end is None:
            end = self._END
        end = self._cleanDate(end)
        assert not isinstance(end, str)
        end_year = (end - datetime.timedelta(days=1)).year
        if start_year > end_year:
            raise RuntimeError(
                f"Provided start year {start_year} is after provided end year {end_year}")

        filename = self._download(geometry, geometry_crs, start_year, end_year, buffer, force_download)
        return xr.open_dataset(filename)
