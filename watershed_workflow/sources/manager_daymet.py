"""Manager for interacting with DayMet datasets."""
from typing import List

import os
import logging
import cftime, datetime
import xarray as xr

import watershed_workflow.crs
from watershed_workflow.crs import CRS
import watershed_workflow.utils.data

from . import manager_dataset
from .manager import ManagerAttributes
from .manager_dataset_cached import cached_dataset_manager
from .cache_info import snapBounds


@cached_dataset_manager
class ManagerDaymet(manager_dataset.ManagerDataset):
    """Daymet meterological datasets.

    Daymet is a historic, spatially interpolated product which ingests large
    number of point-sources of meterological data, aggregates them to daily
    time series, and spatially interpolates them onto a 1km gridded product that
    covers all of North America from 1980 to present [Daymet]_.

    Data is accessed through NASA Earthdata OPeNDAP, and requires a NASA 
    Earthdata login. This is the same account used for MODIS (AppEEARS), so add
    the username and password to ~/.netrc like below:: 

        machine urs.earthdata.nasa.gov
        login MY_USERNAME
        password MY_PASSWORD

    Variable names and descriptions

    .. list-table::
        :widths: 25 25 75
        :header-rows: 1

        * - name
          - units
          - description
        * - prcp
          - :math:`mm / day`
          - Total daily precipitation
        * - tmin, tmax
          - :math:`^\\circ C`
          - Min/max daily air temperature
        * - srad
          - :math:`W / m^2`
          - Incoming solar radiation - per DAYLIT time!
        * - vp
          - :math:`Pa`
          - Vapor pressure
        * - swe
          - :math:`Kg / m^2`
          - Snow water equivalent
        * - dayl
          - :math:`s / day`
          - Duration of sunlight

    .. [Daymet] https://daymet.ornl.gov
    """

    # DayMet-specific constants
    DOI = "10.3334/ORNLDAAC/2129"
    GRANULE_NAME = "*_na_{variable}_{year}.nc"
    VALID_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    DEFAULT_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']

    def __init__(self):
        native_start = cftime.datetime(1980, 1, 1, calendar='noleap')
        # DayMet is released annually, ~30+ days after year-end; update this
        # once the following year's data is confirmed published.
        native_end = cftime.datetime(2025, 12, 31, calendar='noleap')
        native_crs_daymet = CRS.from_proj4(
            '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
        )
        native_resolution = 1000.0  # 1km in meters

        attrs = ManagerAttributes(
            category='meteorology',
            product='DayMet 1km',
            source='NASA Earthdata ORNL DAAC OPeNDAP',
            description='Daily surface weather data on a 1-km grid for North America.',
            product_short='daymet',
            source_short='nasa_earthdata_opendap',
            url='https://daymet.ornl.gov',
            license='public domain',
            citation='Thornton et al. 2022',
            native_crs_in=native_crs_daymet,
            native_crs_out=native_crs_daymet,
            native_resolution=native_resolution,
            native_start=native_start,
            native_end=native_end,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
            is_temporal=True,
        )
        super().__init__(attrs)
        self._session = None


    def _authenticate(self) -> None:
        """Log in to NASA Earthdata once, using ~/.netrc, and cache the OPeNDAP session."""
        if self._session is not None:
            return
        
        import earthaccess
    
        try:
            earthaccess.login(strategy='netrc')
        except Exception as e:
            raise RuntimeError(
                'NASA Earthdata credentials not found. Add an entry to ~/.netrc:\n\n'
                '    machine urs.earthdata.nasa.gov login <username> password <password>\n\n'
                'Register for a free account at https://urs.earthdata.nasa.gov\n'
                f'Original error: {e}'
            ) from e

        self._session = earthaccess.get_requests_https_session()


    def _findOpendapURL(self, var: str, year: int) -> str:
        """Search for the OPeNDAP URL of a (variable, year) granule.

        Parameters
        ----------
        var : str
            Variable name.
        year : int
            A year in the valid range (currently 1980-2025).
        """
        import earthaccess

        results = earthaccess.search_data(
            doi=self.DOI,
            temporal=(f'{year}-01-01', f'{year}-12-31'),
            granule_name=self.GRANULE_NAME.format(variable=var, year=year),
        )

        for item in results:
            for url in item['umm']['RelatedUrls']:
                if 'OPENDAP' in url.get('Description', '').upper():
                    ce = f"?dap4.ce=/{var}%3B/y%3B/x%3B/time"
                    return url['URL'].replace('https', 'dap4') + ce

        raise RuntimeError(f'DayMet: no OPeNDAP granule found for variable "{var}", year {year}')

    
    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Download per-variable, per-year DayMet NetCDF files.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object containing geometry, dates, and variables.
            Files are written to ``request._download_path/{var}_{year}.nc``.
        """
        start_year = request.start.year
        end_year = request.end.year

        # Snap bounds in DayMet CRS.
        snapped_bounds = snapBounds(request.geometry.bounds, self.native_resolution)

        for var in request.variables:
            for year in range(start_year, end_year + 1):
                filename = os.path.join(request._download_path, f'{var}_{year}.nc')
                self._downloadFile(var, year, snapped_bounds, filename)


    def _downloadFile(self,
                      var: str,
                      year: int,
                      snapped_bounds: tuple,
                      filename: str,
                      force: bool = False) -> None:
        """Download a single DayMet NetCDF file.

        Parameters
        ----------
        var : str
            Variable name, see class documentation.
        year : int
            A year in the valid range (currently 1980-2025).
        snapped_bounds : tuple of float
            (xmin, ymin, xmax, ymax) in DayMet CRS, from self._downloadDataset(request).
        filename : str
            Destination NetCDF file path.
        force : bool, optional
            If true, re-download even if the file already exists.
        """
        logging.info(f"Collecting DayMet file to tile bounds: {snapped_bounds}")

        if (not os.path.exists(filename)) or force:
            self._authenticate()
            url = self._findOpendapURL(var, year)
            logging.info(f"  Downloading: {url}")
            logging.info(f"      to file: {filename}")

            xmin, ymin, xmax, ymax = snapped_bounds
            ds = xr.open_dataset(url, engine='pydap', session=self._session)
    
            # y is stored in decreasing order on the DayMet grid
            y_slice = slice(ymax, ymin) if ds.y[0] > ds.y[-1] else slice(ymin, ymax)
            ds_subset = ds.sel(x=slice(xmin, xmax), y=y_slice)
            if ds_subset.sizes['x'] == 0 or ds_subset.sizes['y'] == 0:
                raise RuntimeError(f'DayMet: bounds {snapped_bounds} do not intersect the DayMet domain')
            ds_subset = ds_subset.compute()

            # save to a temporary filename first in case of interruption
            tmp_filename = filename + '.tmp'
            ds_subset.to_netcdf(tmp_filename)
            os.replace(tmp_filename, filename)
        else:
            logging.info(f"  Using existing: {filename}")

                
    def isComplete(self, dir: str, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True if all per-variable, per-year files exist in the cache directory.

        Parameters
        ----------
        dir : str
            Absolute path to a candidate cache directory.
        request : ManagerDataset.Request
            The request being fulfilled.

        Returns
        -------
        bool
            True if all ``{var}_{year}.nc`` files exist for every requested
            variable and year.
        """
        start_year = request.start.year
        end_year = request.end.year
        for var in request.variables:
            for year in range(start_year, end_year + 1):
                if not os.path.isfile(os.path.join(dir, f'{var}_{year}.nc')):
                    return False
        return True


    def _preprocessParameters(self, *args, **kwargs) -> manager_dataset.ManagerDataset.Request:
        """ Make the end date include its 12:00 record.

        DayMet records are stamped at 12:00 of each day. This operation includes
        the end day's record when end is given as a date (00:00)
        """
        request = super()._preprocessParameters(*args, **kwargs)
        if request.end is not None and (
            request.end.hour, request.end.minute, request.end.second
        ) == (0, 0, 0): 
            request.end += datetime.timedelta(hours=12)
        return request


    def _requestDataset(self, 
                        request: manager_dataset.ManagerDataset.Request,
                        ) -> manager_dataset.ManagerDataset.Request:
        """No op -- no server side request."""
        return request


    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — DayMet downloads are synchronous."""
        return True


    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Open all per-variable, per-year files and concatenate into a dataset.

        Parameters
        ----------
        request : ManagerDataset.Request
            Request object with ``_download_path`` set.

        Returns
        -------
        xr.Dataset
            Dataset containing all requested DayMet data.
        """
        start_year = request.start.year
        end_year = request.end.year
        variables = request.variables

        file_lists = []
        for var in variables:
            vfiles = [os.path.join(request._download_path, f'{var}_{year}.nc')
                      for year in range(start_year, end_year + 1)]
            file_lists.append(vfiles)

        return self._openFiles(file_lists, variables)


    def _openFiles(self,
                   file_lists: List[List[str]],
                   variables: List[str]) -> xr.Dataset:
        """Open all files and concatenate them into a single xarray dataset."""
        fnames_by_var = list(zip(file_lists, variables))
        ds_list_allvars = []

        for fnames, var in fnames_by_var:
            ds_list = [xr.open_dataset(fname) for fname in fnames]
            ds_concat = xr.concat(ds_list, dim="time")
            ds_list_allvars.append(ds_concat[var])

        ds_combined = xr.Dataset({da.name: da for da in ds_list_allvars})

        # raw x/y coordinates have already been meters
        new_time = watershed_workflow.utils.data.convertTimesToCFTimeNoleap(
            watershed_workflow.utils.data.convertTimesToCFTime(ds_combined['time'].values))

        ds_combined = ds_combined.assign_coords(time=new_time)
        ds_combined.attrs = ds_concat.attrs

        return ds_combined
