"""Manager for interacting with DayMet datasets."""

import os, sys
import logging
import numpy as np
import pyproj
import requests
import requests.exceptions
import shapely.geometry
import cftime, datetime
import rasterio.transform
import xarray as xr
# import netCDF4

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.crs
import watershed_workflow.config
import watershed_workflow.warp
import watershed_workflow.sources.names
# import watershed_workflow.datasets

def _previous_month():
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 1
    if month == 0:
        year = year - 1
        month = 12
    return cftime.datetime(year, month, 1, calendar='noleap')


class FileManagerDaymet:
    """Daymet meterological datasets.

    Daymet is a historic, spatially interpolated product which ingests large
    number of point-sources of meterological data, aggregates them to daily
    time series, and spatially interpolates them onto a 1km gridded product that
    covers all of North America from 1980 to present [Daymet]_.

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

    _START = cftime.datetime(1980, 1, 1, calendar='noleap')
    _END = _previous_month()
    VALID_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    DEFAULT_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'dayl']
    URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/daymet_v4_daily_na_{variable}_{year}.nc"

    def __init__(self):
        self.name = 'DayMet 1km'
        self.names = watershed_workflow.sources.names.Names(
            self.name, 'meteorology', 'daymet',
            'daymet_{var}_{year}_{north}x{west}_{south}x{east}.nc')

    # def _read_var(self, fname, var):
        
    #     # read the data using xarray
    #     ds = xr.open_dataset(fname, engine="netcdf4") # netcdf4 is the default engine
    #     # convert the x and y coordinates from km to meters and update the attributes
    #     attrs_ref = ds.x.attrs 
    #     attrs_ref['units'] = 'm'
    #     ds = ds.assign_coords(x=ds.x * 1000, y=ds.y * 1000)
    #     ds.x.attrs = attrs_ref
    #     ds.y.attrs = attrs_ref

    #     x = ds.x.values 
    #     y = ds.y.values 
    #     time = ds.time.values
    #     assert (len(time) == 365)
    #     val = ds[var].values
    #     return x, y, val

    def _download(self, 
                  var : str,
                  year : int,
                  bounds : list[float],
                  force : bool = False) -> str:
        """Download a NetCDF file covering the bounds.

        Parameters
        ----------
        var : str
          Name the variable, see table in the class documentation.
        year : int
          A year in the valid range (currently 1980-2018)
        bounds : [xmin, ymin, xmax, ymax]
          Desired bounds, in the LatLon CRS.
        force : bool, optional
          If true, re-download even if a file already exists.

        Returns
        -------
        filename : str
          Resulting NetCDF file.
        """
        logging.info("Collecting DayMet file to tile bounds: {}".format(bounds))

        # check directory structure
        os.makedirs(self.names.folder_name(), exist_ok=True)

        # get the target filename
        bounds = [f"{b:.4f}" for b in bounds]
        filename = self.names.file_name(var=var,
                                        year=year,
                                        north=bounds[3],
                                        east=bounds[2],
                                        west=bounds[0],
                                        south=bounds[1])

        if (not os.path.exists(filename)) or force:
            url_dict = { 'year': str(year), 'variable': var }
            url = self.URL.format(**url_dict)
            logging.info("  Downloading: {}".format(url))
            logging.info("      to file: {}".format(filename))

            request_params = [('var', 'lat'), ('var', 'lon'), ('var', var), ('west', bounds[0]),
                              ('south', bounds[1]), ('east', bounds[2]), ('north', bounds[3]),
                              ('horizStride', '1'),
                              ('time_start', '{}-01-01T12:00:00Z'.format(year)),
                              ('time_end', '{}-12-31T12:00:00Z'.format(year)), ('timeStride', '1'),
                              ('accept', 'netcdf')]

            r = requests.get(url, params=request_params, verify=source_utils.getVerifyOption())
            r.raise_for_status()

            with open(filename, 'wb') as fid:
                fid.write(r.content)

        else:
            logging.info("  Using existing: {}".format(filename))

        return filename

    def _clean_date(self, date : str | datetime.date) -> datetime.date:
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

    def _clean_bounds(self, 
                      polygon_or_bounds : dict | shapely.geometry.Polygon | list[float],
                      crs : str,
                      buffer : float) -> list[float]:
        
        """Compute bounds in the required CRS from a polygon or bounds in a given crs"""
        if type(polygon_or_bounds) is dict:
            polygon_or_bounds = watershed_workflow.utils.create_shply(polygon_or_bounds)
        if type(polygon_or_bounds) is shapely.geometry.Polygon:
            bounds_ll = watershed_workflow.warp.shply(polygon_or_bounds, crs,
                                                      watershed_workflow.crs.latlon_crs).bounds
        else:
            bounds_ll = watershed_workflow.warp.bounds(polygon_or_bounds, crs,
                                                       watershed_workflow.crs.latlon_crs)

        feather_bounds = list(bounds_ll[:])
        feather_bounds[0] = np.round(feather_bounds[0] - buffer, 4)
        feather_bounds[1] = np.round(feather_bounds[1] - buffer, 4)
        feather_bounds[2] = np.round(feather_bounds[2] + buffer, 4)
        feather_bounds[3] = np.round(feather_bounds[3] + buffer, 4)
        return feather_bounds

    def _open_files(self, 
                    filenames : list[str],
                    var : str,
                    start : datetime.date,
                    end : datetime.date) -> dict:
        """Opens and loads the files, making a single array."""
        # NOTE: this probably needs to be refactored to not load the whole thing into memory?
        nyears = len(filenames)
        data = None
        for i, fname in enumerate(filenames):
            x, y, v = self._read_var(fname, var)  # returned v.shape(nband, nrow, ncol)
            if data is None:
                # note nrows, ncols ordering
                data = np.zeros((nyears * 365, len(y), len(x)), 'd')

            # stuff in the right spot
            data[i * 365:(i+1) * 365, :, :] = v

        # times is a range -- note that DayMet works on a noleap calendar
        origin = cftime.datetime(start.year, 1, 1, calendar='noleap')
        times = np.array([origin + datetime.timedelta(days=i) for i in range(365 * nyears)])

        # trim to start, end
        i_start = (start - origin).days
        i_end = i_start + (end - start).days
        times = times[i_start:i_end]
        data = data[i_start:i_end]

        # profile
        profile = dict()
        profile['crs'] = watershed_workflow.crs.daymet_crs()
        profile['width'] = len(x)
        profile['height'] = len(y)
        profile['count'] = len(times)
        profile['dx'] = (x[1:] - x[:-1]).mean()  # convert to m
        profile['dy'] = (y[1:] - y[:-1]).mean()  # convert to m
        profile['resolution'] = (profile['dx'], -profile['dy'])
        profile['driver'] = 'h5'  # hint that this was not a real raster

        profile['transform'] = rasterio.transform.from_bounds(x[0], y[-1], x[-1], y[0],
                                                              profile['width'], profile['height'])
        profile['nodata'] = -9999
        return watershed_workflow.datasets.Data(profile, times, data)

    def _open_files_xarray(self, filenames : list[str], variables : list[str]) -> xr.Dataset:
        """Open all files and concatenate them into a single xarray dataset."""

        fnames_by_var = list(zip(filenames, variables))
        ds_list_allvars = []

        for info in fnames_by_var:
            
            var = info[1]
            fnames = info[0]
            
            ds_list = []
            for fname in fnames:
                ds = xr.open_dataset(fname, engine="netcdf4") # netcdf4 is the default engine
                ds_list.append(ds)

            ds_concat = xr.concat(ds_list, dim="time")
            ds_list_allvars.append(ds_concat[var])

        ds_combined = xr.Dataset({da.name: da for i, da in enumerate(ds_list_allvars)})
        ds_combined.attrs = ds_concat.attrs
        # convert the x and y coordinates from km to meters and update the attributes
        attrs_ref = ds_combined.x.attrs 
        attrs_ref['units'] = 'm'
        ds_combined = ds_combined.assign_coords(x=ds.x * 1000, y=ds.y * 1000)
        ds_combined.x.attrs = attrs_ref
        ds_combined.y.attrs = attrs_ref
        return ds_combined
    
    def _prepare_ats_data(self, data : xr.Dataset) -> dict:
      # Initialize a dictionary to store ATS data
      dout = dict()

      # Extract mean air temperature in Celsius
      mean_air_temp_C = (data['tmax'][:].data  + data['tmin'][:].data) / 2 # in Celsius

      # Convert precipitation from mm/day to m/s
      precip_ms = data['prcp'][:].data / (1.e3 * 24 * 3600)  # mm/day to m/s

      # Calculate time in seconds from the start of the dataset
      time_start_global = data.time.data[0]
      time = (pd.to_datetime(data.time.data) - time_start_global).total_seconds()

      # Populate the ATS dictionary with processed data
      dout['air temperature [K]'] = mean_air_temp_C + 273.15  # Convert to Kelvin
      dout['incoming shortwave radiation [W m^-2]'] = data['srad'][:].data  # W/m^2
      dout['vapor pressure air [Pa]'] = data['vp'][:].data   #  Pa
      dout['precipitation rain [m s^-1]'] = np.where(mean_air_temp_C >= 0, precip_ms, 0)  # Rainfall in m/s
      dout['precipitation snow [m SWE s^-1]'] = np.where(mean_air_temp_C < 0, precip_ms, 0)  # Snowfall in m SWE/s
      dout['day length [s]'] = data['daylength'][:].data  # s
      dout['time [s]'] = time  # Time in seconds

      # Extract x and y coordinates
      coords_x = data.x.data
      coords_y = data.y.data 

      return {'data': dout, 'x': coords_x, 'y': coords_y}



    def getDataset(self,
                 polygon_or_bounds : dict | shapely.geometry.Polygon | list[float],
                 crs : str,
                 start : str | datetime.date = None,
                 end : str | datetime.date = None,
                 variables : list[str] = None,
                 force_download : bool = False,
                 buffer : float = 0.01) -> dict:
        
        """Gets file for a single year and single variable.

        Parameters
        ----------
        polygon_or_bounds : fiona or shapely shape, or [xmin, ymin, xmax, ymax]
          Collect a file that covers this shape or bounds.
        crs : CRS object
          Coordinate system of the above polygon_or_bounds
        start : str or datetime.date object, optional
          Date for the beginning of the data, in YYYY-MM-DD. Valid is
          >= 2002-07-01.
        end : str or datetime.date object, optional
          Date for the end of the data, in YYYY-MM-DD. Valid is
          < the current month (DayMet updates monthly.)
        variables : str or list, optional
          Name the variables to download, see class-level
          documentation for choices.  Default is
          [prcp,tmin,tmax,vp,srad].
        force_download : bool
          Download or re-download the file if true.
        buffer : float
          Buffer the bounds by this amount, in degrees. The default is 0.01.

        Returns
        -------
        datasets.Dataset
          Dataset object containing the met data.
        """
        if start is None:
            start = self._START
        start = self._clean_date(start)
        start_year = start.year

        if end is None:
            end = self._END
        end = self._clean_date(end)
        end_year = (end - datetime.timedelta(days=1)).year
        if start_year > end_year:
            raise RuntimeError(
                f"Provided start year {start_year} is after provided end year {end_year}")

        if variables is None:
            variables = self.DEFAULT_VARIABLES
        for var in variables:
            if var not in self.VALID_VARIABLES:
                raise ValueError("DayMet data supports variables: {} (not {})".format(
                    ', '.join(self.VALID_VARIABLES), var))

        # clean bounds
        bounds = self._clean_bounds(polygon_or_bounds, crs, buffer=buffer)

        # download files
        filenames = []
        for var in variables:
            filename_var = []
            for year in range(start_year, end_year + 1):
                fname = self._download(var, year, bounds, force=force_download)
                filename_var.append(fname)
            filenames.append(filename_var)


        # open files
        ds = self._open_files_xarray(filenames, variables)
        dset = self._prepare_ats_data(ds)

        # # open files
        # dset = None
        # for fnames, var in zip(filenames, variables):
        #     data = self._open_files(fnames, var, start, end)
        #     if dset is None:
        #         dset = watershed_workflow.datasets.Dataset(data.profile, data.times)
        #     dset[var] = data.data
        # return dset
        return ds
