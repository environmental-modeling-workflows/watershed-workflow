"""Manager for downloading SSEBop actual evapotranspiration products."""

import os
import io
import zipfile
import logging
import datetime
import cftime
import xarray as xr
import rioxarray

import watershed_workflow.crs

from . import utils as source_utils
from . import manager_dataset
from . import cache_info
from .manager import ManagerAttributes


#: URL template for SSEBop CONUS MODIS-based ETa products.
_BASE_URL = ('https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/'
             'conus/eta/modis_eta/{res_dir}/downloads/{stem}.zip')

#: Maps temporal_resolution -> (url_res_dir, stem_function).
_TEMPORAL_CONFIGS = {
    'daily':   ('daily',   lambda d: f'det{d.year}{d.timetuple().tm_yday:03d}.modisSSEBopETactual'),
    '8day':    ('8day',    lambda d: f'e{d.year}{d.timetuple().tm_yday:03d}'),
    'monthly': ('monthly', lambda d: f'm{d.year}{d.month:02d}'),
    'yearly':  ('annual',  lambda d: f'y{d.year}'),
}

#: NoData sentinel value in the raw uint16 files.
_NODATA = 9999

#: Scale factor: raw uint16 -> mm/day.
_SCALE = 0.001


class ManagerSSEBop(manager_dataset.ManagerDataset):
    """SSEBop actual evapotranspiration (ETa) for CONUS.

    Downloads USGS FEWS SSEBop MODIS-based ETa products [SSEBop]_ at daily,
    8-day, monthly, or yearly temporal resolution.  Data covers the Continental
    United States (CONUS) at approximately 1 km resolution in EPSG:4326.

    .. [SSEBop] Senay, G.B., et al. (2013). Operational evapotranspiration
       mapping using remote sensing and weather datasets. JAWRA.
       https://doi.org/10.1111/jawr.12057

    Full CONUS GeoTIFF files (~1.8 MB each for daily) are downloaded once and
    cached by temporal resolution under a single shared directory, regardless
    of the spatial domain requested.  Spatial clipping to the requested
    geometry is applied at load time.

    Parameters
    ----------
    temporal_resolution : str, optional
        One of ``'daily'``, ``'8day'``, ``'monthly'``, ``'yearly'``.
        Default is ``'monthly'``.
    """

    #: Data source URL (base).
    SOURCE = 'https://earlywarning.usgs.gov/ssebop'

    def __init__(self, temporal_resolution: str = 'monthly'):
        if temporal_resolution not in _TEMPORAL_CONFIGS:
            raise ValueError(
                f"temporal_resolution must be one of {list(_TEMPORAL_CONFIGS)}, "
                f"got {temporal_resolution!r}")
        self._temporal_resolution = temporal_resolution

        attrs = ManagerAttributes(
            category='evapotranspiration',
            product=f'SSEBop ET {temporal_resolution}',
            source='USGS Early Warning',
            description='SSEBop actual evapotranspiration for CONUS at daily/8-day/monthly/yearly resolution.',
            product_short=f'SSEBop_{temporal_resolution}',
            source_short='usgs_ssebop',
            url='https://earlywarning.usgs.gov/ssebop',
            license='public domain',
            citation='Senay et al. 2013',
            native_crs_in=watershed_workflow.crs.from_epsg(4326),
            native_crs_out=watershed_workflow.crs.from_epsg(4326),
            native_resolution=0.009652,
            native_start=cftime.datetime(2000, 1, 1, calendar='standard'),
            native_end=cftime.datetime(2023, 12, 31, calendar='standard'),
        )
        super().__init__(attrs)

    # ------------------------------------------------------------------
    # Cache directory management
    # ------------------------------------------------------------------

    def _folder(self) -> str:
        """Return the shared cache directory for this temporal resolution."""
        return os.path.join(cache_info.cacheFolder(self.attrs), self._temporal_resolution)

    def _filepath(self, date: cftime.datetime) -> str:
        """Return the full path for the cached GeoTIFF for *date*."""
        return os.path.join(self._folder(), f'{self._dateToFileStem(date)}.tif')

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request
                        ) -> manager_dataset.ManagerDataset.Request:
        """Return the request unchanged — no async server step."""
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — SSEBop downloads are synchronous."""
        return True

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Download all required time-slice GeoTIFFs if not already cached.

        Each file is saved as a full CONUS GeoTIFF under the shared cache
        directory.  Files that already exist on disk are skipped.

        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request storing start, end, and geometry.
        """
        os.makedirs(self._folder(), exist_ok=True)
        for date in self._enumerateDates(request.start, request.end):
            filepath = self._filepath(date)
            if os.path.isfile(filepath):
                logging.info(f'  SSEBop: using cached {filepath}')
                continue
            url = self._buildURL(date)
            logging.info(f'  SSEBop: downloading {url}')
            import requests as _requests
            resp = _requests.get(url, timeout=120,
                                 verify=source_utils.getVerifyOption())
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                tif_names = [n for n in zf.namelist() if n.lower().endswith('.tif')]
                if not tif_names:
                    raise RuntimeError(
                        f'No .tif found in ZIP from {url}; contents: {zf.namelist()}')
                with zf.open(tif_names[0]) as src, open(filepath, 'wb') as dst:
                    dst.write(src.read())
            logging.info(f'  SSEBop: saved {filepath}')

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Open cached GeoTIFFs, clip to request bounds, and stack on time.

        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request storing start, end, and geometry.

        Returns
        -------
        xr.Dataset
            Dataset with variable ``ET`` [mm/day] and a ``time`` dimension.
        """
        dates = self._enumerateDates(request.start, request.end)
        bounds = request.geometry.bounds

        arrays = []
        time_coords = []
        for date in dates:
            filepath = self._filepath(date)
            da = rioxarray.open_rasterio(filepath, masked=False, cache=False)
            da = da.squeeze('band', drop=True)
            da = da.rio.clip_box(*bounds)
            arrays.append(da)
            time_coords.append(date)

        stacked = xr.concat(arrays, dim='time')
        stacked = stacked.assign_coords(time=time_coords)
        stacked.name = 'ET'
        ds = stacked.to_dataset()
        ds = ds.rio.write_crs(self.native_crs_out)
        return ds

    def _postprocessDataset(self, request, dataset):
        """Apply NoData mask and scale factor, then delegate to base class.

        Parameters
        ----------
        request : ManagerDataset.Request
        dataset : xr.Dataset
            Raw dataset from _loadDataset.

        Returns
        -------
        xr.Dataset
        """
        da = dataset['ET'].astype('float32')
        da = da.where(da != _NODATA) * _SCALE
        da.attrs['units'] = 'mm/day'
        da.attrs['long_name'] = 'Actual evapotranspiration'
        dataset['ET'] = da
        return super()._postprocessDataset(request, dataset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enumerateDates(self, start: cftime.datetime,
                        end: cftime.datetime) -> list:
        """Return the list of canonical time-step dates in [start, end].

        Parameters
        ----------
        start : cftime.datetime
        end : cftime.datetime

        Returns
        -------
        list of cftime.datetime
        """
        cal = self.native_calendar
        dates = []

        if self._temporal_resolution == 'daily':
            current = datetime.date(start.year, start.month, start.day)
            end_d = datetime.date(end.year, end.month, end.day)
            step = datetime.timedelta(days=1)
            while current <= end_d:
                dates.append(cftime.datetime(current.year, current.month,
                                             current.day, calendar=cal))
                current += step

        elif self._temporal_resolution == '8day':
            # MODIS 8-day composites start on day 1 of each year, stepping by 8.
            start_d = datetime.date(start.year, start.month, start.day)
            end_d = datetime.date(end.year, end.month, end.day)
            for year in range(start.year, end.year + 1):
                jan1 = datetime.date(year, 1, 1)
                for i in range(46):  # at most 46 8-day periods per year
                    d = jan1 + datetime.timedelta(days=8 * i)
                    if d.year != year:
                        break
                    if d > end_d:
                        break
                    if d >= start_d:
                        dates.append(cftime.datetime(d.year, d.month, d.day,
                                                     calendar=cal))

        elif self._temporal_resolution == 'monthly':
            year, month = start.year, start.month
            end_year, end_month = end.year, end.month
            while (year, month) <= (end_year, end_month):
                dates.append(cftime.datetime(year, month, 1, calendar=cal))
                month += 1
                if month > 12:
                    month = 1
                    year += 1

        elif self._temporal_resolution == 'yearly':
            for year in range(start.year, end.year + 1):
                dates.append(cftime.datetime(year, 1, 1, calendar=cal))

        return dates

    def _dateToFileStem(self, date: cftime.datetime) -> str:
        """Return the base filename stem (without extension) for *date*.

        Parameters
        ----------
        date : cftime.datetime

        Returns
        -------
        str
        """
        _, stem_fn = _TEMPORAL_CONFIGS[self._temporal_resolution]
        std_date = datetime.date(date.year, date.month, date.day)
        return stem_fn(std_date)

    def _buildURL(self, date: cftime.datetime) -> str:
        """Return the full download URL for the ZIP file containing *date*.

        Parameters
        ----------
        date : cftime.datetime

        Returns
        -------
        str
        """
        res_dir, _ = _TEMPORAL_CONFIGS[self._temporal_resolution]
        stem = self._dateToFileStem(date)
        return _BASE_URL.format(res_dir=res_dir, stem=stem)
