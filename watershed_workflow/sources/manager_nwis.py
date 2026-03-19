"""USGS National Water Information System (NWIS) stream gage manager.

Provides access to USGS stream gage locations and metadata via the
NWIS web services, as well as streamflow time series retrieval.
"""

import os
import logging
from typing import Optional, List

import geopandas as gpd
import pandas as pd

import watershed_workflow.crs
from watershed_workflow.crs import CRS

from . import standard_names as names
from . import manager_shapes
from .manager import ManagerAttributes
from . import cache_info as ci


class ManagerNWIS(manager_shapes.ManagerShapes):
    """USGS National Water Information System (NWIS) stream gage data.

    Provides access to USGS stream gage site locations and metadata
    via the NWIS web services, using the ``pygeohydro`` library for
    data retrieval.  Supports spatial queries by bounding box and
    direct lookup by site number.

    Site locations are cached as GeoPackage files.  Streamflow
    retrieval (via :meth:`getStreamflow`) is not cached.

    Data source: https://waterservices.usgs.gov/nwis
    """

    def __init__(self,
                 site_type: str = 'ST',
                 has_data_type: str = 'dv',
                 force_download: bool = False):
        """Initialize the NWIS stream gage manager.

        Parameters
        ----------
        site_type : str, optional
            NWIS site type code.  Common values: ``'ST'`` (stream),
            ``'GW'`` (groundwater well), ``'LK'`` (lake/reservoir).
            Default is ``'ST'``.
        has_data_type : str, optional
            NWIS data type code controlling which sites are returned.
            Common values: ``'dv'`` (daily values), ``'iv'``
            (instantaneous values).  Default is ``'dv'``.  Used for
            both ``hasDataTypeCd`` and ``outputDataTypeCd`` in the
            NWIS query.  Sites are filtered to those measuring
            discharge (parameter code 00060) and gage height (00065).
        force_download : bool, optional
            If ``True``, bypass the cache and always re-download.
            Default is ``False``.
        """
        attrs = ManagerAttributes(
            category='observations',
            product='USGS NWIS',
            source='pygeohydro NWIS',
            description='USGS National Water Information System stream gage locations.',
            product_short='NWIS',
            source_short='nwis',
            url='https://waterservices.usgs.gov/nwis',
            license='public domain',
            citation='USGS NWIS',
            native_crs_in=watershed_workflow.crs.from_epsg(4326),
            native_resolution=0.001,
            native_id_field='site_no',
        )
        super().__init__(attrs)
        self.site_type = site_type
        self.has_data_type = has_data_type
        self.force_download = force_download

    def _getShapes(self) -> gpd.GeoDataFrame:
        """Fetch all shapes in a dataset.

        Returns
        -------
        gpd.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        raise NotImplementedError(
            'ManagerNWIS does not support getting all shapes. '
            'Use getShapesByGeometry or getShapesByID instead.'
        )

    def _download(self, query: dict) -> gpd.GeoDataFrame:
        """Call NWIS API via pygeohydro and return raw site GeoDataFrame."""
        import pygeohydro
        import pynhd
        nwis = pygeohydro.NWIS()
        try:
            df = nwis.get_info(query, fix_names=True)
        except pygeohydro.exceptions.ZeroMatchedError:
            logging.warning('  NWIS: no sites matched query — returning empty GeoDataFrame.')
            return gpd.GeoDataFrame(columns=['site_no', 'station_nm', 'geometry'],
                                    crs=self.native_crs_in)
        logging.info(f'  NWIS: retrieved {len(df)} gage sites.')

        return df

    def _getShapesByGeometry(self, geometry_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch NWIS gage sites within the bounding box of the given geometry.

        Parameters
        ----------
        geometry_gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in EPSG:4326 (already buffered and
            snapped by the base class).

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame of gage site locations with native NWIS column names.
        """
        xmin, ymin, xmax, ymax = geometry_gdf.total_bounds
        bbox_str = f'{xmin:.6f},{ymin:.6f},{xmax:.6f},{ymax:.6f}'
        query = {
            'bBox': bbox_str,
            'siteType': self.site_type,
            'parameterCd': '00060,00065',
            'hasDataTypeCd': self.has_data_type,
            'outputDataTypeCd': self.has_data_type,
        }

        cache_dir = ci.cacheDirname(self.attrs, tuple(geometry_gdf.total_bounds))
        filename = os.path.join(cache_dir, 'shapes.gpkg')

        if os.path.exists(filename) and not self.force_download:
            logging.info(f'  NWIS: reading from cache: {filename}')
            return gpd.read_file(filename)

        df = self._download(query)

        if len(df) == 0:
            logging.warning('  NWIS: no gage sites found for the requested bounding box.')
            return df

        os.makedirs(cache_dir, exist_ok=True)
        df.to_file(filename, driver='GPKG')
        return df

    def _getShapesByID(self, ids: List[str]) -> gpd.GeoDataFrame:
        """Fetch NWIS gage sites by site number.

        Parameters
        ----------
        ids : list of str
            List of USGS site numbers to retrieve.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame of gage site locations with native NWIS column names.
        """
        query = {'sites': ','.join(str(i) for i in ids)}
        return self._download(query)

    def _addStandardNames(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert native NWIS column names to standard names.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame with native NWIS column names.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with standard column names added or renamed.
        """
        if 'site_no' in df.columns:
            df[names.ID] = df['site_no'].astype('string')

        if 'station_nm' in df.columns:
            df[names.NAME] = df['station_nm']

        if 'drain_area_va' in df.columns:
            # Convert sq miles to km²
            df[names.DRAINAGE_AREA] = df['drain_area_va'] * 2.58999

        if 'huc_cd' in df.columns:
            df[names.HUC] = df['huc_cd']

        return df

    def getStreamflow(self,
                      df: gpd.GeoDataFrame,
                      dates: tuple,
                      freq: str = 'dv',
                      min_obs: int | None = None) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Retrieve streamflow time series for the given gage sites.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame of gage sites, as returned by
            :meth:`getShapesByGeometry` or :meth:`getShapesByID`.
            Must contain the ``names.ID`` column (USGS site numbers).
        dates : tuple of str
            Start and end dates as ``('YYYY-MM-DD', 'YYYY-MM-DD')``.
        freq : str, optional
            Retrieval frequency.  ``'dv'`` for daily values (default)
            or ``'iv'`` for instantaneous values.
        min_obs : int, optional
            Minimum number of non-NaN observations required to retain
            a gage.  Gages with fewer valid values are dropped from
            both outputs.  Default is to keep all gages.

        Returns
        -------
        gages : gpd.GeoDataFrame
            Subset of ``df`` containing only gages with at least
            ``min_obs`` valid observations.  ``len(gages)`` equals
            ``len(streamflow.columns)``.
        streamflow : pd.DataFrame
            DataFrame indexed by datetime with one column per retained
            gage, named ``'USGS-{site_no}'``, in cubic meters per
            second (cms).
        """
        import pygeohydro
        site_nos = df[names.ID].tolist()
        nwis = pygeohydro.NWIS()
        streamflow = nwis.get_streamflow(site_nos, dates, freq=freq)
        if min_obs is not None:
            streamflow = streamflow.dropna(axis='columns', thresh=min_obs)
        kept_ids = pd.Index([col.replace('USGS-', '') for col in streamflow.columns])
        gages = df[df[names.ID].isin(kept_ids)].drop_duplicates(subset=names.ID).copy()
        streamflow = streamflow[[f'USGS-{i}' for i in gages[names.ID]]]
        return gages, streamflow

    def addCOMIDs(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fetch NHDPlus COMIDs and reach measures for gage sites via NLDI.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            GeoDataFrame of gage sites as returned by
            :meth:`getShapesByGeometry` or :meth:`getShapesByID`.
            Must contain the ``names.ID`` column (bare USGS site numbers).

        Returns
        -------
        gpd.GeoDataFrame
            Copy of ``df`` with ``names.COMID`` and ``names.MEASURE``
            columns added.  Sites not found in NLDI will have ``NaN``.
            ``names.COMID`` is cast to ``Int64`` (nullable integer).
        """
        import pynhd
        nldi = pynhd.NLDI()
        nldi_ids = [f'USGS-{i}' for i in df[names.ID].tolist()]
        features = nldi.getfeature_byid('nwissite', nldi_ids)
        features = features.set_index(features['identifier'].str.replace('USGS-', '', regex=False))
        df = df.copy()
        df[names.COMID] = df[names.ID].map(features['comid']).astype('Int64')
        df[names.MEASURE] = df[names.ID].map(features['measure'])
        return df
