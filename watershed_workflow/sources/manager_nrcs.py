"""National Resources Conservation Service Soil Survey database.

Author: Ethan Coon (coonet@ornl.gov)
Author: Pin Shuai (pin.shuai@pnnl.gov)

"""
from typing import Optional, List, Tuple

import os
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import collections
import shapely.wkt
import shapely.geometry

import watershed_workflow.crs
from watershed_workflow.crs import CRS
import watershed_workflow.sources.names
import watershed_workflow.warp
import watershed_workflow.utils
import watershed_workflow.soil_properties
import watershed_workflow.sources.utils as source_utils
import watershed_workflow.sources.standard_names as snames

_query_template_props = """
SELECT
 saversion, saverest, l.areasymbol, l.areaname, l.lkey, musym, muname, museq, mu.mukey, comppct_r, compname, localphase, slope_r, c.cokey, hzdept_r, hzdepb_r, ksat_r, sandtotal_r, claytotal_r, silttotal_r, dbthirdbar_r , partdensity, wsatiated_r, ch.chkey

FROM sacatalog sac
  INNER JOIN legend l ON l.areasymbol = sac.areasymbol
    INNER JOIN mapunit mu ON mu.lkey = l.lkey
    AND mu.mukey IN
({})
  LEFT OUTER JOIN component c ON c.mukey = mu.mukey
LEFT OUTER JOIN chorizon ch ON ch.cokey = c.cokey
"""

_query_template_shapes = """
~DeclareGeometry(@aoiGeom)~
~DeclareGeometry(@aoiGeomFixed)~
CREATE TABLE #AoiTable
    ( aoiid INT IDENTITY (1,1),
    landunit VARCHAR(20),
    aoigeom GEOMETRY )
;

SELECT @aoiGeom = GEOMETRY::STGeomFromText('{}', 4326);
SELECT @aoiGeomFixed = @aoiGeom.MakeValid().STUnion(@aoiGeom.STStartPoint());
INSERT INTO #AoiTable ( landunit, aoigeom )
VALUES ('My Geom', @aoiGeomFixed);
-- End of AOI geometry section
-- #AoiSoils table contains intersected soil polygon table with geometry
CREATE TABLE #AoiSoils
    ( polyid INT IDENTITY (1,1),
    aoiid INT,
    landunit VARCHAR(20),
    mukey INT,
    soilgeom GEOMETRY )
;
-- End of CREATE TABLE section
-- Populate #AoiSoils table with intersected soil polygon geometry
INSERT INTO #AoiSoils (aoiid, landunit, mukey, soilgeom)
    SELECT A.aoiid, A.landunit, M.mukey, M.mupolygongeo.STIntersection(A.aoigeom ) AS soilgeom
    FROM mupolygon M, #AoiTable A
    WHERE mupolygongeo.STIntersects(A.aoigeom) = 1
;
-- #AoiSoils2 is single part polygon
SELECT landunit, I.mukey, musym, soilgeom.STGeometryN(Numbers.n).MakeValid() AS wktgeom
FROM #AoiSoils AS I
JOIN Numbers ON Numbers.n <= I.soilgeom.STNumGeometries()
INNER JOIN mapunit M ON I.mukey = M.mukey
;
-- END OF AOI GEOMETRY QUERY
"""


def synthesizeData(df):
    """Renames and calculates derived properties in a MUKEY-based data frame, in place"""
    # rename columns to improve readability
    rename_list = {
        'hzdept_r': 'top depth [cm]',
        'hzdepb_r': 'bot depth [cm]',
        'ksat_r': 'log Ksat [um s^-1]',
        'sandtotal_r': 'total sand pct [%]',
        'silttotal_r': 'total silt pct [%]',
        'claytotal_r': 'total clay pct [%]',
        'dbthirdbar_r': 'bulk density [g/cm^3]',
        'partdensity': 'particle density [g/cm^3]',
        'wsatiated_r': 'saturated water content [%]',
        'comppct_r': 'component pct [%]'
    }

    df.rename(columns=rename_list, inplace=True)

    # preprocess data
    df['thickness [cm]'] = df['bot depth [cm]'] - df['top depth [cm]']
    df['porosity [-]'] = 1 - df['bulk density [g/cm^3]'] / df['particle density [g/cm^3]']

    # log Ksat to average in log space, setting a min perm of 1e-16 m^2 (1.e-3 um/s)
    df['log Ksat [um s^-1]'] = np.log10(np.maximum(df['log Ksat [um s^-1]'].values, 1.e-3))

    # assume null porosity = saturated water content
    df.loc[pd.isnull(df['porosity [-]']), 'porosity [-]'] = \
        df.loc[pd.isnull(df['porosity [-]']), 'saturated water content [%]']/100
    logging.info(f'found {len(df["mukey"].unique())} unique MUKEYs.')


def aggregateComponentValues(df : gpd.GeoDataFrame,
                             agg_var : Optional[List[str]] = None):
    """
    Aggregate horizon value by layer thickness to get component property.

    Parameters
    ----------
    df_chorizon : pandas dataframe
      horizon table
    imukey_df : pandas dataframe
      individual mukey df contains component keys
    agg_var : list
      variables to average

    Returns
    -------
    df_comp : pandas dataframe
      aggregated component properties
    """
    if agg_var is None:
        agg_var = [
            'log Ksat [um s^-1]', 'porosity [-]', 'bulk density [g/cm^3]', 'total sand pct [%]',
            'total silt pct [%]', 'total clay pct [%]',
        ]

    comp_list = ['mukey', 'cokey', 'component pct [%]', 'thickness [cm]'] + agg_var
    horizon_selected_cols = [
        'mukey', 'cokey', 'chkey', 'component pct [%]', 'thickness [cm]', 'top depth [cm]',
        'bot depth [cm]', 'particle density [g/cm^3]',
    ] + agg_var

    df_comp = pd.DataFrame(columns=comp_list)
    cokeys = df['cokey'].unique()

    rows = []
    for icokey in cokeys:
        idf_horizon = df.loc[df['cokey'] == icokey, horizon_selected_cols]

        imukey = idf_horizon['mukey'].values[0]
        assert (np.all(idf_horizon['mukey'].values == imukey))
        icomp_pct = idf_horizon['component pct [%]'].values[0]
        assert (np.all(idf_horizon['component pct [%]'].values == icomp_pct))

        depth_agg_value = []
        # horizon-uniform quantities
        depth_agg_value.append(imukey)
        depth_agg_value.append(icokey)
        depth_agg_value.append(icomp_pct)
        depth_agg_value.append(None)  # thickness placeholder

        # depth-average quantities
        for ivar in agg_var:
            idf = idf_horizon[['thickness [cm]', ivar]].dropna()
            if idf.empty:
                ivalue = np.nan
            else:
                ivalue = sum(idf['thickness [cm]'] / idf['thickness [cm]'].sum() * idf[ivar])
            depth_agg_value.append(ivalue)

        idepth = idf_horizon['bot depth [cm]'].dropna().max()
        depth_agg_value[3] = idepth

        # create the local df
        assert (len(depth_agg_value) == len(comp_list))
        idf_comp = pd.DataFrame(np.array(depth_agg_value).reshape(1, len(depth_agg_value)),
                                    columns=comp_list)

        # normalize sand/silt/clay pct to make the sum(%sand, %silt, %clay)=1
        sum_soil = idf_comp.loc[:, ['total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]']].sum().sum()
        if sum_soil != 100:
            for isoil in ['total sand pct [%]', 'total silt pct [%]', 'total clay pct [%]']:
                idf_comp[isoil] = idf_comp[isoil] / sum_soil * 100

        # append to df
        rows.append(idf_comp)

    return pd.concat(rows)


def aggregateMukeyValues(df : gpd.GeoDataFrame,
                         agg_var : Optional[List[str]] =None):
    """Aggregate component values by component percentage to get MUKEY property.

    Parameters
    ----------
    df : pandas dataframe
      The list of horizons in all components in all mukeys.
    agg_var : list(str), optional
      List of keys to aggregate.  Defaults to normal things from
      SSURGO.

    Returns
    -------
    df_mukey : pandas dataframe
      aggregated mukey properties

    """
    # variables to average across component and horizon. These are hard coded for now.
    if agg_var is None:
        agg_var = [
            'log Ksat [um s^-1]', 'porosity [-]', 'bulk density [g/cm^3]', 'total sand pct [%]',
            'total silt pct [%]', 'total clay pct [%]',
        ]

    # aggregate all horizons to components
    df_comp = aggregateComponentValues(df, agg_var)

    # area-average to mukey
    area_agg_var = ['thickness [cm]', ] + agg_var
    mukey_agg_var = ['mukey', ] + area_agg_var

    df_mukey_rows = []
    for imukey in df_comp['mukey'].unique()[:]:
        idf_comp = df_comp.loc[df_comp['mukey'] == imukey]
        area_agg_value = []

        # component-uniform quantities
        area_agg_value.append(imukey)

        # area-average quantities
        for ivar in area_agg_var:
            idf = idf_comp[['component pct [%]', ivar]].dropna()
            if idf.empty:
                ivalue = np.nan
            else:
                ivalue = sum(idf['component pct [%]'] / idf['component pct [%]'].sum() * idf[ivar])
            area_agg_value.append(ivalue)

        # create the local data frame and append
        assert (len(area_agg_value) == len(mukey_agg_var))
        idf_mukey = pd.DataFrame(np.array(area_agg_value).reshape(1, len(area_agg_value)),
                                     columns=mukey_agg_var)
        df_mukey_rows.append(idf_mukey)

    df_mukey = pd.concat(df_mukey_rows)
    df_mukey['mukey'] = np.array(df_mukey['mukey'], dtype=int)
    return df_mukey


class ManagerNRCS:
    """The National Resources Conservation Service's SSURGO Database [NRCS]_
    contains a huge amount of information about soil texture, parameters, and
    structure, and are provided as shape files containing soil type
    delineations with map-unit-keys (MUKEYs).  These are re-broadcast onto a
    raster (much like gSSURGO, which is unfortunately not readable by open
    tools) and used to index soil parameterizations for simulation.

    Data is accessed via two web APIs -- the first for spatial
    (shapefiles) survey information, the second for properties.

    * https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs
    * https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest

    TODO: Functionality for mapping from MUKEY to soil parameters.

    .. [NRCS] Soil Survey Staff, Natural Resources Conservation
       Service, United States Department of Agriculture. Web Soil
       Survey. Available online at
       https://websoilsurvey.nrcs.usda.gov/. Accessed
       [month/day/year].

    """
    def __init__(self):
        self.name = 'National Resources Conservation Service Soil Survey (NRCS Soils)'
        self.crs = watershed_workflow.crs.from_epsg('4326')
        self.fstring = '{:.4f}_{:.4f}_{:.4f}_{:.4f}'
        self.qstring = self.fstring.replace('_', ',')
        self.name_manager = watershed_workflow.sources.names.Names(
            self.name, os.path.join('soil_structure', 'SSURGO'), '', 'SSURGO_%s.shp' % self.fstring)

        #self.url_spatial = 'https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDMWGS84Geographic.wfs'
        #self.url_spatial = 'https://SDMDataAccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs'
        self.url_spatial = 'https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest'
        self.url_data = 'https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest'

        
    def _getShapesByGeometry(self,
                            geometry : shapely.geometry.base.BaseGeometry,
                            geometry_crs : CRS,
                            force_download : bool = False) -> gpd.GeoDataFrame:
        """Downloads and reads soil shapefiles.

        This accepts only a bounding box.

        Parameters
        ----------
        geometry : shapely.geometry.base.BaseGeometry
            Bounding box to filter shapes.
        crs : CRS
            Coordinate system of the bounding box.
        force_download : bool
          Download or re-download the file if true.

        Returns
        -------
        gpd.GeoDataFrame
        """
        bounds = self._cleanBounds(geometry, geometry_crs)
        filename = self._downloadShapes(bounds, force=force_download)
        shapes = gpd.read_file(filename)
        logging.info('  Found {} shapes.'.format(len(shapes)))
        return shapes

    def _downloadProperties(self, mukeys, filename=None, force=False):
        """Queries REST API for parameters by MUKEY."""
        if filename is None or (not os.path.exists(filename)) or force:
            logging.info(f'  Downloading raw properties data via request:')
            logging.info(f'    to file: {filename}')
            logging.info(f'       from: {self.url_data}')

            mukey_data_headers = [
                'saversion', 'saverest', 'areasymbol', 'areaname', 'lkey', 'musym', 'muname',
                'museq', 'mukey', 'comppct_r', 'compname', 'localphase', 'slope_r', 'cokey',
                'hzdept_r', 'hzdepb_r', 'ksat_r', 'sandtotal_r', 'claytotal_r', 'silttotal_r',
                'dbthirdbar_r', 'partdensity', 'wsatiated_r', 'chkey'
            ]
            mukey_dtypes = [
                int, str, str, str, int, str, str, int, int, int, str, str, float, int, float,
                float, float, float, float, float, float, float, float, int
            ]

            mukey_list_string = ','.join([f"'{k}'" for k in mukeys])
            query = _query_template_props.format(mukey_list_string)

            data = { 'FORMAT': 'JSON', 'QUERY': query }
            r = requests.post(self.url_data, data=data, verify=source_utils.getVerifyOption())
            logging.info(f'  full URL: {r.url}')
            r.raise_for_status()

            table = np.array(r.json()['Table'])
            df = pd.DataFrame()

            def to_type(val, dtype):
                if val is None:
                    if dtype == int:
                        return -1
                    elif dtype == float:
                        return np.nan
                    elif dtype == str:
                        return ''
                return dtype(val)

            for i, (title, dtype) in enumerate(zip(mukey_data_headers, mukey_dtypes)):
                df[title] = np.array([to_type(entry, dtype) for entry in table[:, i]], dtype=dtype)

            if filename is not None:
                df.to_csv(filename)

        else:
            df = pd.read_csv(filename)
        return df

    def _getProperties(self, mukeys, filename=None, force_download=False):
        """Download and aggregate properties for a given set of mukeys, storing raw data in filename."""
        df = self._downloadProperties(mukeys, filename, force_download)

        synthesizeData(df)
        df_agg = aggregateMukeyValues(df)

        # get dataframe with van genuchten models
        df_vgm = watershed_workflow.soil_properties.computeVanGenuchtenModelFromSSURGO(df_agg)

        # fix units
        df_ats = watershed_workflow.soil_properties.convertRosettaToATS(df_vgm)
        # -- convert thickness to [m]
        df_ats['thickness [cm]'] = df_ats['thickness [cm]'] / 100.
        df_ats.rename(columns={'thickness [cm]' : 'thickness [m]'}, inplace=True)

        # all structure data frames are expected to have a 'source' and an 'id' in that source field
        df_ats['source'] = 'NRCS'

        # resort the dataframe to have the same order as was passed in
        # in the list mukeys.  This alleviates potential mistakes if
        # they get zipped or otherwise iterated over together.
        df_ats.set_index('mukey', inplace=True)
        df_ats = df_ats.reindex(mukeys)
        df_ats.reset_index(inplace=True)
        return df_ats

    
    def getShapesByGeometry(self,
                            geometry : shapely.geometry.base.BaseGeometry,
                            geometry_crs : CRS,
                            force_download : bool = False) -> gpd.GeoDataFrame:
        """Downloads and reads soil shapefiles, and aggregates SSURGO data onto MUKEYS

        Accepts either a bounding box, shape, or list of shapes.

        Parameters
        ----------
        shapes : shply, list(shply), or [xmin, ymin, xmax, ymax]
            Shapes on which to run the query.
        crs : CRS
            Coordinate system of the bounding box.
        force_download : bool
          Download or re-download the file if true.

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list
            List of fiona shapes that match the index or bounds.
        properties : pandas dataframe
            Dataframe of data by mukey = shape['id']
        """
        bounds = self._cleanBounds(geometry, geometry_crs)
        filename = self.name_manager.file_name(*bounds)

        try:
            shapes = self._getShapesByGeometry(geometry, geometry_crs, force_download)
        except requests.HTTPError as e:
            if '500 Server Error' in str(e):
                if isinstance(geometry, shapely.geometry.MultiPolygon):
                    shape_dfs = [self.getShapesByGeometry(poly, geometry_crs, force_download)[['mukey', 'geometry']]
                                 for poly in geometry]
                    shapes = pd.concat(shape_dfs)
                else:
                    raise ValueError(
                        'NRCS.getShapesByGeometry() called with too large of a shape for NRCS servers.  '
                        'Trying splitting it into smaller polygons and trying again.'
                    )
            else:
                raise e

        # Group by ID and union geometries
        shapes_no_crs = shapes.groupby('mukey')['geometry'].apply(
            lambda x: shapely.ops.unary_union(x.tolist()) if len(x) > 1 else x.iloc[0]
        ).reset_index()
        shapes = shapes_no_crs.write_crs(shapes.crs)
        assert shapes.crs is not None

        # read the properties
        data_filename = filename[:-4] + "_properties.csv"
        props = self._getProperties(shapes['mukey'], data_filename, force_download)

        # merge properties and shapes dfs
        df = shapes.merge(props, on='mukey')
        df[snames.ID] = df['mukey']
        return df


    def _cleanBounds(self, 
                     geometry : shapely.geometry.base.BaseGeometry,
                     geometry_crs : CRS,
                     buffer : float = 0.0001) -> Tuple[float,float,float,float]:
        """Compute bounds in the required CRS from a polygon or bounds in a given crs"""
        bounds_ll = watershed_workflow.warp.shply(geometry, geometry_crs,
                                                  watershed_workflow.crs.latlon_crs).bounds
        feather_bounds = list(bounds_ll[:])
        feather_bounds[0] = np.round(feather_bounds[0] - buffer, 4)
        feather_bounds[1] = np.round(feather_bounds[1] - buffer, 4)
        feather_bounds[2] = np.round(feather_bounds[2] + buffer, 4)
        feather_bounds[3] = np.round(feather_bounds[3] + buffer, 4)
        return tuple(feather_bounds)

    
    def _downloadShapes(self,
                  bounds : Tuple[float,float,float,float],
                  force : bool = False) -> str:
        """Downloads the data and writes it to disk."""
        os.makedirs(self.name_manager.data_dir(), exist_ok=True)
        filename = self.name_manager.file_name(*bounds)
        logging.info("Attempting to download source for target '%s'" % filename)

        if not os.path.exists(filename) or force:
            logging.info('  Downloading spatial data via request:')
            logging.info(f'    to file: {filename}')
            logging.info(f'       from: {self.url_spatial}')

            box = shapely.geometry.box(*bounds)
            query = _query_template_shapes.format(box.wkt)
            data = { 'FORMAT': 'JSON', 'QUERY': query }
            r = requests.post(self.url_data, data=data, verify=source_utils.getVerifyOption())
            logging.info(f'  full URL: {r.url}')
            r.raise_for_status()

            logging.info(f'  Converting to shapely')
            table = r.json()['Table']

            shps = [shapely.wkt.loads(ent[3]) for ent in table]
            df = gpd.GeoDataFrame({'mukey':[int(t[1]) for t in table]},
                                         geometry=shps, crs=self.crs)
            df.to_file(filename)
        return filename
