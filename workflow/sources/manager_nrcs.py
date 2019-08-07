"""Manager for interacting with National Resources Conservation Service Soil Survey database.
"""
import os, sys
import logging
import fiona
import shapely
import requests
import numpy as np

import workflow.sources.utils as source_utils
import workflow.conf
import workflow.sources.names
import workflow.warp
import workflow.utils

class FileManagerNRCS:
    def __init__(self):
        self.name = 'National Resources Conservation Service Soil Survey (NRCS Soils)'
        self.crs = fiona.crs.from_epsg('4326')
        self.fstring = '{:.4f}_{:.4f}_{:.4f}_{:.4f}'
        self.qstring = self.fstring.replace('_',',')
        self.name_manager = workflow.sources.names.Names(self.name,
                                                         'soil_survey',
                                                         '',
                                                         'soil_survey_shape_%s.gml'%self.fstring)
        self.url = 'https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDMWGS84Geographic.wfs'

    def get_shapes_in_bounds(self, bounds, bounds_crs):
        """Downloads and reads soil shapefiles."""
        bounds = self.bounds(bounds, bounds_crs)
        filename = self._download(bounds)

        with fiona.open(filename, 'r') as fid:
            shps = [workflow.utils.shply(shp['geometry'], shp['properties'], True) for shp in fid]
            ids = [shp['id'] for shp in fid]
            profile = fid.profile

        logging.info('  Found {} shapes.'.format(len(shps)))
        bnds = shapely.ops.cascaded_union(shps)
        logging.info('  With bounds: {}'.format(bnds.bounds))
        logging.info('  and crs: {}'.format(profile['crs']))
        return profile, shps, ids

    def bounds(self, b, bounds_crs):
        b = workflow.warp.warp_bounds(b, bounds_crs, self.crs)
        b = [np.round(b[0],4)-.0001, np.round(b[1],4)-.0001,
                  np.round(b[2],4)+.0001, np.round(b[3],4)+.0001]
        return b
        
        

    def _download(self, bounds, force=False):
        """Downloads the data and writes it to disk."""
        os.makedirs(self.name_manager.data_dir(), exist_ok=True)
        filename = self.name_manager.file_name(*bounds)
        logging.info('  Using filename: {}'.format(filename))

        if not os.path.exists(filename) or force:
            logging.info('  Downloading via request.')
            params = {'REQUEST':'GetFeature',
                      'TYPENAME':'MapunitPoly',
                      'BBOX':self.qstring.format(*bounds)}
            r = requests.get(self.url, params=params)
            r.raise_for_status()

            with open(filename, 'w') as fid:
                fid.write(r.text)

        return filename

    
