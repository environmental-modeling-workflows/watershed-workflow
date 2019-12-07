"""National Resources Conservation Service Soil Survey database."""
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
    """The National Resources Conservation Service's SSURGO Database [NRCS]_
    contains a huge amount of information about soil texture, parameters, and
    structure, and are provided as shape files containing soil type
    delineations with map-unit-keys (MUKEYs).  These are re-broadcast onto a
    raster (much like gSSURGO, which is unfortunately not readable by open
    tools) and used to index soil parameterizations for simulation.

    TODO: Functionality for mapping from MUKEY to soil parameters.

    .. [NRCS] https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/?cid=nrcs142p2_053627
    """
    def __init__(self):
        self.name = 'National Resources Conservation Service Soil Survey (NRCS Soils)'
        self.crs = workflow.crs.from_epsg('4326')
        self.fstring = '{:.4f}_{:.4f}_{:.4f}_{:.4f}'
        self.qstring = self.fstring.replace('_',',')
        self.name_manager = workflow.sources.names.Names(self.name,
                                                         'soil_survey',
                                                         '',
                                                         'soil_survey_shape_%s.gml'%self.fstring)
        self.url = 'https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDMWGS84Geographic.wfs'

    def get_shapes(self, bounds, bounds_crs):
        """Downloads and reads soil shapefiles.

        This accepts only a bounding box.  

        Parameters
        ----------
        bounds : [xmin, ymin, xmax, ymax]
            Bounding box to filter shapes.
        crs : CRS
            Coordinate system of the bounding box.

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list
            List of fiona shapes that match the index or bounds.
        """
        if type(bounds) is int:
            raise TypeError('NRCS file manager only handles bounds, not indices.')
            
        bounds = self.bounds(bounds, bounds_crs)
        filename = self._download(bounds)

        def _flip(shp):
            """Generate a new fiona shape in long-lat from one in lat-long"""
            for ring in workflow.utils.generate_rings(shp):
                for i,c in enumerate(ring):
                    ring[i] = c[1],c[0]
            return shp
        
        with fiona.open(filename, 'r') as fid:
            profile = fid.profile
            shapes = [_flip(s) for s in fid]

        for s in shapes:
            s['properties']['id'] = s['id']

        logging.info('  Found {} shapes.'.format(len(shapes)))
        logging.info('  and crs: {}'.format(profile['crs']))
        return profile, shapes

    def bounds(self, b, bounds_crs):
        """Create a bounds in the NRCS coordinate system for use in downloading."""
        b = workflow.warp.bounds(b, bounds_crs, self.crs)
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

    
