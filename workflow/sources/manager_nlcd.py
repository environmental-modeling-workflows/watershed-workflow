"""Manager for interacting with NED datasets.
"""
import os,sys
import logging
import numpy as np
import shapely
import rasterio.merge
import requests
import requests.exceptions


import workflow.sources.utils as source_utils
import workflow.conf
import workflow.warp
import workflow.sources.names



class FileManagerNLCD:
    def __init__(self, layer='Land_Cover', year=None, location='L48'):
        self.layer, self.year, self.location = self.validate_input(layer, year, location)
        
        self.layer_name = 'NLCD_{1}_{0}_{2}'.format(self.layer, self.year, self.location)
        self.name = 'National Land Cover Database (NLCD) {}'.format(layer)
        self.file_format = 'geotiff'
        self.names = workflow.sources.names.Names(self.name, 'land_cover', None,
                                                  'NLCD_Land_Cover_epsg{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.tif')

    def validate_input(self, layer, year, location):
        valid_layers = ['Land_Cover', 'Imperviousness']
        if layer not in valid_layers:
            raise ValueError('NLCD invalid layer "{}" requested, valid are: {}'.format(layer, valid_layers))

        valid_locations = ['L48', 'AK', 'HI', 'PR']
        if location not in valid_locations:
            raise ValueError('NLCD invalid location "{}" requested, valid are: {}'.format(location, valid_locations))

        valid_years = {'L48': [2016, 2013, 2011, 2008, 2006, 2004, 2001],
                       'AK': [2011,2001],
                       'HI': [2001,],
                       'PR': [2001,],
                       }
        if year is None:
            year = valid_years[location][0]
        else:
            if year not in valid_years[location]:
                raise ValueError('NLCD invalid year "{}" requested for location {}, valid are: {}'.format(year, location, valid_years[location]))

        return layer, year, location
        

    def get_raster(self, shape, crs):
        """Download and read a DEM for this shape, clipping to the shape."""
        # get shape as a shapely, single Polygon
        if type(shape) is dict:
            shply = workflow.utils.shply(shape['geometry'])
        if type(shape) is shapely.geometry.MultiPolygon:
            shply = shapely.ops.cascaded_union(shape)

        # warp to lat-lon
        shply = workflow.warp.warp_shapely(shply, crs, workflow.conf.latlon_crs())

        # get the bounds and download
        bounds = shply.bounds
        fname = self.download(bounds, workflow.conf.latlon_crs())

        with rasterio.open(fname, 'r') as fid:
            profile = fid.profile
            band = fid.read(1)
        return profile,band

    def request(self, bounds, crs):
        """Forms the REST API get to find URLs."""
        if crs == workflow.conf.latlon_crs():
            res = 0.0003
        else:
            res = 30

        feather_bounds = list(bounds[:])
        feather_bounds[0] = feather_bounds[0] - 100*res
        feather_bounds[1] = feather_bounds[1] - 100*res
        feather_bounds[2] = feather_bounds[2] + 100*res
        feather_bounds[3] = feather_bounds[3] + 100*res

        # NLCD requires width and height
        # Guess at this assuming 30m is ~.00003 degrees, at least in north america...
        height = int(np.round((feather_bounds[3] - feather_bounds[1]) / res))
        width = int(np.round((feather_bounds[2] - feather_bounds[0]) / res))
        print('  image size: ({},{})'.format(width, height))
        rest_bounds = ','.join(str(b) for b in feather_bounds)
        print('  rest bounds: {}'.format(rest_bounds))
        try:
            rest_url = 'https://www.mrlc.gov/geoserver/mrlc_display/NLCD_{}_{}_{}/wms'.format(self.year, self.layer, self.location)
            r = requests.get(rest_url, params={'service':'WMS',
                                               'request':'GetMap',
                                               'layers':self.layer_name,
                                               'width':width,
                                               'height':height,
                                               'bbox':rest_bounds,
                                               'format':'image/geotiff',
                                               'crs':crs['init']})
        except requests.exceptions.ConnectionError as err:
            logging.error('{}: Failed to access REST API for NED DEM products.'.format(self.name))
            raise err

        r.raise_for_status()
        return r

    def download(self, bounds, crs, force=False):
        """Download the files, returning list of filenames."""
        logging.info("Collecting images for bounds: {} in crs: {}".format(bounds, crs['init']))
        
        # check directory structure
        os.makedirs(self.names.data_dir(), exist_ok=True)
        os.makedirs(self.names.raw_folder_name(), exist_ok=True)

        filename = self.names.file_name(crs['init'][5:], *bounds)
        print('  filename: {}'.format(filename))
        if not os.path.exists(filename) or force:
            r = self.request(bounds, crs)
            with open(filename, 'wb') as fid:
                fid.write(r.content)
        return filename
        


