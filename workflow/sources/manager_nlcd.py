"""Manager for interacting with NLCD datasets."""
import os,sys
import logging
import numpy as np
import shapely
import rasterio
import rasterio.windows
import rasterio.transform

import workflow.sources.utils as source_utils
import workflow.conf
import workflow.warp
import workflow.sources.names

# No API for getting NLCD locally -- must download the whole thing.
urls = {'NLCD_2016_Land_Cover_L48': 'https://s3-us-west-2.amazonaws.com/mrlc/NLCD_2016_Land_Cover_L48_20190424.zip',
        }

class FileManagerNLCD:
    """National Land Cover Database provides a raster for indexed land cover types
    [NLCD]_.

    .. note:: NLCD does not provide an API for subsetting the data, so the
       first time this is used, it WILL result in a long download time as it
       grabs the big file.  After that it will be much faster as the file is
       already local.

    TODO: Labels and colors for these indices should get moved here, but
    currently reside in workflow.colors.

    Parameter
    ---------
    layer : str, optional
      Layer of interest.  Default is `"Land_Cover`", should also be one for at
      least imperviousness, maybe others?
    year : int, optional
      Year of dataset.  Defaults to the most current available at the location.
    location : str, optional
      Location code.  Default is `"L48`" (lower 48), valid include `"AK`"
      (Alaska), `"HI`" (Hawaii, and `"PR`" (Puerto Rico).

    .. [NLCD] https://www.mrlc.gov/

    """
    def __init__(self, layer='Land_Cover', year=None, location='L48'):
        self.layer, self.year, self.location = self.validate_input(layer, year, location)
        
        self.layer_name = 'NLCD_{1}_{0}_{2}'.format(self.layer, self.year, self.location)
        self.name = 'National Land Cover Database (NLCD) Layer: {}'.format(self.layer_name)
        self.names = workflow.sources.names.Names(self.name, 'land_cover', self.layer_name,
                                                  self.layer_name+'.img')

    def validate_input(self, layer, year, location):
        """Validates input to the __init__ method."""
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
        

    def get_raster(self, shply, crs):
        """Download and read a DEM for this shape, clipping to the shape.

        Parameters
        ----------
        shply : fiona or shapely shape
          Shape to provide bounds of the raster.
        crs : CRS
          CRS of the shape.

        Returns
        -------
        profile : rasterio profile
          Profile of the raster.
        raster : np.ndarray
          Array containing the elevation data.

        Note that the raster provided is in NLCD native CRS (which is in the
        rasterio profile), not the shape's CRS.
        """
        # get shape as a shapely, single Polygon
        if type(shply) is dict:
            shply = workflow.utils.shply(shply['geometry'])
        if type(shply) is shapely.geometry.MultiPolygon:
            shply = shapely.ops.cascaded_union(shply)

        # download (or hopefully don't) the file
        filename, nlcd_profile = self._download()
        
        
        logging.info('CRS: {}'.format(nlcd_profile['crs']))

        # warp to crs
        shply = workflow.warp.shply(shply, crs, workflow.crs.from_rasterio(nlcd_profile['crs']))

        # calculate a window
        bounds = shply.bounds
        offset_y, offset_x = rasterio.transform.rowcol(nlcd_profile['transform'], bounds[0], bounds[3])
        offset_x = max(0, offset_x - 10)
        offset_y = max(0, offset_y - 10)
        
        lr_y, lr_x = rasterio.transform.rowcol(nlcd_profile['transform'], bounds[2], bounds[1])
        nx, ny = nlcd_profile['width'], nlcd_profile['height']
        lr_x = min(lr_x + 10, nx)
        lr_y = min(lr_y + 10, ny)
        
        window = rasterio.windows.Window(offset_x, offset_y, lr_x - offset_x, lr_y - offset_y)
        with rasterio.open(filename, 'r') as fid:
            profile = fid.profile
            band = fid.read(1, window=window)

        # shift the profile by the offset
        profile['transform'] = profile['transform'] * profile['transform'].translation(offset_x, offset_y)

        return profile,band

    def _download(self, force=False):
        """Download the files, returning list of filenames."""
        # check directory structure
        os.makedirs(self.names.data_dir(), exist_ok=True)
        work_folder = self.names.raw_folder_name()
        os.makedirs(work_folder, exist_ok=True)

        filename = self.names.file_name()
        logging.debug('  filename: {}'.format(filename))
        if not os.path.exists(filename) or force:
            try:
                url = urls[self.layer_name]
            except KeyError:
                raise NotImplementedError('Not yet implemented (but trivial to add, just ask!): {}'.format(self.layer_name))

            logging.warning('Downloading NLCD dataset: {} -- this will take a long time, depending upon internet connection.'.format(self.layer_name))

            downloadfile = os.path.join(work_folder, url.split("/")[-1])
            if not os.path.exists(downloadfile) or force:
                logging.debug("Attempting to download source for target '%s'"%filename)
                source_utils.download(url, downloadfile)
            source_utils.unzip(downloadfile, work_folder)

            # hope we can find it?
            img_files = [f for f in os.listdir(work_folder) if f.endswith('.img')]
            assert(len(img_files) == 1)
            target = os.path.join(work_folder, img_files[0])
            os.rename(target, filename)
            os.rename(target[:-3]+'ige', filename[:-3]+'ige')

        with rasterio.open(filename, 'r') as fid:
            profile = fid.profile
        return filename, profile
        


