"""Manager for interacting with NLCD datasets."""
import os, sys
import logging
import numpy as np
import shapely
import rasterio
import rasterio.mask

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.config
import watershed_workflow.warp
import watershed_workflow.sources.names

# No API for getting NLCD locally -- must download the whole thing.
urls = {
    'NLCD_2016_Land_Cover_L48':
    'https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2016_land_cover_l48_20210604.zip'
    'NLCD_2019_Land_Cover_L48':
    'https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2019_land_cover_l48_20210604.zip'
}

colors = {
    0: ('None', (0.00000000000, 0.00000000000, 0.00000000000)),
    11: ('Open Water', (0.27843137255, 0.41960784314, 0.62745098039)),
    12: ('Perrenial Ice/Snow', (0.81960784314, 0.86666666667, 0.97647058824)),
    21: ('Developed, Open Space', (0.86666666667, 0.78823529412, 0.78823529412)),
    22: ('Developed, Low Intensity', (0.84705882353, 0.57647058824, 0.50980392157)),
    23: ('Developed, Medium Intensity', (0.92941176471, 0.00000000000, 0.00000000000)),
    24: ('Developed, High Intensity', (0.66666666667, 0.00000000000, 0.00000000000)),
    31: ('Barren Land', (0.69803921569, 0.67843137255, 0.63921568628)),
    41: ('Deciduous Forest', (0.40784313726, 0.66666666667, 0.38823529412)),
    42: ('Evergreen Forest', (0.10980392157, 0.38823529412, 0.18823529412)),
    43: ('Mixed Forest', (0.70980392157, 0.78823529412, 0.55686274510)),
    51: ('Dwarf Scrub', (0.64705882353, 0.54901960784, 0.18823529412)),
    52: ('Shrub/Scrub', (0.80000000000, 0.72941176471, 0.48627450980)),
    71: ('Grassland/Herbaceous', (0.88627450980, 0.88627450980, 0.75686274510)),
    72: ('Sedge/Herbaceous', (0.78823529412, 0.78823529412, 0.46666666667)),
    73: ('Lichens', (0.60000000000, 0.75686274510, 0.27843137255)),
    74: ('Moss', (0.46666666667, 0.67843137255, 0.57647058824)),
    81: ('Pasture/Hay', (0.85882352941, 0.84705882353, 0.23921568628)),
    82: ('Cultivated Crops', (0.66666666667, 0.43921568628, 0.15686274510)),
    90: ('Woody Wetlands', (0.72941176471, 0.84705882353, 0.91764705882)),
    95: ('Emergent Herbaceous Wetlands', (0.43921568628, 0.63921568628, 0.72941176471)),
}

indices = dict([(pars[0], id) for (id, pars) in colors.items()])


class FileManagerNLCD:
    """National Land Cover Database provides a raster for indexed land cover types
    [NLCD]_.

    .. note:: NLCD does not provide an API for subsetting the data, so the
       first time this is used, it WILL result in a long download time as it
       grabs the big file.  After that it will be much faster as the file is
       already local.

    TODO: Labels and colors for these indices should get moved here, but
    currently reside in watershed_workflow.colors.

    Parameters
    ----------
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
    colors = colors
    indices = indices

    def __init__(self, layer='Land_Cover', year=None, location='L48'):
        self.layer, self.year, self.location = self.validate_input(layer, year, location)

        self.layer_name = 'NLCD_{1}_{0}_{2}'.format(self.layer, self.year, self.location)
        self.name = 'National Land Cover Database (NLCD) Layer: {}'.format(self.layer_name)
        self.names = watershed_workflow.sources.names.Names(self.name, 'land_cover',
                                                            self.layer_name,
                                                            self.layer_name + '.img')

    def validate_input(self, layer, year, location):
        """Validates input to the __init__ method."""
        valid_layers = ['Land_Cover', 'Imperviousness']
        if layer not in valid_layers:
            raise ValueError('NLCD invalid layer "{}" requested, valid are: {}'.format(
                layer, valid_layers))

        valid_locations = ['L48', 'AK', 'HI', 'PR']
        if location not in valid_locations:
            raise ValueError('NLCD invalid location "{}" requested, valid are: {}'.format(
                location, valid_locations))

        valid_years = {
            'L48': [2019, 2016, 2013, 2011, 2008, 2006, 2004, 2001],
            'AK': [2011, 2001],
            'HI': [2001, ],
            'PR': [2001, ],
        }
        if year is None:
            year = valid_years[location][0]
        else:
            if year not in valid_years[location]:
                raise ValueError(
                    'NLCD invalid year "{}" requested for location {}, valid are: {}'.format(
                        year, location, valid_years[location]))

        return layer, year, location

    def get_raster(self, shply, crs, force_download=False):
        """Download and read a DEM for this shape, clipping to the shape.

        Parameters
        ----------
        shply : fiona or shapely shape
          Shape to provide bounds of the raster.
        crs : CRS
          CRS of the shape.
        force_download : bool, optional
          Download or re-download the file if true.

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
            shply = watershed_workflow.utils.create_shply(shply['geometry'])
        if type(shply) is shapely.geometry.MultiPolygon:
            shply = shapely.ops.unary_union(shply)

        # download (or hopefully don't) the file
        filename, nlcd_profile = self._download()

        logging.debug('CRS: {}'.format(nlcd_profile['crs']))

        # warp to crs
        shply = watershed_workflow.warp.shply(
            shply, crs, watershed_workflow.crs.from_rasterio(nlcd_profile['crs']))

        # load raster
        with rasterio.open(filename, 'r') as fid:
            profile = fid.profile
            out_image, out_transform = rasterio.mask.mask(fid, [shply, ], crop=True, nodata=0)

        profile.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 0
        })

        assert (len(out_image.shape) == 3)
        return profile, out_image[0, :, :]

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
                raise NotImplementedError(
                    'Not yet implemented (but trivial to add, just ask!): {}'.format(
                        self.layer_name))

            downloadfile = os.path.join(work_folder, url.split("/")[-1])
            source_utils.download(url, downloadfile, force)
            source_utils.unzip(downloadfile, work_folder)

            # hope we can find it?
            img_files = [f for f in os.listdir(work_folder) if f.endswith('.img')]
            assert (len(img_files) == 1)
            target = os.path.join(work_folder, img_files[0])
            os.rename(target, filename)
            os.rename(target[:-3] + 'ige', filename[:-3] + 'ige')

        with rasterio.open(filename, 'r') as fid:
            profile = fid.profile
        return filename, profile
