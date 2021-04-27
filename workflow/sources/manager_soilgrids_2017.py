"""Manager for downloading SoilGrids250m-2017 products."""

import os,sys
import logging
import numpy as np
import shapely
import rasterio
import rasterio.mask

import workflow.sources.utils as source_utils
import workflow.sources.names
import workflow.warp


class FileManagerSoilGrids2017:
    """SoilGrids 250m (2017) datasets.

    SoilGrids 2017 maintains, to date, the only complete
    characterization of all soil properties needed for a hydrologic
    model.  The resolution is decent, and the accuracy is ok, but most
    importantly it is complete.

    .. [SoilGrids2017] https://www.isric.org/explore/soilgrids/faq-soilgrids-2017

    .. [hengl2014soilgrids] Hengl, Tomislav, et al. "SoilGrids1km—global soil information based on automated mapping." PloS one 9.8 (2014): e105992.

    .. [hengl2017soilgrids] Hengl, Tomislav, et al. "SoilGrids250m: Global gridded soil information based on machine learning." PLoS one 12.2 (2017): e0169748.
    
    See the above link for a complete listing of potential variable
    names; included here are a subset used by this code.  That said,
    any 2017 filename can be used with this source manager.

    .. list-table::
        :widths: 25 25 75
    
        * - name
          - units
          - description
        * - BDTICM
          - :math:`cm`
          - Absolute depth to continuous, unfractured bedrock.
        * - BLDFIE
          - :math:`kg m^-3`
          - Bulk density of fine earth
        * - CLYPPT
          - :math:`%`
          - percent clay
        * - SLTPPT
          - :math:`%`
          - percent silt
        * - SNDPPT
          - :math:`%`
          - percent sand
        * - WWP
          - :math:`%`
          - Soil water capacity % at wilting point
    """
    URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/"
    DEPTHS = [0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0]

    def __init__(self, variant=None):
        if variant == 'US':
            self.name = 'SoilGrids2017_US'
            self.names = workflow.sources.names.Names(self.name, 'soil_structure',
                                                      self.name, '{variable}_M_{soillevel}250m_ll_us.tif')
        else:
            self.name = 'SoilGrids2017'
            self.names = workflow.sources.names.Names(self.name, 'soil_structure',
                                                      self.name, '{variable}_M_{soillevel}250m_ll.tif')

        
    def get_raster(self, shply, crs, variable, layer=None, force_download=False):
        """Download and read a raster for this shape, clipping to the shape.

        Parameters
        ----------
        shply : fiona or shapely shape or bounds
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

        Note that the raster provided is in SoilGrids native CRS
        (which is in the rasterio profile), not the shape's CRS.
        """
        # get shape as a shapely, single Polygon
        if type(shply) is dict:
            shply = workflow.utils.shply(shply['geometry'])
        if type(shply) is shapely.geometry.MultiPolygon:
            shply = shapely.ops.cascaded_union(shply)

        # download (or hopefully don't) the file
        filename, profile = self._download(variable, layer)
        logging.info(f"CRS: {profile['crs']}")

        # warp to crs
        shply = workflow.warp.shply(shply, crs, workflow.crs.from_rasterio(profile['crs']))

        # load the raster
        with rasterio.open(filename, 'r') as fid:
            profile = fid.profile
            out_image, out_transform = rasterio.mask.mask(fid, [shply,], crop=True)

        profile.update({ "height" : out_image.shape[1],
                         "width" : out_image.shape[2],
                         "transform" : out_transform})

        assert(len(out_image.shape) == 3)
        return profile, out_image[0,:,:]

        
    def get_depth_to_bedrock(self, shply, crs, force_download=False):
        return self.get_raster(shply, crs, 'BDTICM', None, force_download)

    
    def get_soil_texture(self, shply, crs, layer, force_download=False):
        rasters = []
        if layer == -1:
            layer = 7
        for i, variable in enumerate(['SNDPPT', 'SLTPPT', 'CLYPPT']):
            prof, raster = self.get_raster(shply, crs, variable, layer, force_download)
            rasters.append(raster)
        rasters = np.array(rasters)
        return prof, rasters


    def get_all_soil_texture(self, shply, crs, force_download=False):
        rasters = []
        for layer in range(1,8):
            prof, raster = self.get_soil_texture(shply, crs, layer, force_download)
            rasters.append(raster)
        rasters = np.array(rasters)
        return prof, rasters
    

    def get_bulk_density(self, shply, crs, layer, force_download=False):
        if layer == -1:
            layer = 7
        return self.get_raster(shply, crs, 'BLDFIE', layer, force_download)


    def get_all_bulk_density(self, shply, crs, force_download=False):
        rasters = []
        for layer in range(1,8):
            prof, raster = self.get_bulk_density(shply, crs, layer, force_download)
            rasters.append(raster)
        rasters = np.array(rasters)
        return prof, rasters


    def get_layer7(self, shply, crs, force_download=False):
        data = dict()
        prof, data['bulk density [kg m^-3]'] = self.get_bulk_density(shply, crs, -1, force_download)
        _, data['texture [%]'] = self.get_soil_texture(shply, crs, -1, force_download)
        _, data['depth to bedrock [cm]'] = self.get_depth_to_bedrock(shply, crs, force_download)
        return prof, data

    def get_all(self, shply, crs, force_download=False):
        data = dict()
        prof, data['bulk density [kg m^-3]'] = self.get_all_bulk_density(shply, crs, force_download)
        _, data['texture [%]'] = self.get_all_soil_texture(shply, crs, force_download)
        _, data['depth to bedrock [cm]'] = self.get_depth_to_bedrock(shply, crs, force_download)
        return prof, data
    
    def _download(self, variable, layer=None, force=False):
        """Downloads individual files via direct download."""
        os.makedirs(self.names.folder_name(), exist_ok=True)

        if layer is None:
            soillevel = ''
        else:
            soillevel = f'sl{layer}_'

        filename = self.names.file_name(variable=variable, soillevel=soillevel)

        # download file
        filename_base = self.names.file_name_base(variable=variable,
                                                  soillevel=soillevel)
        url = self.URL + filename_base
        source_utils.download_progress_bar(url, filename, force)

        # return raster profile
        with rasterio.open(filename, 'r') as fid:
            profile = fid.profile
        return filename, profile
        
            
            
                         
        
