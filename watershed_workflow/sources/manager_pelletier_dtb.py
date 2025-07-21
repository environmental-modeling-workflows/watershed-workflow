"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os, sys
import logging
import numpy as np
import pandas
import xarray as xr
import geopandas as gpd
from typing import Tuple


import watershed_workflow.sources.manager_raster
import watershed_workflow.sources.names
import watershed_workflow.soil_properties

# No API for getting GLHYMPS locally -- must download the whole thing.
urls = {
    'Pelletier at NASA DAAC':
    'https://daac.ornl.gov/SOILS/guides/Global_Soil_Regolith_Sediment.html'
}


class ManagerPelletierDTB(watershed_workflow.sources.manager_raster.ManagerRaster):
    """The [PelletierDTB]_ global soil regolith sediment map provides global values of
    depth to bedrock at a 1km spatial resolution.

    .. note:: Pelletier DTB is served through ORNL's DAAC, does not
       have an API, and is a large (~1GB) download.  Download the file
       from the below citation DOI and unzip the file into:
       
       <data_directory>/soil_structure/PelletierDTB/

       which should yield a set of tif files, 

       Global_Soil_Regolith_Sediment_1304/data/*.tif

    .. [PelletierDTB] Pelletier, J.D., P.D. Broxton, P. Hazenberg,
       X. Zeng, P.A. Troch, G. Niu, Z.C. Williams, M.A. Brunke, and
       D. Gochis. 2016. Global 1-km Gridded Thickness of Soil,
       Regolith, and Sedimentary Deposit Layers. ORNL DAAC, Oak Ridge,
       Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1304

    """
    def __init__(self, filename=None):
        if filename is None:
            self.name = 'Pelletier DTB'
            self.names = watershed_workflow.sources.names.Names(
                self.name,
                os.path.join('soil_structure', 'PelletierDTB', 'Global_Soil_Regolith_Sediment_1304',
                             'data'), '', 'average_soil_and_sedimentary-deposit_thickness.tif')
            super(ManagerPelletierDTB, self).__init__(self.names.file_name())
        else:
            self.name = filename
            self.names = None
            super(ManagerPelletierDTB, self).__init__(self.name)

    def getDataset(self, 
                   shape : gpd.GeoDataFrame | gpd.GeoSeries | Tuple[float, float, float, float] | list[float] | np.ndarray,
                   crs : str,
                   band : int = 1) -> xr.DataArray:
        """Read the DTB raster.

        Parameters
        ----------
        shape : gpd.GeoDataFrame | gpd.GeoSeries | Tuple[float, float, float, float] | list[float] | np.ndarray
          Subset the raster to cover this shape.
        crs : str
          CRS of the shape

        Returns
        -------
        profile : dict
            Rasterio profile
        raster : np.ndarray
            Array containing the DTB data.
        """
        if isinstance(shape, (gpd.GeoDataFrame, gpd.GeoSeries)):
            logging.info(f'Getting raster of Pelletier DTB on bounds: {shape.bounds}')
        elif isinstance(shape, (Tuple, list, np.ndarray)):
            logging.info(f'Getting raster of Pelletier DTB on bounds: {shape}')

        filename = self._download()
        return super(ManagerPelletierDTB, self).getDataset(shape, crs, band)

    def _download(self):
        """Download the files, returning downloaded filename."""
        # check directory structure
        if self.names is None:
            return self.name
        filename = self.names.file_name()
        logging.info('  from file: {}'.format(filename))
        if not os.path.exists(filename):
            logging.error(f'PelletierDTB download file {filename} not found.')
            logging.error('See download instructions below\n\n')
            logging.error(self.__doc__)
            raise RuntimeError(f'PelletierDTB download file {filename} not found.')
        return filename
