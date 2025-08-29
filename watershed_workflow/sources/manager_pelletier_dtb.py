"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os, sys
import logging
import xarray as xr
import shapely.geometry
import numpy as np

from watershed_workflow.crs import CRS

from . import manager_raster
from . import filenames

# No API for getting GLHYMPS locally -- must download the whole thing.
urls = {
    'Pelletier at NASA DAAC':
    'https://daac.ornl.gov/SOILS/guides/Global_Soil_Regolith_Sediment.html'
}


class ManagerPelletierDTB(manager_raster.ManagerRaster):
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
            # Use default file location via Names system
            self.names = filenames.Names(
                'Pelletier DTB',
                os.path.join('soil_structure', 'PelletierDTB', 'Global_Soil_Regolith_Sediment_1304',
                             'data'), '', 'average_soil_and_sedimentary-deposit_thickness.tif')
            filename = self.names.file_name()
        else:
            # Use provided filename directly
            self.names = None
            
        # Initialize ManagerRaster with the resolved filename
        # ManagerRaster will set name and source attributes appropriately
        super(ManagerPelletierDTB, self).__init__(filename, None, None, None, None)


    def _download(self, force : bool = False):
        """Validate the files exist, returning the filename."""
        filename = self.names.file_name()
        logging.info('  from file: {}'.format(filename))
        if not os.path.exists(filename):
            logging.error(f'PelletierDTB download file {filename} not found.')
            logging.error('See download instructions below\n\n')
            logging.error(self.__doc__)
            raise RuntimeError(f'PelletierDTB download file {filename} not found.')
        return filename


    def _fetchDataset(self, request : manager_raster.ManagerRaster.Request) -> xr.Dataset:
        """Fetch the data."""
        dset = super(ManagerPelletierDTB, self)._fetchDataset(request)

        # DTB in pelletier is an int with -1 indicating nodata
        dset['band_1'] = dset['band_1'].astype("float32")
        dset['band_1'].encoding["_FillValue"] = np.nan
        dset.encoding["_FillValue"] = np.nan
        dset['band_1'].where(dset['band_1'] < 0)
        return dset
