"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os, sys
import logging
import xarray as xr
import shapely.geometry

import watershed_workflow.sources.manager_raster
import watershed_workflow.sources.names
from watershed_workflow.crs import CRS


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
                   geometry : shapely.geometry.base.BaseGeometry,
                   geometry_crs : CRS,
                   band : int = 1) -> xr.DataArray:
        """Read the DTB raster.

        Parameters
        ----------
        geometry : shapely.geometry.base.BaseGeometry
          Subset the raster to cover this shape.
        geometry_crs : CRS
          CRS of the geometry

        Returns
        -------
        xr.DataArray
        """
        filename = self._download()
        raster = super(ManagerPelletierDTB, self).getDataset(geometry, geometry_crs, band)
        return raster

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
