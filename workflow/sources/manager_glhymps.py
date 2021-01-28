"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os,sys
import logging
import numpy as np
import pandas

import workflow.sources.manager_shape
import workflow.sources.names

# No API for getting GLHYMPS locally -- must download the whole thing.
urls = {'GLHYMPS version 2.0' : 'https://doi.org/10.5683/SP2/TTJNIU'}


class FileManagerGLHYMPS(workflow.sources.manager_shape.FileManagerShape):
    """The [GLHYMPS]_ global hydrogeology map provides global values of a
    two-layer (unconsolidated, consolidated) structure.

    .. note:: GLHYMPS does not have an API, and is a large (~4GB)
       download.  Download the file from the below citation DOI and
       unzip the file into:
       
       <data_directory>/soil_structure/GLHYMPS/

       which should yield GLHYMPS.shp (amongst other files).

    .. [GLHYMPS] Huscroft, J.; Gleeson, T.; Hartmann, J.; Börker, J.,
       2018, "Compiling and mapping global permeability of the
       unconsolidated and consolidated Earth: GLobal HYdrogeology MaPS
       2.0 (GLHYMPS 2.0). [Supporting Data]",
       https://doi.org/10.5683/SP2/TTJNIU, Scholars Portal Dataverse,
       V1

    """

    def __init__(self):
        self.name = 'GLHYMPS version 2.0'
        self.names = workflow.sources.names.Names(self.name,
                                                  os.path.join('soil_structure','GLHYMPS'),
                                                  '', 'GLHYMPS.shp')
        super(FileManagerGLHYMPS, self).__init__(self.names.file_name())

    def get_shapes(self, shape, crs, force_download=None):
        """Read the shapes in bounds provided by shape object.

        Parameters
        ----------
        shape : fiona or shapely shape
          Shape to provide bounds of the raster.
        crs : CRS
          CRS of the shape.

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list
            List of fiona shapes that match the index or bounds.
        """
        filename = self._download()
        return super(FileManagerGLHYMPS, self).get_shapes(shape, crs)

    def _download(self):
        """Download the files, returning downloaded filename."""
        # check directory structure
        filename = self.names.file_name()
        logging.debug('  requires: {}'.format(filename))
        if not os.path.exists(filename):
            logging.error(f'GLHYMPS download file {filename} not found.')
            logging.error('See download instructions below\n\n')
            logging.error(self.__doc__)
            raise RuntimeError(f'GLHYMPS download file {filename} not found.')
        return filename

    def get_shapes_and_properties(self, index_or_bounds=-1, crs=None):
        """Read shapes and process properties.

        Parameters
        ----------
        shape : fiona or shapely shape
          Shape to provide bounds of the raster.
        crs : CRS
          CRS of the shape.

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list
            List of fiona shapes that match the index or bounds.
        properties : pandas dataframe
            Dataframe including geologic properties.
        """
        profile, shapes = self.get_shapes(index_or_bounds, crs)

        ids = np.array([shp['properties']['OBJECTID_1'] for shp in shapes], dtype=int)
        for shp in shapes:
            shp['properties']['id'] = shp['properties']['OBJECTID_1']
        
        Ksat = np.array([shp['properties']['logK_Ferr_'] for shp in shapes], dtype=float)
        Ksat = 10**(Ksat / 100) # units = m^2, division by 100 is per GLHYMPS Readme file
        Ksat_std = np.array([shp['properties']['K_stdev_x1'] for shp in shapes], dtype=float) # standard deviation
        Ksat_std = Ksat_std / 100 # division by 100 is per GLHYMPS readme
        poro = np.array([shp['properties']['Porosity_x'] for shp in shapes], dtype=float) # [-]
        poro = poro / 100 # division by 100 is per GLHYMPS readme
        poro = np.maximum(poro, 0.01) # some values of fine clays are 0
        dtb = np.array([shp['properties']['MEAN'] for shp in shapes], dtype=float)
        dtb = dtb / 100 # cm --> m
                       

        properties = pandas.DataFrame(data={'id' : ids,
                                            'source' : 'GLHYMPS',
                                            'permeability [m^2]' : Ksat,
                                            'logk_stdev [-]' : Ksat_std,
                                            'porosity [-]' : poro,
                                            'depth to bedrock [m]' : dtb,
                                            })
        return profile, shapes, properties
