"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os,sys
import logging
import numpy as np
import pandas

import workflow.sources.manager_shape
import workflow.sources.names
import workflow.soil_properties

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

    def get_shapes(self, bounds, crs, force_download=None):
        """Read the shapes in bounds provided by shape object.

        Parameters
        ----------
        bounds : bounds tuple [x_min, y_min, x_max, y_max]
          bounds in which to find GLHYMPS shapes.
        crs : CRS
          CRS of the bounds

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list
            List of fiona shapes that match the bounds.
        """
        logging.info(f'Getting shapes of GLHYMPS on bounds: {bounds}')
        filename = self._download()
        return super(FileManagerGLHYMPS, self).get_shapes(bounds, crs)

    def _download(self):
        """Download the files, returning downloaded filename."""
        # check directory structure
        filename = self.names.file_name()
        logging.info('  from file: {}'.format(filename))
        if not os.path.exists(filename):
            logging.error(f'GLHYMPS download file {filename} not found.')
            logging.error('See download instructions below\n\n')
            logging.error(self.__doc__)
            raise RuntimeError(f'GLHYMPS download file {filename} not found.')
        return filename

    def get_shapes_and_properties(self, bounds, crs, min_porosity=0.01, max_permeability=np.inf):
        """Read shapes and process properties.

        Parameters
        ----------
        bounds : bounds tuple [x_min, y_min, x_max, y_max]
          bounds in which to find GLHYMPS shapes.
        crs : CRS
          CRS of the bounds.
        min_porosity : optional, double in [0,1]
          Some GLHYMPs formations have zero porosity, and this breaks
          most codes.  This allows the user to set the minimum valid
          porosity.  Defaults to 0.01 (1%).
        max_permeability : optional, double > 0
          Some GLHYMPs formations (fractured bedrock?) have very 
          high permeability, and this results in very slow runs.  This
          allows the user to set a maximum valid permeability [m^2].
          Defaults to inf.

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list
            List of fiona shapes that match the index or bounds.
        properties : pandas dataframe
            Dataframe including geologic properties.

        """
        profile, shapes = self.get_shapes(bounds, crs)
        ids = np.array([shp['properties']['OBJECTID_1'] for shp in shapes], dtype=int)
        for shp in shapes:
            shp['properties']['id'] = shp['properties']['OBJECTID_1']
        
        Ksat = np.array([shp['properties']['logK_Ferr_'] for shp in shapes], dtype=float)
        Ksat = 10**(Ksat / 100) # units = m^2, division by 100 is per GLHYMPS Readme file
        Ksat = np.minimum(Ksat, max_permeability)
        Ksat_std = np.array([shp['properties']['K_stdev_x1'] for shp in shapes], dtype=float) # standard deviation
        Ksat_std = Ksat_std / 100 # division by 100 is per GLHYMPS readme
        poro = np.array([shp['properties']['Porosity_x'] for shp in shapes], dtype=float) # [-]
        poro = poro / 100 # division by 100 is per GLHYMPS readme
        poro = np.maximum(poro, min_porosity) # some values are 0?

        descriptions = [shp['properties']['Descriptio'] for shp in shapes]
        # derived properties
        # - this scaling law has trouble for really small porosity, especially high permeability low porosity
        vg_alpha = workflow.soil_properties.alpha_from_permeability(Ksat, poro)
        vg_n = 2.0  # arbitrarily chosen
        sr = 0.01  # arbitrarily chosen

        properties = pandas.DataFrame(data={'id' : ids,
                                            'source' : 'GLHYMPS',
                                            'permeability [m^2]' : Ksat,
                                            'logk_stdev [-]' : Ksat_std,
                                            'porosity [-]' : poro,
                                            'van Genuchten alpha [Pa^-1]' : vg_alpha,
                                            'van Genuchten n [-]' : vg_n,
                                            'residual saturation [-]' : sr,
                                            'description' : descriptions,
                                            })
        return profile, shapes, properties
