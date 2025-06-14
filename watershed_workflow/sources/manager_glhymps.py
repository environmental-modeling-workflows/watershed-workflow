"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os, sys
import logging
import numpy as np
import pandas, geopandas
import shapely
from shapely.geometry.base import BaseGeometry

import watershed_workflow.sources.manager_shapefile
import watershed_workflow.sources.names
import watershed_workflow.soil_properties
from watershed_workflow.crs import CRS

# No API for getting GLHYMPS locally -- must download the whole thing.
urls = { 'GLHYMPS version 2.0': 'https://doi.org/10.5683/SP2/TTJNIU'}


class ManagerGLHYMPS(watershed_workflow.sources.manager_shapefile.ManagerShapefile):
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
    def __init__(self, filename=None):
        if filename is None:
            self.name = 'GLHYMPS version 2.0'
            self.names = watershed_workflow.sources.names.Names(
                self.name, os.path.join('soil_structure', 'GLHYMPS'), '', 'GLHYMPS.shp')
            super(ManagerGLHYMPS, self).__init__(self.names.file_name())
        else:
            self.name = filename
            self.names = None
            super(ManagerGLHYMPS, self).__init__(self.name)

    def _getShapesByGeometry(self,
                            geom : BaseGeometry,
                            geom_crs : CRS) -> geopandas.GeoDataFrame:
        """Read the file and filter to get shapes."""
        filename = self._download()
        return super(ManagerGLHYMPS, self).getShapesByGeometry(geom, geom_crs)

    def _download(self):
        """Download the files, returning downloaded filename."""
        # check directory structure
        if self.names is None:
            return self.name
        filename = self.names.file_name()
        logging.info('  from file: {}'.format(filename))
        if not os.path.exists(filename):
            logging.error(f'GLHYMPS download file {filename} not found.')
            logging.error('See download instructions below\n\n')
            logging.error(self.__doc__)
            raise RuntimeError(f'GLHYMPS download file {filename} not found.')
        return filename

    def getShapesByGeometry(self, geom, geom_crs, **kwargs):
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
        shapes = self._getShapesByGeometry(geom, geom_crs)
        props = watershed_workflow.soil_properties.mangleGLHYMPSProperties(shapes, **kwargs)
        return props
