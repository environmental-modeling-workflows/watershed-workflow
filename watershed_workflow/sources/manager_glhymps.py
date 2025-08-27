"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os, sys
import logging
import numpy as np
import pandas, geopandas
import shapely
from shapely.geometry.base import BaseGeometry

import watershed_workflow.sources.manager_shapefile
import watershed_workflow.sources.names
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
            super(ManagerGLHYMPS, self).__init__(self.names.file_name(), id_name='OBJECTID_1')
        else:
            self.name = filename
            self.names = None
            super(ManagerGLHYMPS, self).__init__(self.name, id_name='OBJECTID_1')

            
    def _download(self, force : bool = False):
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

    
    def _getShapesByGeometry(self, geometry_gdf: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
        """Fetch shapes for the given geometry, ensuring file exists first.

        Parameters
        ----------
        geometry_gdf : geopandas.GeoDataFrame
            GeoDataFrame with geometries in native_crs_in to search for shapes.

        Returns
        -------
        geopandas.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        # Ensure GLHYMPS file exists before attempting to read
        self._download()
        return super()._getShapesByGeometry(geometry_gdf)

    
    def _getShapesByID(self, ids: list[str]) -> geopandas.GeoDataFrame:
        """Fetch shapes by ID list, ensuring file exists first.

        Parameters
        ----------
        ids : list[str]
            List of IDs to retrieve.

        Returns
        -------
        geopandas.GeoDataFrame
            Raw GeoDataFrame with native column names and CRS properly set.
        """
        # Ensure GLHYMPS file exists before attempting to read
        self._download()
        return super()._getShapesByID(ids)

