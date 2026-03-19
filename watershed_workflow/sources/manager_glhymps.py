"""Manager for interacting with GLHYMPS v2.0 dataset."""
import os
import logging
import geopandas

from . import manager_shapefile
from . import cache_info as ci


class ManagerGLHYMPS(manager_shapefile.ManagerShapefile):
    """The [GLHYMPS]_ global hydrogeology map provides global values of a
    two-layer (unconsolidated, consolidated) structure.

    .. note:: GLHYMPS does not have an API, and is a large (~4GB)
       download.  Download the file from the below citation DOI and
       unzip the file into:

       <data_directory>/soil_structure/GLHYMPS/dataverse_https/

       which should yield GLHYMPS.shp (amongst other files).

    .. [GLHYMPS] Huscroft, J.; Gleeson, T.; Hartmann, J.; Börker, J.,
       2018, "Compiling and mapping global permeability of the
       unconsolidated and consolidated Earth: GLobal HYdrogeology MaPS
       2.0 (GLHYMPS 2.0). [Supporting Data]",
       https://doi.org/10.5683/SP2/TTJNIU, Scholars Portal Dataverse,
       V1

    """
    def __init__(self, filename=None):
        # Set metadata attrs before calling super so localFilePath works.
        from .manager import ManagerAttributes
        _attrs = ManagerAttributes(
            category='soil_structure',
            product='GLHYMPS 2.0',
            product_short='GLHYMPS',
            source='Scholars Portal Dataverse',
            source_short='dataverse_https',
            url='https://doi.org/10.5683/SP2/TTJNIU',
            license='CC BY 4.0',
            citation='Huscroft et al. 2018',
            description='GLHYMPS 2.0 global hydrogeology map of permeability.',
        )
        if filename is None:
            filepath = ci.localFilePath(_attrs, 'GLHYMPS.shp')
        else:
            filepath = filename
        super().__init__(filepath, id_name='OBJECTID_1', attrs=_attrs)

    def _download(self, force : bool = False):
        """Download the files, returning downloaded filename."""
        filename = self.filename
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
