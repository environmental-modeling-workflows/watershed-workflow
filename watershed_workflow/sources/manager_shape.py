"""Basic manager for interacting with shapefiles.
"""

import attr

@attr.s
class FileManagerShape:
    """A simple class for reading shapefiles.

    Parameters
    ----------
    filename : str
      Path to the shapefile.
    """
    _filename = attr.ib(type=str)
    name = 'shapefile'

    def get_shape(self, *args, **kwargs):
        """Read the file and filter to get shapes, then ensures there is only one
        match.

        Parameters
        ----------
        See that of get_shapes().

        Returns
        -------
        shapes : geopandas.GeoDataFrame
            Shapes in the file.

        """
        return self.get_shapes(*args, **kwargs)

    def get_shapes(self, bounds=None, bounds_crs=None):
        """Read the file and filter to get shapes.

        This accepts either an index, which is the integer index of the desired
        shape in the file, or a bounding box.  

        Parameters
        ----------
        bounds : [xmin, ymin, xmax, ymax], optional
            Bounding box to filter shapes.
        bounds_crs : crs-type
            Coordinate system of the bounding box.

        Returns
        -------
        shapes : geopandas.GeoDataFrame
            Shapes in the file.
        
        """
        if bounds is not None:
            info = pyogrio.read_info(self._filename)
            file_crs = watershed_workflow.crs.from_string(info['crs'])
            bounds = watershed_workflow.warp.bounds(bounds, bounds_crs, file_crs)
        return geopandas.read_file(self._filename, bounds)
