"""Basic manager for interacting with raster files.
"""

import attr
import rasterio

@attr.s
class FileManagerRaster:
    """A simple class for reading rasters.

    Parameter
    ---------
    filename : str
      Path to the raster file.
    """
    _filename = attr.ib(type=str)
    
    def get_raster(self, band=1):
        """Gets a raster from the file.

        Parameter
        ---------
        band : int,optional
          Default is 1, the first band (1-indexed).
        """
        with rasterio.open(self._filename, 'r') as fid:
            profile = fid.profile
            raster = fid.read(band)
        return profile, raster
