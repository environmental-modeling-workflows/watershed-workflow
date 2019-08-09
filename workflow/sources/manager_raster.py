"""Basic manager for interacting with raster files.
"""

import attr
import rasterio

@attr.s
class FileManagerRaster:
    _filename = attr.ib(type=str)
    
    def get_raster(self, band=1):
        """Gets a raster, ignores the shape.
        """
        with rasterio.open(self._filename, 'r') as fid:
            profile = fid.profile
            raster = fid.read(band)
        return profile, raster
