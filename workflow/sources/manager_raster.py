"""Basic manager for interacting with raster files.
"""

import numpy as np
import attr
import rasterio
import rasterio.windows
import workflow.crs

@attr.s
class FileManagerRaster:
    """A simple class for reading rasters.

    Parameter
    ---------
    filename : str
      Path to the raster file.
    """
    _filename = attr.ib(type=str)
    
    def get_raster(self, shape, crs, band=1):
        """Download and read a DEM for this shape, clipping to the shape.
        
        Parameters
        ----------
        shape : fiona or shapely shape
          Shape to provide bounds of the raster.
        crs : CRS
          CRS of the shape.
        band : int,optional
          Default is 1, the first band (1-indexed).

        Returns
        -------
        profile : rasterio profile
          Profile of the raster.
        raster : np.ndarray
          Array containing the elevation data.

        Note that the raster provided is in its native CRS (which is in the
        rasterio profile), not the shape's CRS.
        """
        if type(shape) is dict:
            shape = workflow.utils.shply(shape)

        with rasterio.open(self._filename, 'r') as fid:
            profile = fid.profile
            inv_transform = ~profile['transform']

            # warp to my crs
            my_crs = workflow.crs.from_rasterio(profile['crs'])
            shply = workflow.warp.shply(shape, crs, my_crs)
            bounds = shply.bounds

            # find an appropriate window offset
            x0, y0 = inv_transform * (bounds[0], bounds[3])
            x0 = int(np.floor(x0))
            y0 = int(np.floor(y0))

            # find an appropriate window size
            x1, y1 = inv_transform * (bounds[2], bounds[1])
            x1 = int(np.ceil(x1))
            y1 = int(np.ceil(y1))

            # create the window
            window_profile = profile.copy()
            window_profile['height'] = y1 - y0
            window_profile['width'] = x1 - x0

            window = rasterio.windows.Window(col_off=x0, row_off=y0,
                                             width=window_profile['width'], height=window_profile['height'])
            window_profile['transform'] = rasterio.windows.transform(window, profile['transform'])

            raster = fid.read(band, window=window)
            assert(raster.shape == (window_profile['height'], window_profile['width']))

        if 'nodata' in window_profile:
            window_profile['nodata'] = np.array([window_profile['nodata'],], dtype=raster.dtype)[0]
        return window_profile, raster
