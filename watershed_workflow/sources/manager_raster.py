"""Basic manager for interacting with raster files.
"""

import numpy as np
import attr
import rasterio
import rasterio.windows
import watershed_workflow.crs
import logging


@attr.s
class ManagerRaster:
    """A simple class for reading rasters.

    Parameters
    ----------
    filename : str
      Path to the raster file.
    """
    _filename = attr.ib(type=str)
    name = 'raster'

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
            shape = watershed_workflow.utils.create_shply(shape)

        with rasterio.open(self._filename, 'r') as fid:
            profile = fid.profile

            # some raster profiles end up with inconsistent dtype and type(nodata)?
            if 'nodata' in profile and profile['nodata'] is not None:
                profile['nodata'] = np.array([profile['nodata'], ], dtype=profile['dtype'])[0]
            else:
                profile['nodata'] = None

            inv_transform = ~profile['transform']

            # warp to my crs
            my_crs = watershed_workflow.crs.from_rasterio(profile['crs'])
            shply = watershed_workflow.warp.shply(shape, crs, my_crs)
            bounds = shply.bounds
            logging.info(f'bounds in my_crs: {bounds}')

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

            window = rasterio.windows.Window(col_off=x0,
                                             row_off=y0,
                                             width=window_profile['width'],
                                             height=window_profile['height'])
            window_profile['transform'] = rasterio.windows.transform(window, profile['transform'])

            raster = fid.read(band, window=window)
            if (raster.shape != (window_profile['height'], window_profile['width'])):
                # the raster does not cover the domain!
                assert (raster.shape[0] <= window_profile['height'])
                assert (raster.shape[1] <= window_profile['width'])

                # create a new raster and set this raster in the right place
                raster_fullsize = window_profile['nodata'] * np.ones(
                    (window_profile['height'], window_profile['width']), raster.dtype)

                if x0 < 0: x0_off = -x0
                else: x0_off = 0

                if x1 >= profile['width']:
                    x1_off = window_profile['width'] - (x1 - profile['width'])
                else:
                    x1_off = window_profile['width']

                if y0 < 0: y0_off = -y0
                else: y0_off = 0

                if y1 >= profile['height']:
                    y1_off = window_profile['height'] - (y1 - profile['height'])
                else:
                    y1_off = window_profile['height']

                raster_fullsize[y0_off:y1_off, x0_off:x1_off] = raster
                raster = raster_fullsize

        if 'nodata' in window_profile and window_profile['nodata'] is not None:
            window_profile['nodata'] = np.array([window_profile['nodata'], ], dtype=raster.dtype)[0]
        return window_profile, raster
