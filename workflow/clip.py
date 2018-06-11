"""Mosaics rasters to cover the bounds of a shape, then clips those rasters to those bounds."""

import math
import numpy as np
import logging

import fiona
import rasterio
import rasterio.mask
import rasterio._base
from rasterio.transform import Affine
from rasterio._base import get_window

import workflow.conf
import workflow.download


def clip_dem(shp, feather=10, nodata=-999, precision=7):
    """Writes a raster which provides the dem for a range that covers a given huc.
    """
    hname = shp['properties'][next(k for k in shp['properties'].keys() if k.startswith('HUC'))]
    logging.info('Clipping: "%s"'%hname)

    # determine the bounds, in lat-long, at 1 degree granularity, of the huc
    xy = np.array(shp['geometry']['coordinates'][0])
    bounds = [xy[:,0].min(), xy[:,1].min(), xy[:,0].max(), xy[:,1].max()]
    bounds_1deg = [math.floor(bounds[0]), math.floor(bounds[1]), math.ceil(bounds[2]), math.ceil(bounds[3])]
    dst_bounds = []  # must snap these to the raster

    # collect the raster data covering those bounds
    infiles = workflow.download.download_dem(bounds_1deg)
    assert(len(infiles) > 0)

    # merge and clip
    for infile in infiles:
        logging.info('  including tile: "%s"'%infile)

        with rasterio.open(infile,'r') as fin:
            src_w, src_s, src_e, src_n = fin.bounds
            logging.info('    img bounds: %r'%list(fin.bounds))
        
            if len(dst_bounds) is 0:
                # first time, determine true bounds for the destination
                res_x, res_y = fin.res
                logging.info('    resolution: %d, %d'%fin.res)

                # feather the bounds
                dst_bounds = [bounds[0] - feather*res_x, bounds[1] - feather*res_y,
                              bounds[2] + feather*res_x, bounds[3] + feather*res_y]

                # snap the origin to fit the raster
                i_w = int(np.round((dst_bounds[0] - src_w) / res_x))
                j_n = int(np.round((src_n - dst_bounds[3]) / res_y))
                logging.debug('    exact bounds = %r'%list(bounds))
                logging.debug('    feathered bounds = %r'%list(dst_bounds))

                dst_w, dst_n = fin.profile['affine'] * (i_w, j_n)

                # set the affine map
                dst_affine = Affine.translation(dst_w, dst_n)
                dst_affine *= Affine.scale(res_x, -res_y)
            
                # determine the profile
                dst_profile = fin.profile.copy()
                dst_profile['height'] = int(math.ceil((dst_bounds[3] - dst_bounds[1]) / res_y))
                dst_profile['width'] = int(math.ceil((dst_bounds[2] - dst_bounds[0]) / res_x))
                dst_profile['transform'] = dst_affine
                dst_profile['affine'] = dst_affine
                if nodata is not None:
                    dst_profile['nodata'] = nodata

                # Adjust bounds to fit.
                dst_e, dst_s = dst_affine * (dst_profile['width'], dst_profile['height'])
                dst_bounds = [dst_w, dst_s, dst_e, dst_n]
                logging.debug('    snapped bounds = %r'%dst_bounds)

                # create destination array
                dst_array = np.empty((dst_profile['count'], dst_profile['height'], dst_profile['width']), dtype=dst_profile['dtype'])
                dst_array.fill(dst_profile['nodata'])

            # Compute spatial intersection of destination and source.
            int_w = src_w if src_w > dst_w else dst_w
            int_s = src_s if src_s > dst_s else dst_s
            int_e = src_e if src_e < dst_e else dst_e
            int_n = src_n if src_n < dst_n else dst_n

            # Compute the source window.
            src_window = get_window(int_w, int_s, int_e, int_n, fin.affine, precision=precision)
            logging.debug("Src %s window: %r", fin.name, src_window)

            # Compute the destination window.
            dst_window = get_window(int_w, int_s, int_e, int_n, dst_affine, precision=precision)
            logging.debug("Dst window: %r", dst_window)

            # Initialize temp array.
            trows, tcols = tuple(b - a for a, b in dst_window)

            temp_shape = (dst_profile['count'], trows, tcols)
            logging.debug("Temp shape: %r", temp_shape)

            temp = np.zeros(temp_shape, dtype=dst_profile['dtype'])
            temp = fin.read(out=temp, window=src_window, boundless=False, masked=True)

            # Copy elements of temp into dest.
            roff, coff = dst_window[0][0], dst_window[1][0]
            region = dst_array[:, roff:roff + trows, coff:coff + tcols]
            np.copyto(region, temp, where=np.logical_and(region == dst_profile['nodata'], temp.mask == False))

    return dst_profile, dst_array


        
    

