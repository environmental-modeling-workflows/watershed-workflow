"""Mosaics rasters to cover the bounds of a shape, then clips those rasters to those bounds."""

import math
import numpy as np
import logging

import fiona
import rasterio
import rasterio.mask
import rasterio._base
from rasterio.transform import Affine
from rasterio.merge import merge as merge_tool
    

import workflow.conf


def clip_dem(shp, source, feather=10, nodata=-999, precision=7):
    """Writes a raster which provides the dem for a range that covers a given huc.
    """
    hname = shp['properties'][next(k for k in shp['properties'].keys() if k.startswith('HUC'))]
    logging.info('Clipping: "%s"'%hname)

    # determine the bounds, in lat-long, at 1 degree granularity, of the huc
    xy = np.array(shp['geometry']['coordinates'][0])
    bounds = [xy[:,0].min(), xy[:,1].min(), xy[:,0].max(), xy[:,1].max()]
    bounds_1deg = [math.floor(bounds[0]), math.floor(bounds[1]), math.ceil(bounds[2]), math.ceil(bounds[3])]
    
    # collect the raster data covering those bounds
    infiles = source.download(bounds_1deg)
    assert(len(infiles) > 0)

    datasets = [rasterio.open(f) for f in infiles]
    dest, output_transform = merge_tool(datasets, bounds=bounds,
                                        nodata=nodata, precision=precision)

    profile = datasets[0].profile
    profile['transform'] = output_transform
    profile['height'] = dest.shape[1]
    profile['width'] = dest.shape[2]
    profile['count'] = dest.shape[0]
    if nodata is not None:
        profile['nodata'] = nodata
    return profile, dest


        
    

