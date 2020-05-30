"""Used to warp shapefiles and rasters into new coordinate systems."""


import shutil
import numpy as np
import logging

import pyproj
import fiona
import fiona.crs
import rasterio.warp
import shapely.geometry

import warnings
import workflow.crs

pyproj_version = int(pyproj.__version__[0])

def xy(x, y, old_crs, new_crs):
    """Warp a set of points from old_crs to new_crs."""
    if workflow.crs.equal(old_crs, new_crs):
        return x,y

    old_crs_proj = workflow.crs.to_proj(old_crs)
    new_crs_proj = workflow.crs.to_proj(new_crs)
    
    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    x1,y1 = transformer.transform(x,y)
    return x1,y1

def bounds(bounds, old_crs, new_crs):
    """Warp a bounding box from old_crs to new_crs."""
    return shply(shapely.geometry.box(*bounds), old_crs, new_crs).bounds

def shply(shp, old_crs, new_crs):
    """Warp a shapely object from old_crs to new_crs."""
    if workflow.crs.equal(old_crs, new_crs):
        return shp
    
    old_crs_proj = workflow.crs.to_proj(old_crs)
    new_crs_proj = workflow.crs.to_proj(new_crs)
    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    return shapely.ops.transform(lambda x,y:transformer.transform(x,y), shp)

def shplys(shps, old_crs, new_crs):
    """Warp a collection of shapely objects from old_crs to new_crs."""
    return [shply(shp, old_crs, new_crs) for shp in shps]

def shape(feature, old_crs, new_crs):
    """Warp a fiona shape from old_crs to new_crs, in place."""
    # find the dimension -- can't trust the shape
    dim = -1
    ptr = feature['geometry']['coordinates']
    done = False
    while not done:
        if hasattr(ptr, '__len__'):        
            assert(len(ptr) is not 0)
            dim += 1
            ptr = ptr[0]
        else:
            done = True

    if dim == 0:
        # point
        x,y = xy(np.array([feature['geometry']['coordinates'][0],]), np.array([feature['geometry']['coordinates'][1],]), old_crs, new_crs)
        feature['geometry']['coordinates'] = (x[0], y[0])

    elif dim == 1:
        # line-like or polygon with no holes
        coords = np.array(feature['geometry']['coordinates'],'d')
        assert(len(coords.shape) is 2 and coords.shape[1] in [2,3] )
        x,y = xy(coords[:,0], coords[:,1], old_crs, new_crs)
        new_coords = [xy for xy in zip(x,y)]
        feature['geometry']['coordinates'] = new_coords

    elif dim == 2:
        # multi-line or polygon with holes
        new_rings = []
        for i in range(len(feature['geometry']['coordinates'])):
            coords = np.array(feature['geometry']['coordinates'][i],'d')
            assert(len(coords.shape) is 2 and coords.shape[1] in [2,3])
            x,y = xy(coords[:,0], coords[:,1], old_crs, new_crs)
            new_coords = list(zip(x,y))
            new_rings.append(new_coords)
        feature['geometry']['coordinates'] = new_rings
        
    elif dim == 3:
        # multi-polygon
        for i in range(len(feature['geometry']['coordinates'])):
            for j in range(len(feature['geometry']['coordinates'][i])):
                coords = np.array(feature['geometry']['coordinates'][i][j],'d')
                assert(len(coords.shape) is 2 and coords.shape[1] in [2,3])
                x,y = xy(coords[:,0], coords[:,1], old_crs, new_crs)
                new_coords = [xy for xy in zip(x,y)]
                feature['geometry']['coordinates'][i][j] = new_coords
    return feature
                    
    
def raster(src_profile, src_array, dst_crs=None, dst_profile=None):
    """Warp a raster from src_profile to dst_crs or dst_profile."""
    if (dst_crs is None and dst_profile is None):
        return src_profile, src_array
        
    if dst_profile is not None and dst_crs is not None:
        if not workflow.crs.equal(dst_crs, workflow.crs.from_rasterio(dst_profile['crs'])):
            raise RuntimeError("Given both destination profile and crs, but not matching!")

    if dst_crs is None:
        dst_crs_rasterio = dst_profile['crs']
        dst_crs = workflow.crs.from_rasterio(dst_crs_rasterio)
    else:
        dst_crs_rasterio = workflow.crs.to_rasterio(dst_crs)

    # return if no warp needed
    src_crs = workflow.crs.from_rasterio(src_profile['crs'])
    if workflow.crs.equal(dst_crs, src_crs):
        return src_profile, src_array

    src_bounds = rasterio.transform.array_bounds(src_profile['height'], src_profile['width'], src_profile['transform'])
    logging.debug('Warping raster with bounds: {} to CRS: {}'.format(src_bounds, dst_crs))
        
    if dst_profile is None:
        dst_profile = src_profile.copy()

        # Calculate the ideal dimensions and transformation in the new crs
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src_profile['crs'], dst_crs_rasterio, src_profile['width'], src_profile['height'], *src_bounds)

        # update the relevant parts of the profile
        dst_profile.update({
            'crs': dst_crs_rasterio,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })

    # Reproject and return
    dst_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)
    rasterio.warp.reproject(src_array, dst_array, src_profile['transform'], src_crs=src_profile['crs'],
                            dst_transform=dst_transform, dst_crs=dst_crs_rasterio, resampling=rasterio.warp.Resampling.nearest)

    dst_bounds = rasterio.transform.array_bounds(dst_profile['height'], dst_profile['width'], dst_profile['transform'])
    return dst_profile, dst_array

                
