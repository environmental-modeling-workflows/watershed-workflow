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
import workflow.utils

def warp_xy(x, y, old_crs, new_crs):
    """Warps a set of points from old_crs to new_crs."""
    if workflow.crs.equal(old_crs, new_crs):
        return x,y
    
    old_crs_proj = workflow.crs.to_proj(old_crs)
    new_crs_proj = workflow.crs.to_proj(new_crs)

    return pyproj.transform(old_crs_proj, new_crs_proj, x,y)

def warp_bounds(bounds, old_crs, new_crs):
    """Uses proj to reproject bounds, NOT IN PLACE"""
    return warp_shapely(shapely.geometry.box(*bounds), old_crs, new_crs).bounds
    
    # x = np.array([bounds[0], bounds[2]])
    # y = np.array([bounds[1], bounds[3]])
    # x2,y2 = warp_xy(x,y,old_crs, new_crs)
    # return [x2[0],y2[0],x2[1],y2[1]]

def warp_shapely(shp, old_crs, new_crs):
    """Uses proj to reproject shapes, NOT IN PLACE"""
    if workflow.crs.equal(old_crs, new_crs):
        return shp

    old_crs_proj = workflow.crs.to_proj(old_crs)
    new_crs_proj = workflow.crs.to_proj(new_crs)
    
    return shapely.ops.transform(lambda x,y:pyproj.transform(old_crs_proj, new_crs_proj, x,y), shp)

def warp_shapelys(shps, old_crs, new_crs):
    return [warp_shapely(shp, old_crs, new_crs) for shp in shps]

def warp_shape(feature, old_crs, new_crs):
    """Uses proj to reproject shapes, IN PLACE"""
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
        x,y = warp_xy(np.array([feature['geometry']['coordinates'][0],]), np.array([feature['geometry']['coordinates'][1],]), old_crs, new_crs)
        feature['geometry']['coordinates'] = (x[0], y[0])

    elif dim == 1:
        # line-like or polygon with no holes
        coords = np.array(feature['geometry']['coordinates'],'d')
        assert(len(coords.shape) is 2 and coords.shape[1] in [2,3] )
        x,y = warp_xy(coords[:,0], coords[:,1], old_crs, new_crs)
        new_coords = [xy for xy in zip(x,y)]
        feature['geometry']['coordinates'] = new_coords

    elif dim == 2:
        # multi-line or polygon with holes
        new_rings = []
        for i in range(len(feature['geometry']['coordinates'])):
            coords = np.array(feature['geometry']['coordinates'][i],'d')
            assert(len(coords.shape) is 2 and coords.shape[1] in [2,3])
            x,y = warp_xy(coords[:,0], coords[:,1], old_crs, new_crs)
            new_coords = list(zip(x,y))
            new_rings.append(new_coords)
        feature['geometry']['coordinates'] = new_rings
        
    elif dim == 3:
        # multi-polygon
        for i in range(len(feature['geometry']['coordinates'])):
            for j in range(len(feature['geometry']['coordinates'][i])):
                coords = np.array(feature['geometry']['coordinates'][i][j],'d')
                assert(len(coords.shape) is 2 and coords.shape[1] in [2,3])
                x,y = warp_xy(coords[:,0], coords[:,1], old_crs, new_crs)
                new_coords = [xy for xy in zip(x,y)]
                feature['geometry']['coordinates'][i][j] = new_coords
            
                    
    
def warp_shapefile(infile, outfile, epsg=None):
    """Changes the projection of a shapefile."""
    if epsg is None:
        new_crs = workflow.crs.default_crs()
    else:
        new_crs = fiona.crs.from_epsg(epsg)

    with fiona.open(infile, 'r') as shp:
        old_crs = shp.crs

        if old_crs == new_crs:
            warnings.warn("Requested destination CRS is the same as the source CRS")
            shutil.copy(infile, outfile)            
        else:
            with fiona.open(outfile, 'w', 'ESRI Shapefile', schema=shp.schema.copy(), crs=new_crs) as fid:
                for feat in shp:
                    warp_shape(feat, old_crs, new_crs)
                    fid.write(feat)


def warp_raster(src_profile, src_array, dst_crs=None, dst_profile=None):
    """Changes the projection of a raster."""
    if dst_profile is None and dst_crs is None:
        dst_crs = workflow.crs.default_crs()
        
    if dst_profile is not None and dst_crs is not None:
        if dst_crs != dst_profile['crs']:
            raise RuntimeError("Given both destination profile and crs, but not matching!")

    if dst_crs is None:
        dst_crs = dst_profile['crs']

    # return if no warp needed
    if dst_crs == src_profile['crs']:
        return src_profile, src_array

        
    src_bounds = rasterio.transform.array_bounds(src_profile['height'], src_profile['width'], src_profile['transform'])
    logging.debug('Warping raster with bounds: {} to CRS: {}'.format(src_bounds, dst_crs['init']))
        
    if dst_profile is None:
        dst_profile = src_profile.copy()

        # Calculate the ideal dimensions and transformation in the new crs
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src_profile['crs'], dst_crs, src_profile['width'], src_profile['height'], *src_bounds)

        # update the relevant parts of the profile
        dst_profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })

    # Reproject and return
    dst_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)
    rasterio.warp.reproject(src_array, dst_array, src_profile['transform'], src_crs=src_profile['crs'],
                            dst_transform=dst_transform, dst_crs=dst_crs, resampling=rasterio.warp.Resampling.nearest)

    dst_bounds = rasterio.transform.array_bounds(dst_profile['height'], dst_profile['width'], dst_profile['transform'])
    return dst_profile, dst_array

def warp_raster_file(infile, outfile, epsg=None):
    """Reads infile, writes outfile in destination epsg"""
    if epsg is None:
        dst_crs = workflow.crs.default_crs()
    else:
        dst_crs = fiona.crs.from_epsg(epsg)

    with rasterio.open(infile, 'r') as src:
        dst_profile, dst_array = warp_raster(src.profile, src.read(1), dst_crs=dst_crs)

        with rasterio.open(outfile, 'w', **dst_profile) as dst:
            dst.write(dst_array)

            for i in range(1,src.count):
                dst_profile, dst_array = warp_raster(src.profile, src.read(i+1), dst_profile=dst_profile)
                dst.write(dst_array)
                
