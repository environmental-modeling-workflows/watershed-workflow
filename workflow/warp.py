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
import workflow.conf

def warp_xy(x, y, old_crs, new_crs):
    """Warps a set of points from old_crs to new_crs."""
    if old_crs == new_crs:
        return x,y

    old_crs_proj = pyproj.Proj(old_crs)
    new_crs_proj = pyproj.Proj(new_crs)
    return pyproj.transform(old_crs_proj, new_crs_proj, x,y)

def warp_shapely(shp, old_crs, new_crs):
    """Uses proj to reproject shapes, NOT IN PLACE"""
    x,y = warp_xy(shp.boundary.xy[0], shp.boundary.xy[1], old_crs, new_crs)
    return shapely.geometry.Polygon(x,y)

def warp_shape(feature, old_crs, new_crs):
    """Uses proj to reproject shapes, IN PLACE"""
    if len(feature['geometry']['coordinates']) is 0:
        return
    
    if type(feature['geometry']['coordinates']) is np.ndarray or type(feature['geometry']['coordinates'][0]) is tuple:
        # single object
        coords = np.array(feature['geometry']['coordinates'],'d')
        assert(len(coords.shape) is 2)
        x,y = warp_xy(coords[:,0], coords[:,1], old_crs, new_crs)
        new_coords = [xy for xy in zip(x,y)]
        feature['geometry']['coordinates'] = new_coords
                
    else:
        # object collection
        for i,c in enumerate(feature['geometry']['coordinates']):
            coords = np.array(c,'d')
            if len(coords.shape) is 2:
                assert(coords.shape[-1] is 2)
                x,y = warp_xy(coords[:,0], coords[:,1], old_crs, new_crs)
                new_coords = [xy for xy in zip(x,y)]
                feature['geometry']['coordinates'][i] = new_coords
            elif len(coords.shape) is 3:
                for j,c2 in enumerate(feature['geometry']['coordinates'][i]):
                    coords = np.array(c2,'d')
                    assert(coords.shape[-1] is 2)
                    x,y = warp_xy(coords[:,0], coords[:,1], old_crs, new_crs)
                    new_coords = [xy for xy in zip(x,y)]
                    feature['geometry']['coordinates'][i][j] = new_coords
                    
    
def warp_shapefile(infile, outfile, epsg=None):
    """Changes the projection of a shapefile."""
    if epsg is None:
        new_crs = workflow.conf.default_crs()
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
        dst_crs = workflow.conf.default_crs()
        
    if dst_profile is not None and dst_crs is not None:
        if dst_crs != dst_profile['crs']:
            raise RuntimeError("Given both destination profile and crs, but not matching!")

    if dst_crs is None:
        dst_crs = dst_profile['crs']

    logging.debug('Warping raster with bounds: %s to CRS: "%s"'%(workflow.conf.bounds_from_profile(src_profile), dst_crs['init']))
        
    if dst_profile is None:
        dst_profile = src_profile.copy()

        # Calculate the ideal dimensions and transformation in the new crs
        dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src_profile['crs'], dst_crs, src_profile['width'], src_profile['height'], *workflow.conf.bounds_from_profile(src_profile))

        # update the relevant parts of the profile
        dst_profile.update({
            'crs': dst_crs,
            'transform': dst_affine,
            'affine': dst_affine,
            'width': dst_width,
            'height': dst_height
        })

        
    # return if no warp needed
    if dst_crs == src_profile['crs']:
        return src_profile, src_array

    # Reproject and return
    dst_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)
    rasterio.warp.reproject(
        source=src_array,
        src_crs=src_profile['crs'],
        src_transform=src_profile['affine'],
        destination=dst_array,
        dst_transform=dst_affine,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
        num_threads=2)

    logging.debug('  new bounds: %s'%workflow.conf.bounds_from_profile(dst_profile))

    return dst_profile, dst_array

def warp_raster_file(infile, outfile, epsg=None):
    """Reads infile, writes outfile in destination epsg"""
    if epsg is None:
        dst_crs = workflow.conf.default_crs()
    else:
        dst_crs = fiona.crs.from_epsg(epsg)

    with rasterio.open(infile, 'r') as src:
        dst_profile, dst_array = warp_raster(src.profile, src.read(1), dst_crs=dst_crs)

        with rasterio.open(outfile, 'w', **dst_profile) as dst:
            dst.write(dst_array)

            for i in range(1,src.count):
                dst_profile, dst_array = warp_raster(src.profile, src.read(i+1), dst_profile=dst_profile)
                dst.write(dst_array)
                
