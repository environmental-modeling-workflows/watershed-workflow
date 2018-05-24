#from pyproj import Proj, transform
import shutil
import numpy as np

import pyproj
import fiona
import fiona.crs
import rasterio.warp

import warnings
import workflow.conf


def warp_shape(feature, old_crs, new_crs):
    """Uses proj to reproject shapes, IN PLACE"""
    old_crs_proj = pyproj.Proj(old_crs)
    new_crs_proj = pyproj.Proj(new_crs)

    coords =  np.array(feat['geometry']['coordinates'][0])
    x,y = pyproj.transform(old_crs_proj, new_crs_proj, coords[:,0], coords[:,1])
    new_coords = [xy for xy in zip(x,y)]

    # change only the coordinates of the feature
    feature['geometry']['coordinates'][0] = new_coords
    return feature

    
def warp_shapefile(infile, outfile, epsg=None):
    """Changes the projection of a shapefile."""
    if epsg is None:
        epsg = workflow.conf.rcParams['epsg']
    new_crs = fiona.crs.from_epsg(epsg)

    with fiona.open(infile, 'r') as shp:
        old_crs = shp.crs

        if old_crs == new_crs:
            warnings.warn("Requested destination CRS is the same as the source CRS")
            shutil.copy(infile, outfile)            
        else:
            with fiona.open(outfile, 'w', 'ESRI Shapefile', schema=shp.schema.copy(), crs=new_crs) as fid:
                for feat in shp:
                    feat = warp_shape(feat, old_crs, new_crs)
                    fid.write(feat)


def warp_raster(infile, outfile, epsg=None):
    """Changes the projection of a raster."""
    if epsg is None:
        epsg = workflow.conf.rcParams['epsg']
    dst_crs = fiona.crs.from_epsg(epsg)

    with rasterio.drivers(CHECK_WITH_INVERT_PROJ=True):
        with rasterio.open(infile, 'r') as src:
            profile = src.profile

            # Calculate the ideal dimensions and transformation in the new crs
            dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

            # update the relevant parts of the profile
            profile.update({
                'crs': dst_crs,
                'transform': dst_affine,
                'affine': dst_affine,
                'width': dst_width,
                'height': dst_height
            })

            # Reproject and write each band
            with rasterio.open(outfile, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    src_array = src.read(i)
                    dst_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)
                    
                    rasterio.warp.reproject(
                        # Source parameters
                        source=src_array,
                        src_crs=src.crs,
                        src_transform=src.affine,
                        # Destination paramaters
                        destination=dst_array,
                        dst_transform=dst_affine,
                        dst_crs=dst_crs,
                        # Configuration
                        resampling=rasterio.warp.Resampling.nearest,
                        num_threads=2)
                    dst.write(dst_array, i)

                
