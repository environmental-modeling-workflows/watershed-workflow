"""The base interfaces of workflow."""

import os

import rasterio
import fiona

import workflow.conf
from workflow.conf import rcParams as rc
import workflow.clip
import workflow.warp

def generate_package(huc, include=12):
    """Gathers dems, shapefiles, etc, for making a mesh on a given huc"""
    huc = workflow.conf.huc_string(huc)

    # get the package directory
    packagedir = os.path.join(rc['data dir'], rc['packages data dir'], huc)
    if not os.path.isdir(packagedir):
        os.mkdir(packagedir)

    # get the shape of the containing HUC and the dem
    (shp_profile, shp), (img_profile, img_array) = workflow.clip(huc)

    # write dem to disk
    with rasterio.open(os.path.join(packagedir, 'huc_%s_dem.img'%huc), 'w', **img_profile) as dst_img:
        dst_img.write(img_array)

    # write shape to disk
    with fiona.open(os.path.join(packagedir, 'huc_%s.shp'%huc), 'w', 'ESRI Shapefile', **shp_profile) as dst_shp:
        dst_shp.write(shp)

    # now collect all HUC-12s in this HUC, project, and write a HUC12 shapefile
    if include is not None:
        if type(include) is int:
            include = "%i"%include

        huc_8_in_12 = huc + '0'*(12-len(huc))
        filename = huc_path(huc_8_in_12)
        with fiona.open(filename, 'r') as fid:
            matching = [h for h in fid if h['properties']['HUC%i'%include].startswith(huc)]
            profile = fid.profile.copy()
        if len(matching) == 0:
            raise RuntimeError("Invalid collection of HUC?")

        matching_proj = []
        for match in matching:
            matching_proj.append(workflow.warp.warp_shape(match, profile['crs'], shp_profile['crs']))

        with fiona.open(os.path.join(packagedir, 'huc_%s_12s.shp'%huc), 'w', 'ESRI Shapefile', **shp_profile) as dst_shp:
            for match in matching_proj:
                dst_shp.write(match)

    return packagedir
            
        
