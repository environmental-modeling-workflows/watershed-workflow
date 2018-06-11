"""Generates a collection of Polygons and DEMs for use in mesh making.

This should not really be used yet?  Instead prefer go.py
"""

import os,sys

import rasterio
import fiona

import workflow.conf
from workflow.conf import rcParams as rc
import workflow.clip
import workflow.warp

def generate_package(huc, include=12):
    """Gathers dems, shapefiles, etc, for making a mesh on a given huc"""
    huc = workflow.conf.huc_str(huc)

    # get the package directory
    packagedir = os.path.join(rc['data dir'], rc['packages data dir'], huc)
    if not os.path.isdir(packagedir):
        os.mkdir(packagedir)

    # get the shape of the containing HUC and the dem
    (shp_profile, shp), (img_profile, img_array) = workflow.clip.merge_and_clip(huc)

    # write dem to disk
    with rasterio.open(os.path.join(packagedir, 'huc_%s_dem.img'%huc), 'w', **img_profile) as dst_img:
        dst_img.write(img_array)

    # write shape to disk
    with fiona.open(os.path.join(packagedir, 'huc_%s.shp'%huc), 'w', **shp_profile) as dst_shp:
        dst_shp.write(shp)

    # now collect all HUC-12s in this HUC, and write a HUC12 shapefile
    if include is not None:
        if type(include) is int:
            include = "%i"%include

        huc_8_in_12 = huc + '0'*(12-len(huc))
        filename = workflow.conf.huc_path(huc_8_in_12)
        with fiona.open(filename, 'r') as fin:
            with fiona.open(os.path.join(packagedir, 'huc_%s_12s.shp'%huc), 'w', **fin.profile) as fout:
                matching = [h for h in fin if h['properties']['HUC%s'%include].startswith(huc)]
                assert len(matching) > 0
                for match in matching:  #matching_proj:
                    fout.write(match)

    return packagedir


if __name__ == "__main__":
    huc = sys.argv[-1]

    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('rasterio').setLevel(logging.INFO)
    logging.getLogger('Fiona').setLevel(logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    generate_package(huc)
    
    
