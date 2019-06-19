"""Utilities for working with sources."""

import sys, os
import logging
import requests
import zipfile
import shutil
import numpy as np
import shapely

import workflow.utils

def huc_str(huc):
    """Converts a huc int or string to a standard-format huc string."""
    if type(huc) is str:
        if len(huc)%2 == 1:
            huc = "0"+huc
    elif type(huc) is int:
        digits = math.ceil(math.log10(huc))
        if digits % 2 == 1:
            digits += 1
        huc = ("%%0%ii"%digits)%huc
    else:
        raise RuntimeError("Cannot convert type %r to huc"%type(huc))
    return huc



def _in_huc(shply, huc_shp):
    """Checks whether shp is in HUC"""
    huc_shply = workflow.utils.shply(huc_shp['geometry'])
    if huc_shply.contains(shply):
        return 2
    elif huc_shply.intersects(shply):
        return 1
    else:
        return 0

def _find_huc(shply, crs, hint, source):
    """Searches in hint to find shp."""
    print('searching: %s'%hint)
    hint_level = len(hint)
    search_level = hint_level + 2
    if search_level > source.lowest_level:
        return hint

    profile, subhucs = source.get_hucs(hint, search_level, crs)
    for subhuc in subhucs:
        inhuc = _in_huc(shply, subhuc)
        if inhuc == 2:
            # fully contained in try_huc, recurse
            hname = subhuc['properties']['HUC{:d}'.format(search_level)]
            print('  subhuc: %s contains'%hname)
            return _find_huc(shply, crs, hname, source)
        elif inhuc == 1:
            hname = subhuc['properties']['HUC{:d}'.format(search_level)]
            print('  subhuc: %s partially contains'%hname)
            # partially contained in try_huc, return this
            return hint
        else:
            hname = subhuc['properties']['HUC{:d}'.format(search_level)]
            print('  subhuc: %s does not contain'%hname)
            
    assert(False)


def find_huc(shp, crs, hint, source, shrink=1.e-5):
    """Finds the smallest HUC containing shp, starting with a potential
    hint, i.e. '06' for Tennessee River Valley.
    """
    if type(shp) is shapely.geometry.Polygon:
        shply = shp
    else:
        shply = workflow.utils.shply(shp['geometry'])

    # must shrink the poly a bit in case it is close to or on a boundary
    radius = np.sqrt(shply.area/np.pi)
    shply_s = shply.buffer(-shrink*radius)

    hint = huc_str(hint)

    profile, hinthuc = source.get_huc(hint, crs=crs)
    inhuc = _in_huc(shply_s, hinthuc)
    if inhuc is not 2:
        raise RuntimeError("{}: shape not found in hinted HUC '{}'".format(source.name, hint))

    result = _find_huc(shply_s, crs, hint, source)
    return result

def download(url, location, force=False):
    """Download a file from a URL to a location.  If force, clobber whatever is there."""
    if os.path.isfile(location):
        if force:
            os.remove(location)
        else:
            return True
    try:
        logging.info('Downloading: "%s"'%url)
        logging.info('         to: "%s"'%location)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(location, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except requests.exceptions.HTTPError as err:
        logging.info('  ERROR: %r'%err)
        raise err
    else:
        logging.info('  SUCCESS')
    return os.path.isfile(location)

def unzip(filename, to_location):
    """Unzip the corresponding, assumed to exist, zipped DEM into the DEM directory."""
    logging.info('Unzipping: "%s"'%filename)
    logging.info('       to: "%s"'%to_location)
    
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(to_location)
    return to_location

def move(filename, to_location):
    """Move a file to a folder."""
    shutil.move(filename, to_location)
