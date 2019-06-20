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
