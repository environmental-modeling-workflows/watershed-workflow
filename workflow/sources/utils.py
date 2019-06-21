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
    if os.path.isfile(location) and force:
        os.remove(location)

    if not os.path.isfile(location):
        logging.info('Downloading: "%s"'%url)
        logging.info('         to: "%s"'%location)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(location, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    return os.path.isfile(location)

def unzip(filename, to_location):
    """Unzip the corresponding, assumed to exist, zipped DEM into the DEM directory."""
    logging.info('Unzipping: "%s"'%filename)
    logging.info('       to: "%s"'%to_location)

    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(to_location)
    except zipfile.BadZipFile as err:
        logging.error('Failed to unzip: "{}"'.format(filename))
        logging.error('Likely this is the result of a previous job failing, partial download, internet connection issues, or other failed download.  Try removing the file, which will result in it being re-downloaded.')
        raise err
    return to_location

def move(filename, to_location):
    """Move a file to a folder."""
    logging.info('Moving: "%s"'%filename)
    logging.info('    to: "%s"'%to_location)
    shutil.move(filename, to_location)
