"""Utilities for working with sources."""

import sys, os
import logging
import requests
import shutil
import numpy as np
import shapely
import math
import urllib.request
import attr

import watershed_workflow.utils
import watershed_workflow.config

# register 7z zip/unzip
from py7zr import pack_7zarchive, unpack_7zarchive
shutil.register_archive_format('7zip', pack_7zarchive, description='7zip archive')
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


def get_code(fiona_or_shply_obj, level):
    """Gets the huc string from a HUC shape."""
    try:
        prop = fiona_or_shply_obj.properties
    except AttributeError:
        prop = fiona_or_shply_obj['properties']

    key = 'HUC{:d}'.format(level)
    try:
        return prop[key]
    except KeyError:
        return prop[key.lower()]


def huc_str(huc):
    """Converts a huc int or string to a standard-format huc string."""
    if type(huc) is str:
        if len(huc) % 2 == 1:
            huc = "0" + huc
    elif type(huc) is int:
        digits = math.ceil(math.log10(huc))
        if digits % 2 == 1:
            digits += 1
        huc = ("%%0%ii"%digits) % huc
    else:
        raise RuntimeError("Cannot convert type %r to huc" % type(huc))
    return huc


def download(url, location, force=False, **kwargs):
    """Download a file from a URL to a location.  If force, clobber whatever is there.

    Note that kwargs are supplied to the requests call.
    """
    if os.path.isfile(location) and force:
        os.remove(location)

    if not os.path.isfile(location):
        logging.info('Downloading: "%s"' % url)
        logging.info('         to: "%s"' % location)
        verify = watershed_workflow.config.rcParams['DEFAULT']['ssl_cert']
        logging.info('       cert: "%s"' % verify)
        if verify == "True":
            verify = True
        elif verify == "False":
            verify = False

        # with requests.get(url, stream=True, verify=verify) as r:
        #     r.raise_for_status()
        #     with open(location, 'wb') as f:
        #         for chunk in r.iter_content(chunk_size=128):
        #             f.write(chunk)

        with requests.get(url, stream=True, verify=verify, **kwargs) as r:
            r.raise_for_status()
            with open(location, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    return os.path.isfile(location)


def download_progress_bar(url, location, force=False):
    """Download a file from URL to location, with a progress bar.

    If force, clobber whatever is there.
    """
    from tqdm.autonotebook import tqdm

    if os.path.isfile(location) and force:
        os.remove(location)

    if not os.path.isfile(location):
        logging.info('Downloading: "%s"' % url)
        logging.info('         to: "%s"' % location)
        verify = watershed_workflow.config.rcParams['DEFAULT']['ssl_cert']
        logging.info('       cert: "%s"' % verify)
        if verify == "True":
            verify = True
        elif verify == "False":
            verify = False

        r = requests.get(url, stream=True)
        total = int(r.headers.get('content-length', 0))
        with open(location, 'wb') as file, tqdm(desc=os.path.split(location)[-1],
                                                total=total,
                                                unit='iB',
                                                unit_scale=True,
                                                unit_divisor=1024,
                                                ) as bar:
            for data in r.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    return os.path.isfile(location)


def unzip(filename, to_location, format=None):
    """Unzip the corresponding, assumed to exist, zipped DEM into the DEM directory."""
    logging.info(f'Unzipping: "{filename}"')
    logging.info(f'       to: "{to_location}"')

    import shutil
    shutil.unpack_archive(filename, to_location, format)
    return to_location


def move(filename, to_location):
    """Move a file to a folder."""
    logging.info('Moving: "%s"' % filename)
    logging.info('    to: "%s"' % to_location)
    shutil.move(filename, to_location)
