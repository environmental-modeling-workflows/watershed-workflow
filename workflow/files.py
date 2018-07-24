"""Deals with collecting and curating data files."""

import sys,os
import numpy as np
import requests
import requests.exceptions
import zipfile
import logging
import shutil

import fiona

import workflow.conf
from workflow.conf import rcParams as rc

class FileSystem:
    """File system meta data for downloading a file."""
    def __init__(self, name, url, base_folder, folder_template, file_template,
                 download_template=None, zip=True):
        self.name = name
        self._url = url
        self.base_folder = base_folder
        self.folder_template = folder_template
        self.file_template = file_template

        if download_template is None:
            self.download_template = folder_template
        else:
            self.download_template = download_template
        self.zip = zip

    def data_dir(self):
        return os.path.join(rc['data dir'], self.base_folder)
    
    def zip_dir(self):
        return os.path.join(self.data_dir(), 'zips')

    def folder_name(self, *args):
        if self.folder_template is not None:
            args = self.format_args(*args)
            return os.path.join(self.data_dir(), self.folder_template.format(*args))
        else:
            return self.data_dir()

    def download_base(self, *args):
        fname = self.download_template.format(*args)
        if self.zip:
            fname += '.zip'

    def download(self, *args):
        return os.path.join(self.zip_dir(), self.download_base(*args))

    def url(self, *args):
        return self._url + self.download_base()

    def file_name_base(self, *args):
        args = self.format_args(*args)
        return self.file_template.format(*args)

    def file_name(self, *args):
        return os.path.join(self.folder_name(*args), self.file_name_base(*args))
    
class HucFileSystem(FileSystem):
    """A FileSystem class based on HUC digits."""
    def __init__(self, digits, *args, **kwargs):
        super(HucFileSystem,self).__init__(*args, **kwargs)
        self.digits = digits

    def format_args(self, *args):
        assert(len(args) > 0)
        huc = workflow.conf.huc_str(args[0])
        if len(args) > 1:
            size = args[1]
        else:
            size = len(args[0])
        assert(len(huc) >= self.digits)
        return huc[0:self.digits], size


class LatLonFileSystem(FileSystem):
    """A FileSystem class based upon lat/lon"""
    def format_args(self, lat, lon):
        if type(lat) is int:
            lat = 'n%i'%lat
        if type(lon) is int:
            lon = 'w%i'%lon
        return [lat,lon]


class FileManager:
    """A class that actually manages the files."""
    def __init__(self, filesystems):
        self.filesystems = filesystems

    def download(self, *args, force=False):
        for fs in self.filesystems:
            logging.debug("Attempting to download with args '%r'"%list(args))
            filename = fs.file_name(*args)
            logging.debug("Attempting to download '%s'"%filename)
            try:
                if not os.path.isfile(filename) or force:
                    url = fs.url(*args)
                    downloadfile = fs.download(*args)
                    _download(url, downloadfile, force)
                    if fs.zip:
                        _unzip(downloadfile, fs.folder_name(*args))
                    else:
                        _move(downloadfile, fs.folder_name(*args))
            except Exception as err:
                logging.info(str(err))
            else:
                logging.info('success')
                return fs.file_name(*args)
        raise RuntimeError('File not downloadable')

    def file_name(self, *args):
        for fs in self.filesystems:
            logging.debug("Searching '%s' for HUC '%s'"%(fs.name, args[0]))
            fname = fs.file_name(*args)
            logging.debug("   ...file '%s'"%fname)
            if os.path.isfile(fname):
                return fname
        raise RuntimeError("File not yet downloaded or found.")

class TiledFileManager(FileManager):
    """A class that manages tiled downloads."""
    def download(self, bounds, force=False):
        logging.info('Collecting tiles in: "%r"'%(bounds))
        latlons = []
        dems = []
        for west in range(bounds[0], bounds[2]):
            for north in range(bounds[1]+1, bounds[3]+1):
                dems.append(super(TiledFileManager,self).download(north, -west, force=force))
        logging.info('  collected: "%r"'%(dems))
        return dems


def _download(url, location, force=False):
    """Download a file from a URL to a location.  If force, clobber whatever is there."""
    if os.path.isfile(location):
        if force:
            os.remove(location)
        else:
            return True
    try:
        logging.info('Downloading: "%s" \n  to: "%s"'%(url, location))
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

def _unzip(filename, to_location):
    """Unzip the corresponding, assumed to exist, zipped DEM into the DEM directory."""
    logging.info('Unzipping: "%s" \n  to: "%s"'%(filename, to_location))
    
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(to_location)
    return to_location

def _move(filename, to_location):
    """Move a file to a folder."""
    shutil.move(filename, to_location)

def _normalize(list_of_shps, digits=7):
    """Rounds and standardizes shapefile formats to deal with multiple or single entry files."""
    for shp in list_of_shps:
        assert(type(shp['geometry']['coordinates']) is list)
        if len(shp['geometry']['coordinates']) is 0:
            continue
        if type(shp['geometry']['coordinates'][0]) is tuple:
            # single object
            coords = np.array(shp['geometry']['coordinates'], 'd').round(digits)
            if coords.shape[-1] is 3:
                coords = coords[:,0:2]
            assert(len(coords.shape) is 2)
            shp['geometry']['coordinates'] = coords
        else:
            # object collection
            for i,c in enumerate(shp['geometry']['coordinates']):
                coords = np.array(c,'d').round(digits)
                assert(len(coords.shape) is 2)
                if coords.shape[-1] is 3:
                    coords = coords[:,0:2]
            shp['geometry']['coordinates'][i] = coords
    return list_of_shps

    
def load_huc(huc, source, digits=7):
    """Reads a file to get a huc"""
    filename = source.file_name(huc)
    logging.debug("Searching '%s' for HUC '%s'"%(filename, huc))
    with fiona.open(filename, 'r') as fid:
        matching = [h for h in fid if h['properties']['HUC%i'%len(huc)] == huc]
        profile = fid.profile
    if len(matching) is not 1:
        raise RuntimeError("Invalid collection of HUC?")
    return profile, _normalize(matching, digits)[0]

def load_hucs_in(huc, source, size, digits=7):
    """Reads a file to get a huc"""
    filename = source.file_name(huc, size)
    with fiona.open(filename, 'r') as fid:
        matching = [h for h in fid if h['properties']['HUC%i'%size].startswith(huc)]
        profile = fid.profile
    return profile, _normalize(matching, digits)

def load_dem(filename, index=1):
    """Reads a file to get an image raster"""
    with rasterio.open(filename, 'r') as fid:
        profile = fid.profile.copy()
        array = fid.read(index)
    return profile, array

# def bounds_from_profile(profile):
#     """Uses a profile to determine a raster bounds.  

#     Note that if you have a file, this is equivalent to fid.bounds and
#     that should be preferred.
#     """
#     xmin, ymax = profile['affine'] * (0,0)
#     xmax, ymin = profile['affine'] * (profile['width'], profile['height'])
#     return [xmin, ymin, xmax, ymax]

def load_hydro(huc, source):
    """Returns the path to hydrography in this huc."""
    filename = source.file_name(huc)
    with fiona.open(filename, 'r') as fid:
        profile = fid.profile
        shps = [s for s in fid]
    return profile, _normalize(shps)
    
