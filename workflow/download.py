"""URLs for collecting data."""

import sys,os
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
            args = self.format_args(args)
            return os.path.join(self.data_dir(), self.folder_template.format(*args))
        else:
            self.data_dir()

    def download_base(self, *args):
        fname = self.download_template.format(*args)
        if self.zip:
            fname += '.zip'

    def download(self, *args):
        return os.path.join(self.zip_dir(), self.download_base(*args))

    def url(self, *args):
        return self._url + self.download_base()

    def file_name_base(self, *args):
        args = self.format_args(args)
        return self.file_template.format(*args)

    def file_name(self, *args):
        return os.path.join(self.folder_name(*args), self.file_name_base(*args))
    
class HucFileSystem(FileSystem):
    """A FileSystem class based on HUC digits."""
    def __init__(self, digits, *args, **kwargs):
        super(FileSystem,self).__init__(*args, **kwargs)
        self.digits = digits

    def format_args(self, huc):
        huc = workflow.conf.huc_str(huc)
        huc_level = len(huc)
        assert(len(huc) >= self.digits)
        return huc[0:self.digits], huc_level


class LatLonFileSystem(FileSystem):
    """A FileSystem class based upon lat/lon"""
    def format_args(self, lat, lon):
        if type(lat) is int:
            lat = 'n%02i'%lat
        if type(lon) is int:
            lon = 'w%03i'%lon
        return [lat,lon]


class FileManager:
    """A class that actually manages the files."""
    def __init__(self, filesystems):
        self.filesystems = filesystems

    def download(self, *args, force=False):
        for fs in self.filesystems:
            filename = fs.filename(*args)
            try:
                if not os.path.isfile(filename) or force:
                    url = fs.url(*args)
                    downloadfile = fs.download(*args)
                    _download(url, downloadfile, force)
                    if fs.zip:
                        _unzip(downloadfile, fs.folder_name(*args))
                    else:
                        _rename(downloadfile, fs.folder_name(*args))
            except Exception as err:
                logging.info(str(err))
            else:
                logging.info('success')
                return fs.file_name(*args)

class TiledFileManager(FileManager):
    """A class that manages tiled downloads."""
    def download(self, bounds, force=False):
        logging.info('Collecting tiles in: "%r"'%(bounds))
        latlons = []
        dems = []
        for west in range(bounds[0], bounds[2]):
            for north in range(bounds[1]+1, bounds[3]+1):
                latlons.append((north,-west))
                dems.append(super(TiledFileManager,self).download(bounds,force))
        return dems


        
        


def download_huc(huc):
    """Ensures a given huc is downloaded, and returns the containing filename."""
    huc = workflow.conf.huc_str(huc)
    logging.debug('Collecting HUC: "%s"'%(huc))
    filename = workflow.conf.huc_path(huc)
    if not os.path.isfile(filename):
        huc2 = huc[0:2]
        filebase = _download_huc(huc2)
        if not filebase:
            raise RuntimeError("Cannot download HUC file for %s"%huc2)
        success = _unzip_huc(filebase, huc2)
        if not success:
            raise RuntimeError("Error in unzipping HUC file for %s"%huc2)
    return filename


def download_hydro(huc):
    """Ensures hydrography data is downloaded."""
    huc = workflow.conf.huc_str(huc)
    filename = workflow.conf.hydro_path(huc)


    assert len(huc) >= 8
    huc = huc[0:8]
    logging.debug('Collecting HUC Hydrography: "%s"'%(huc))
    if not os.path.isfile(filename):
        filebase = _download_hydro(huc)
        if not filebase:
            raise RuntimeError("Cannot download hydrography file for %s"%huc)
        success = _unzip_hydro(filebase, huc)
        if not success:
            raise RuntimeError("Error in unzipping hydrography file for %s"%huc)
    return filename


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


def _download_dem(lat,lon, force=False):
    """Downloads a dem, storing it in the right place

    Force clobbers and re-downloads the file.
    """
    for i in range(len(rc['NED file templates'])):
        try:
            filebase = _dem_filename_base(lat,lon,i)
            zipfilename = _dem_zip(filebase)
            url = _dem_url(filebase)
            success = _download(url, zipfilename, force)
        except requests.exceptions.HTTPError:
            continue
        else:
            return filebase
    return False

def _unzip_dem(filebase, lat, lon):
    """Unzip the corresponding, assumed to exist, zipped DEM into the DEM directory."""
    zipfilename = _dem_zip(filebase)
    target_location = os.path.join(rc['data dir'], rc['dem data dir'])

    if type(lat) is int:
        lat = 'n%i'%lat
    if type(lon) is int:
        lon = 'w%i'%lon
    filebase0 = rc['NED file templates'][0]%(lat,lon)
    to_location = os.path.join(rc['data dir'], rc['dem data dir'], filebase0+'.img')

    if not os.path.isfile(to_location):
        logging.info('Unzipping: "%s" \n  to: "%s"'%(zipfilename, target_location))
        logging.info('   and moving to: "%s"'%(to_location))

        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(target_location, members=[filebase+'.img',])
        if filebase != filebase0:
            from_location = os.path.join(rc['data dir'], rc['dem data dir'], filebase+'.img')
            os.rename(from_location, to_location)
    return to_location


def _huc_filename_base(huc2, templatenum=0):
    """Base filename of HUC data"""
    if type(huc2) is int:
        huc2 = '%02i'%huc2
    return rc['HUC file templates'][templatenum]%huc2

def _huc_url(filebase):
    """URL, filename for HUC data shapefile """
    return rc['HUC base URL']+filebase+'.zip'

def _huc_zip(filebase):
    """Return the name of the downloaded zip."""
    return os.path.join(rc['data dir'], rc['HUC data dir'], 'zips', filebase+'.zip')

def _download_huc(huc2, force=False):
    """Downloads a huc, storing it in the right place

    Force clobbers and re-downloads the file.
    """
    for i in range(len(rc['HUC file templates'])):
        try:
            filebase = _huc_filename_base(huc2, i)
            zipfilename = _huc_zip(filebase)
            url = _huc_url(filebase)
            success = _download(url, zipfilename, force)
        except requests.exceptions.HTTPError:
            continue
        else:
            return filebase
    return False

def _unzip_huc(filebase, huc2):
    """Unzip the corresponding, assumed to exist, zipped HUC into the HUC directory."""
    zipfilename = _huc_zip(filebase)
    target_location = os.path.join(rc['data dir'], rc['HUC data dir'], 'tmp-'+filebase)
    to_location = os.path.join(rc['data dir'], rc['HUC data dir'], filebase)

    if not os.path.isdir(os.path.join(to_location, filebase)):
        if not os.path.isdir(target_location):
            os.mkdir(target_location)
            logging.info('Unzipping: "%s" \n  to: "%s"'%(zipfilename, target_location))
            
            with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
                zip_ref.extractall(target_location)

        logging.info('   and moving to: "%s"'%(to_location))
        os.rename(os.path.join(target_location,'Shape'), to_location)
        os.rmdir(target_location)
    return os.path.isdir(os.path.join(to_location, filebase))


def _hydro_filename_base(huc8, templatenum=0):
    """Base filename of HUC data"""
    if type(huc8) is int:
        huc8 = '%08i'%huc8
    return rc['hydrography file templates'][templatenum]%huc8

def _hydro_url(filebase):
    """URL, filename for HUC data shapefile """
    return rc['hydrography base URL']+filebase+'.zip'

def _hydro_zip(filebase):
    """Return the name of the downloaded zip."""
    return os.path.join(rc['data dir'], rc['hydrography data dir'], 'zips', filebase+'.zip')


def _download_hydro(huc8, force=False):
    """Downloads a huc, storing it in the right place

    Force clobbers and re-downloads the file.
    """
    for i in range(len(rc['hydrography file templates'])):
        try:
            filebase = _hydro_filename_base(huc8, i)
            zipfilename = _hydro_zip(filebase)
            url = _hydro_url(filebase)
            success = _download(url, zipfilename, force)
        except requests.exceptions.HTTPError:
            continue
        else:
            return filebase
    return False

def _unzip_hydro(filebase, huc8):
    """Unzip the corresponding, assumed to exist, zipped HUC into the HUC directory."""
    zipfilename = _hydro_zip(filebase)
    target_location = os.path.join(rc['data dir'], rc['hydrography data dir'], 'tmp-'+filebase)
    to_location = os.path.join(rc['data dir'], rc['hydrography data dir'], filebase)

    if not os.path.isdir(target_location):
        os.mkdir(target_location)
        logging.info('Unzipping: "%s" \n  to: "%s"'%(zipfilename, target_location))
            
        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(target_location)

    logging.info('   and moving to: "%s"'%(to_location))
    os.rename(os.path.join(target_location, 'Shape'), to_location)
    os.rmdir(target_location)
    return os.path.isdir(os.path.join(to_location))






