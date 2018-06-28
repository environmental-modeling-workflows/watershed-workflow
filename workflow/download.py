"""URLs for collecting data."""

import sys,os
import requests
import requests.exceptions
import zipfile
import logging
import shutil

import workflow.conf
from workflow.conf import rcParams as rc


def download_dem(bounds):
    """Collect all files that tile a given set of bounds."""
    logging.info('Collecting DEM tiles in: "%r"'%(bounds))
    latlons = []
    dems = []
    for west in range(bounds[0], bounds[2]):
        for north in range(bounds[1]+1, bounds[3]+1):
            latlons.append((north,-west))
            dems.append(_download_dem(north, -west))

    files = []
    for (lat,lon), filebase in zip(latlons, dems):
        files.append(_unzip_dem(filebase, lat, lon))
    return files


def download_huc(huc):
    """Ensures a given huc is downloaded, and returns the containing filename."""
    huc = workflow.conf.huc_str(huc)
    logging.info('Collecting HUC: "%s"'%(huc))
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
    assert len(huc) == 8
    logging.info('Collecting HUC Hydrography: "%s"'%(huc))
    filename = workflow.conf.hydro_path(huc)
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


def _dem_filename_base(lat, lon, templatenum=0):
    """Base filename of DEM data"""
    if type(lat) is int:
        lat = 'n%02i'%lat
    if type(lon) is int:
        lon = 'w%03i'%lon
    return rc['NED file templates'][templatenum]%(lat,lon)

def _dem_url(filebase):
    """URL, filename for DEM data

    10m proeduct from USGS NED.  Lat,lon are the upper-left corner, in format, e.g.:

         n37, w86
    """
    return rc['NED base URL']+filebase+'.zip'

def _dem_zip(filebase):
    """Return the name of the downloaded zip."""
    return os.path.join(rc['data dir'], rc['dem data dir'], 'zips', filebase+'.zip')

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






