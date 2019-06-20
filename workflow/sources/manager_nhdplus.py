"""Manager for interacting with USGS NHD+ datasets.
"""
import os, sys
import logging
import fiona
import shapely

import workflow.sources.utils as source_utils
import workflow.conf
import workflow.sources.names
import workflow.utils
import workflow.warp
import workflow.sources.manager_mixins

class FileManagerNHDPlus:
    def __init__(self):
        self.name = 'National Hydrography Dataset Plus High Resolution (NHDPlus HR)'
        self.file_level = 4
        self.lowest_level = 12
        self.names = workflow.sources.names.Names(self.name,
                                             'hydrography',
                                             'NHDPlus_H_{}_GDB',
                                             'NHDPlus_H_{}.gdb')

    def get_huc(self, huc):
        huc = source_utils.huc_str(huc)
        profile, hus = self.get_hucs(huc, len(huc))
        assert(len(hus) == 1)
        return profile, hus[0]

    def get_hucs(self, huc, level):
        """Loads HUCs from file."""
        huc = source_utils.huc_str(huc)
        huc_level = len(huc)

        # error checking on the levels, require file_level <= huc_level <= level <= lowest_level
        if self.lowest_level < level:
            raise ValueError("{}: files include HUs at max level {}.".format(self.name, self._lowest_level))
        if level < huc_level:
            raise ValueError("{}: cannot ask for HUs at level {} contained in {}.".format(self.name, level, huc_level))
        if huc_level < self.file_level:
            raise ValueError("{}: files are organized at HUC level {}, so cannot ask for a larger HUC than that level.".format(self.name, self.file_level))

        # download the file
        filename = self._download(huc[0:self.file_level])

        # read the file
        layer = 'WBDHU{}'.format(level)
        logging.debug("{}: opening '{}' layer '{}' for HUCs in '{}'".format(self.name, filename, layer, huc))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            hus = [hu for hu in fid if hu['properties']['HUC{:d}'.format(level)].startswith(huc)]
            profile = fid.profile
        return profile, hus
        
    def get_hydro(self, bounds, bounds_crs, huc_hint):
        """Downloads and reads hydrography within these bounds.

        Note this requires a HUC hint of a level 4 HUC which contains bounds.
        """
        huc_hint = source_utils.huc_str(huc_hint)
        hint_level = len(huc_hint)

        # error checking on the levels, require file_level <= huc_level <= lowest_level
        if hint_level < self.file_level:
            raise ValueError("{}: files are organized at HUC level {}, so cannot ask for a larger HUC than that level.".format(self.name, self.file_level))
        
        # download the file
        filename = self._download(huc_hint[0:self.file_level])
        
        # find and open the hydrography layer        
        filename = self.names.file_name(huc_hint[0:self.file_level])
        layer = 'NHDFlowline'
        logging.debug("{}: opening '{}' layer '{}' for streams in '{}'".format(self.name, filename, layer, bounds))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            profile = fid.profile
            bounds = workflow.warp.warp_bounds(bounds, bounds_crs, profile['crs'])
            rivers = [r for (i,r) in fid.items(bbox=bounds)]
        return profile, rivers
            
    def _url(self, hucstr):
        """Use the REST API to find the URL."""
        import requests
        rest_url = 'https://viewer.nationalmap.gov/tnmaccess/api/products'

        hucstr = hucstr[0:self.file_level]
        r = requests.get(rest_url, params={'datasets':self.name,
                                           'polyType':'huc{}'.format(self.file_level),
                                           'polyCode':hucstr})
        json = r.json()
        matches = [m for m in json['items'] if hucstr in m['title']]
        if len(matches) == 0:
            raise ValueError('{}: not able to find HUC {}'.format(self.name, hucstr))
        return matches[0]['downloadURL']

    def _download(self, hucstr, force=False):
        """Download the data."""
        # check directory structure
        os.makedirs(self.names.data_dir(), exist_ok=True)
        os.makedirs(self.names.folder_name(hucstr), exist_ok=True)

        work_folder = self.names.raw_folder_name(hucstr)
        os.makedirs(work_folder, exist_ok=True)

        filename = self.names.file_name(hucstr)
        if not os.path.exists(filename) or force:
            url = self._url(hucstr)

            downloadfile = os.path.join(work_folder, url.split("/")[-1])
            if not os.path.exists(downloadfile) or force:
                logging.debug("Attempting to download source for target '%s'"%filename)
                source_utils.download(url, downloadfile, force)
                source_utils.unzip(downloadfile, work_folder)

                # hope we can find it?
                gdb_files = [f for f in os.listdir(work_dir) if f.endswith('.gdb')]
                assert(len(gdb_files) == 1)
                source_utils.move(os.path.join(work_dir, gdb_files[0]), filename)

        if not os.path.exists(filename):
            raise RuntimeError("Cannot find or download file for source target '%s'"%filename)
        return filename
    
    
