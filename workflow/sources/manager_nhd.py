import os, sys
import logging
import fiona
import shapely
import attr
import requests

import workflow.sources.utils as source_utils
import workflow.conf
import workflow.sources.names
import workflow.utils
import workflow.warp


@attr.s
class _FileManagerNHD:
    """Manager for interacting with USGS National Hydrography Datasets.

    Note that this includes NHD, NHDPlus, and WBD -- this class should not
    be used directly but instead use one of the derived classes:

    * `manager_nhd.FileManagerNHD`
    * `manager_nhd.FileManagerNHDPlus`
    * `manager_nhd.FileManagerWBD`

    Watershed Workflow leverages the Watershed Boundary Dataset (WBD) and the
    National Hydrography Dataset (NHD), USGS and EPA datasets available at
    multiple resolutions to represent United States watersheds, including
    Alaska [NHD]_.  Also used is the NHD Plus dataset, an augmented dataset
    built on watershed boundaries and elevation products.  By default, the
    1:100,000 High Resolution datasets are used.  Data is discovered through
    The National Map's [TNM]_ REST API, which allows querying for data files
    organized by HUC and resolution via HTTP POST requests, providing
    direct-download URLs.  Files are downloaded on first request, unzipped, and
    stored in the data library for future use.  Currently, files are indexed by
    2-digit (WBD), 4-digit (NHD Plus HR) and 8-digit (NHD) HUCs.

    .. [NHD] https://www.usgs.gov/core-science-systems/ngp/national-hydrography
    .. [TNM] https://viewer.nationalmap.gov/help/documents/TNMAccessAPIDocumentation/TNMAccessAPIDocumentation.pdf

    """
    name = attr.ib(type=str)
    file_level = attr.ib(type=int)
    lowest_level = attr.ib(type=int)
    name_manager = attr.ib()

    def get_huc(self, huc):
        """Get the specified HUC in its native CRS.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        hu : dict
          Fiona shape object representing the hydrologic unit.

        Note this finds and downloads files as needed.
        """        
        huc = source_utils.huc_str(huc)
        profile, hus = self.get_hucs(huc, len(huc))
        assert(len(hus) == 1)
        return profile, hus[0]

    def get_hucs(self, huc, level):
        """Get all sub-catchments of a given HUC level within a given HUC.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code
        level : int
          Level of requested sub-catchments.  Must be larger or equal to the
          level of the input huc.

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        hus : list(dict)
          List of fiona shape objects representing the hydrologic units.

        Note this finds and downloads files as needed.
        """        
        huc = source_utils.huc_str(huc)
        huc_level = len(huc)

        # error checking on the levels, require file_level <= huc_level <= level <= lowest_level
        if self.lowest_level < level:
            raise ValueError("{}: files include HUs at max level {}.".format(self.name, self.lowest_level))
        if level < huc_level:
            raise ValueError("{}: cannot ask for HUs at level {} contained in {}.".format(self.name, level, huc_level))
        if huc_level < self.file_level:
            raise ValueError("{}: files are organized at HUC level {}, so cannot ask for a larger HUC than that level.".format(self.name, self.file_level))

        # download the file
        filename = self._download(huc[0:self.file_level])
        logging.info('Using HUC file "{}"'.format(filename))
        

        # read the file
        layer = 'WBDHU{}'.format(level)
        logging.debug("{}: opening '{}' layer '{}' for HUCs in '{}'".format(self.name, filename, layer, huc))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            hus = [hu for hu in fid if hu['properties']['HUC{:d}'.format(level)].startswith(huc)]
            profile = fid.profile
        profile['always_xy'] = True
        return profile, hus
        
    def get_hydro(self, huc, bounds=None, bounds_crs=None):
        """Get all reaches within a given HUC and/or coordinate bounds.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code
        bounds : [xmin, ymin, xmax, ymax], optional
          Coordinate bounds to filter reaches returned.  If this is provided,
          bounds_crs must also be provided.
        bounds_crs : CRS, optional
          CRS of the above bounds.

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        reaches : list(dict)
          List of fiona shape objects representing the stream reaches.

        Note this finds and downloads files as needed.
        """        
        if 'WBD' in self.name:
            raise RuntimeError('{}: does not provide hydrographic data.'.format(self.name))
        
        huc = source_utils.huc_str(huc)
        hint_level = len(huc)

        # try to get bounds if not provided
        if bounds is None:
            # can we infer a bounds by getting the HUC?
            profile, hu = self.get_huc(huc)
            bounds = workflow.utils.bounds(hu)
            bounds_crs = workflow.crs.from_fiona(profile['crs'])
        
        # error checking on the levels, require file_level <= huc_level <= lowest_level
        if hint_level < self.file_level:
            raise ValueError("{}: files are organized at HUC level {}, so cannot ask for a larger HUC than that level.".format(self.name, self.file_level))
        
        # download the file
        filename = self._download(huc[0:self.file_level])
        logging.info('Using Hydrography file "{}"'.format(filename))
        
        # find and open the hydrography layer        
        filename = self.name_manager.file_name(huc[0:self.file_level])
        layer = 'NHDFlowline'
        logging.debug("{}: opening '{}' layer '{}' for streams in '{}'".format(self.name, filename, layer, bounds))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            profile = fid.profile
            bounds = workflow.warp.bounds(bounds, bounds_crs, workflow.crs.from_fiona(profile['crs']))
            rivers = [r for (i,r) in fid.items(bbox=bounds)]
        return profile, rivers
            
    def _url(self, hucstr):
        """Use the USGS REST API to find the URL to download a file for a given huc.

        Parameters
        ----------
        hucstr : str
          The USGS Hydrologic Unit Code

        Returns
        -------
        url : str
          The URL to download a file containing shapes for the HUC.
        """
        rest_url = 'https://viewer.nationalmap.gov/tnmaccess/api/products'
        hucstr = hucstr[0:self.file_level]

        def attempt(params):        
            r = requests.get(rest_url, params=params)
            try:
                r.raise_for_status()
            except Exception as e:
                logging.error(e)
                return 1,e
                
            json = r.json()

            # this feels hacky, but it does not appear that USGS has their
            # 'prodFormat' get option or 'format' return json value
            # working correctly.
            matches = [m for m in json['items']]

            # filter for GDBs
            matches = [m for m in matches if 'downloadURL' in m and 'GDB' in m['downloadURL']]

            # filter for title contains HUC string
            matches_f = [m for m in matches if hucstr in m['title'].split()]
            if len(matches_f) > 0:
                matches = matches_f
        
            if len(matches) == 0:
                return 1, '{}: not able to find HUC {}'.format(self.name, hucstr)
            if len(matches) > 1:
                logging.error('{}: too many matches for HUC {} ({})'.format(self.name, hucstr, len(matches)))
                for m in matches:
                    logging.error(' {}\n   {}'.format(m['title'], m['downloadURL']))
                return 1, '{}: too many matches for HUC {}'.format(self.name, hucstr)
            return 0, matches[0]['downloadURL']

        # cheaper if it works, may not work in alaska?
        a1 = attempt({'datasets':self.name,
                      'polyType':'huc{}'.format(self.file_level),
                      'polyCode':hucstr})
        if not a1[0]:
            return a1[1]

        # works more univerasally but is a BIG lookup, then filter locally
        a2 = attempt({'datasets':self.name})
        if not a2[0]:
            return a2[1]

        raise ValueError('{}: cannot find HUC {}'.format(self.name, hucstr))
        

    def _download(self, hucstr, force=False):
        """Find and download data from a given HUC.

        Parameters
        ----------
        hucstr : str
          The USGS Hydrologic Unit Code
        force : bool, optional
          If true, re-download even if a file already exists.

        Returns
        -------
        filename : str
          The path to the resulting downloaded dataset.
        """
        # check directory structure
        os.makedirs(self.name_manager.data_dir(), exist_ok=True)
        os.makedirs(self.name_manager.folder_name(hucstr), exist_ok=True)

        work_folder = self.name_manager.raw_folder_name(hucstr)
        os.makedirs(work_folder, exist_ok=True)

        filename = self.name_manager.file_name(hucstr)
        if not os.path.exists(filename) or force:
            url = self._url(hucstr)

            downloadfile = os.path.join(work_folder, url.split("/")[-1])
            if not os.path.exists(downloadfile) or force:
                logging.debug("Attempting to download source for target '%s'"%filename)
                source_utils.download(url, downloadfile, force)
                
            source_utils.unzip(downloadfile, work_folder)

            # hope we can find it?
            gdb_files = [f for f in os.listdir(work_folder) if f.endswith('.gdb')]
            assert(len(gdb_files) == 1)
            source_utils.move(os.path.join(work_folder, gdb_files[0]), filename)

        if not os.path.exists(filename):
            raise RuntimeError("Cannot find or download file for source target '%s'"%filename)
        return filename
    
    
class FileManagerNHDPlus(_FileManagerNHD):
    def __init__(self):
        name = 'National Hydrography Dataset Plus High Resolution (NHDPlus HR)'
        super().__init__(name, 4, 12,
                         workflow.sources.names.Names(name, 'hydrography',
                                                      'NHDPlus_H_{}_GDB',
                                                      'NHDPlus_H_{}.gdb'))

class FileManagerNHD(_FileManagerNHD):
    def __init__(self):
        name = 'National Hydrography Dataset (NHD)'
        super().__init__(name, 8, 12,
                         workflow.sources.names.Names(name, 'hydrography',
                                                      'NHD_H_{}_GDB',
                                                      'NHD_H_{}.gdb'))


class FileManagerWBD(_FileManagerNHD):
    def __init__(self):
        name = 'National Watershed Boundary Dataset (WBD)'
        super().__init__(name, 2, 12,
                         workflow.sources.names.Names(name, 'hydrography',
                                                      'WBD_{}_GDB',
                                                      'WBD_{}.gdb'))    
    
