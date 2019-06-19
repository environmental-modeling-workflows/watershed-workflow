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

class FileManagerNHDPlus(workflow.sources.manager_mixins.FileManagerMixin_HUCs):
    def __init__(self):
        super().__init__('National Hydrography Dataset Plus High Resolution (NHDPlus HR)', 4, 12)
        self.names = workflow.sources.names.Names(self.name,
                                             'hydrography',
                                             'NHDPlus_H_{}_GDB',
                                             'NHDPlus_H_{}.gdb')

    def _get_hucs(self, hucstr, level):
        """Loads HUCs from file, no error checking or coordinate transformation."""
        # download the file
        filename = self._download(hucstr[0:self.file_level])

        # read the file
        layer = 'WBDHU{}'.format(level)
        logging.debug("{}: opening '{}' layer '{}' for HUCs in '{}'".format(self.name, filename, layer, hucstr))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            things = [h for h in fid if h['properties']['HUC{:d}'.format(level)].startswith(hucstr)]
            profile = fid.profile
        return profile, things
        
    def get_hydro(self, shape, crs=None, hint=None, intersect=None):
        """Downloads and reads hydrography in this shape.

        shape     | either a fiona shape object, a shapely shape, or a HUC
        crs       | crs of the shape (not required if shape is a HUC)
        hint      | If shape is not a HUC, indicates at least the 4-digit 
                  | HUC in which the shape exists.
        intersect | If None, only filters for the bounding box of 
                  | shape.  If intersect == 'intersects', then keeps all 
                  | segments that intersects shape.  If intersect == 
                  | 'contains', then only internal objects.

        TODO: re-write this to find the HUC via REST API to USGS
        instead of relying on user to supply a hint.
        """
        if type(shape) is str:
            # shape is a HUC: load the containing huc
            if crs is None:
                crs = workflow.conf.default_crs()

            containing_huc = source_utils.huc_str(shape)
            shp_profile, shape = self.get_huc(containing_huc, crs=crs)
            shply = workflow.utils.shply(shape['geometry'])
            if type(shply) is not shapely.geometry.Polygon:
                shply = shapely.ops.cascaded_union(shply)

        else:
            # shape is a shape, find the containig huc
            if crs is None:
                raise ValueError('{}: if providing get_hydro() with shape, must provide what CRS that shape is in.'.format(self.name))
            if hint is None or len(hint) < self.file_level:
                raise ValueError('{}: if providing get_hydro() with shape, must provide what hint of at least length {} to find the HUC.'.format(self.name, self.file_level))
            if type(shape) is not shapely.geometry.Polygon:
                shply = workflow.utils.shply(shape['geometry'])
                if type(shply) is not shapely.geometry.Polygon:
                    shply = shapely.ops.cascaded_union(shply)
            else:
                shply = shape
            assert(type(shply) is shapely.geometry.Polygon)
            containing_huc = source_utils.find_huc(shply, crs, hint, self)

        # find and open the hydrography layer        
        filename = self.names.file_name(containing_huc[0:self.file_level])
        layer = 'NHDFlowline'
        print('Opening "{}" file for streams'.format(filename))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            profile = fid.profile

            # map the shape to the file's crs for filtering
            shply = workflow.warp.warp_shapely(shply, crs, profile['crs'])

            # filter
            if not intersect:
                rivers = [r for (i,r) in fid.items(bbox=shply.bounds)]
            elif intersect == 'intersects':
                rivers = [r for (i,r) in fid.items(bbox=shply.bounds) if shply.intersects(workflow.utils.shply(r['geometry']))]
            elif intersect == 'contains':
                rivers = [r for (i,r) in fid.items(bbox=shply.bounds) if shply.contains(workflow.utils.shply(r['geometry']))]
                
            self._native_crs = profile['crs']

        # round
        workflow.utils.round(rivers, workflow.conf.rcParams['digits'])

        # map to the target crs
        for river in rivers:
            workflow.warp.warp_shape(river, self._native_crs, crs)
        profile['crs'] = crs
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
    
    
