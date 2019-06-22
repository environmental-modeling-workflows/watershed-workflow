"""Basic manager for interacting with shapefiles.
"""

import attr
import fiona

import workflow.warp
import workflow.utils
import workflow.conf

@attr.s
class FileManagerShape:
    _filename = attr.ib(type=str)
    _crs = attr.ib(type=str, default=workflow.conf.default_crs())
    _native_crs = attr.ib(default=None)
    
    def get_shapes(self, filter=None):
        """Gets all shapes in a file that meet a specific conditional.

        Note filter must be of the form:
        def filter(index, obj):
            return bool
        """
        
        if filter is None:
            filter = lambda i,a: True
        
        with fiona.open(self._filename, 'r') as fid:
            profile = fid.profile
            shps = [shp for (i,shp) in enumerate(fid) if filter(i,shp)]

        return profile, shps
    
    
    
