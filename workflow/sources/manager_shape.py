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
    
    def get_shape(self, filter=None, crs=None, squeeze=True):
        """Gets all shapes in a file that meet a specific conditional.

        Note filter must be of the form:
        def filter(index, obj):
            return bool
        """
        
        if filter is None:
            filter = lambda i,a: True
        if crs is None:
            crs = self._crs
        
        with fiona.open(self._filename, 'r') as fid:
            profile = fid.profile
            self._native_crs = profile['crs']
            things = [shp for (i,shp) in enumerate(fid) if filter(i,shp)]

        workflow.utils.round(things, workflow.conf.rcParams['digits'])
        
        if crs is not None and crs != 'native':
            for thing in things:
                workflow.warp.warp_shape(thing, self._native_crs, crs)
            profile['crs']['init'] = crs

        if squeeze and len(things) is 1:
            things = things[0]
        return profile, things

    def get_native_crs(self):
        if self._native_crs is None:
            with fiona.open(self._filename,'r') as fid:
                self._native_crs = fid.profile['crs']
        return self._native_crs
    
    
    
