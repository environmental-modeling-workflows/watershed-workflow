"""Helper mixins for workfing with subclassed mixins."""

import os, sys
import logging
import fiona
import shapely
import attr

import workflow.sources.utils as source_utils
import workflow.conf
import workflow.sources.names
import workflow.utils
import workflow.warp

@attr.s
class FileManagerMixin_HUCs:
    """Mixin class for working with managers that load HUCs.

    Subclasses of this should provide the following methods:

      _get_huc(self, huc,level)
                which does the actual loading.

    This class deals with providing the expected public interface,
    coordinate transformations, rounding, etc.
    """
    name = attr.ib(type=str)
    file_level = attr.ib(type=int)
    lowest_level = attr.ib(type=int)


    def get_huc(self, huc, crs=None):
        """Download and read a HUC file.

        crs provides the output coordinate system, 
        default set by workflow.conf.rcParams.
        """
        
        profile, hucs = self.get_hucs(huc, crs=crs)
        assert(len(hucs) is 1)
        return profile, hucs[0]
    

    def get_hucs(self, huc, level=None, crs=None):
        """Downloads and reads a HUC file for all subhucs of a given level.

        crs provides the output coordinate system, 
        default set by workflow.conf.rcParams.
        """
        hucstr = source_utils.huc_str(huc)
        huc_level = len(hucstr)
        if crs is None:
            crs = workflow.conf.default_crs()

        if huc_level < self.file_level:
            raise ValueError("{}: files are organized at HUC level {}, so cannot ask for a larger HUC than that level.".format(self.name, self.file_level))
        elif huc_level > self.lowest_level:
            raise ValueError("{}: files are have at max level {}.".format(self.name, self._lowest_level))

        if level is None:
            level = huc_level
        else:
            if level < huc_level:
                raise ValueError("{}: cannot ask for HUC level {} in a HUC of level {}".format(self.name, level,huc_level))

        # download/read the file
        profile, things = self._get_hucs(hucstr, level)

        # round
        workflow.utils.round(things, workflow.conf.rcParams['digits'])

        # map to the target crs
        self._native_crs = profile['crs']
        if crs != 'native':
            for thing in things:
                workflow.warp.warp_shape(thing, self._native_crs, crs)
            profile['crs'] = crs

        return profile, things


    
