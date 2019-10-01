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
    
    def get_shape(self, *args, **kwargs):
        profile, shps = self.get_shapes(*args, **kwargs)
        if len(shps) is not 1:
            raise RuntimeError("Filtered shapefile contains more than one match.")
        return profile, shps[0]
    
    def get_shapes(self, index_or_bounds=-1, crs=None):
        """Read the file and filter to get shapes.

        This accepts either an index, which is the integer index of the desired
        shape in the file, or a bounding box.  

        Parameters
        ----------
        index_or_bounds : int or :obj:`[xmin, ymin, xmax, ymax]`
            Index of the requested shape in filename, or bounding box to filter 
            shapes, or defaults to -1 to get them all.

        crs : :obj:`crs`
            Coordinate system of the bounding box (or None if index).

        Returns
        -------
        :obj:`profile`
            Fiona profile of the shapefile.
        :obj:`list(Polygon)`
            List of fiona shapes that match the index or bounds.
        
        """
        with fiona.open(self._filename, 'r') as fid:
            profile = fid.profile

            if type(index_or_bounds) is int:
                if index_or_bounds >= 0:
                    shps = [fid[index_or_bounds],]
                else:
                    shps = fid[:]
            else:
                if crs['init'] != profile['crs']['init']:
                    bounds = workflow.warp.bounds(index_or_bounds, crs, profile['crs'])
                shps = [s for (i,s) in fid.items(bbox=bounds)]

        return profile, shps
    
    
    
