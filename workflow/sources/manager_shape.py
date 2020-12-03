"""Basic manager for interacting with shapefiles.
"""

import attr
import fiona

import workflow.warp
import workflow.utils
import workflow.conf

@attr.s
class FileManagerShape:
    """A simple class for reading shapefiles.

    Parameter
    ---------
    filename : str
      Path to the shapefile.
    """
    _filename = attr.ib(type=str)
    
    def get_shape(self, *args, **kwargs):
        """Read the file and filter to get shapes, then ensures there is only one
        match.

        Parameters
        ----------
        See that of get_shapes().

        Returns
        -------
        :obj:`profile`
            Fiona profile of the shapefile.
        :obj:`list(Polygon)`
            List of fiona shapes that match the index or bounds.

        """
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

            if index_or_bounds is None or type(index_or_bounds) is int:
                if index_or_bounds is not None and index_or_bounds >= 0:
                    shps = [fid[index_or_bounds],]
                else:
                    shps = [s for s in fid]
            else:
                if crs is not None and not workflow.crs.equal(crs, workflow.crs.from_fiona(profile['crs'])):
                    bounds = workflow.warp.bounds(index_or_bounds, crs, workflow.crs.from_fiona(profile['crs']))
                shps = [s for (i,s) in fid.items(bbox=bounds)]

        return profile, shps
    
    
    
