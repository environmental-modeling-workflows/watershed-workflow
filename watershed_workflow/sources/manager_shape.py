"""Basic manager for interacting with shapefiles.
"""

import attr
import fiona

import watershed_workflow.warp
import watershed_workflow.utils
import watershed_workflow.config
import watershed_workflow.crs


@attr.s
class FileManagerShape:
    """A simple class for reading shapefiles.

    Parameters
    ----------
    filename : str
      Path to the shapefile.
    """
    _filename = attr.ib(type=str)
    name = 'shapefile'

    def get_shape(self, *args, **kwargs):
        """Read the file and filter to get shapes, then ensures there is only one
        match.

        Parameters
        ----------
        See that of get_shapes().

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list(dict)
            List of fiona shapes that match the index or bounds.

        """
        profile, shps = self.get_shapes(*args, **kwargs)
        if len(shps) != 1:
            raise RuntimeError("Filtered shapefile contains more than one match.")
        return profile, shps[0]

    def get_shapes(self, index_or_bounds=-1, crs=None):
        """Read the file and filter to get shapes.

        This accepts either an index, which is the integer index of the desired
        shape in the file, or a bounding box.  

        Parameters
        ----------
        index_or_bounds : int or [xmin, ymin, xmax, ymax]
            Index of the requested shape in filename, or bounding box to filter 
            shapes, or defaults to -1 to get them all.

        crs : crs-type
            Coordinate system of the bounding box (or None if index).

        Returns
        -------
        profile : dict
            Fiona profile of the shapefile.
        shapes : list(dict)
            List of fiona shapes that match the index or bounds.
        
        """
        with fiona.open(self._filename, 'r') as fid:
            profile = fid.profile

            if index_or_bounds is None or type(index_or_bounds) is int:
                if index_or_bounds != None and index_or_bounds >= 0:
                    shps = [fid[index_or_bounds], ]
                else:
                    shps = [s for s in fid]
            else:
                crs_file = profile['crs']
                try:
                    crs_file = watershed_workflow.crs.from_fiona(profile['crs'])
                except watershed_workflow.crs.CRSError:
                    # try to read a damaged file with only wkt
                    crs_file = watershed_workflow.crs.from_wkt(profile['crs_wkt'])
                    profile['crs'] = watershed_workflow.crs.to_fiona(crs_file)

                bounds = watershed_workflow.warp.bounds(index_or_bounds, crs, crs_file)
                shps = [s for (i, s) in fid.items(bbox=bounds)]

        return profile, shps
