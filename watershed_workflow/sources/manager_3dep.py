"""Manager for downloading 3DEP data."""

from typing import Tuple

import shapely.geometry
import xarray as xr
import py3dep

import watershed_workflow.crs

class Manager3DEP:
    name = '3DEP'
    
    def __init__(self, resolution : int):
        """Downloades DEM data from the 3DEP.

        Valid resolutions are: 60, 30, or 10 (m)
        """
        self._resolution = resolution

    def getDataset(self,
                   geometry : shapely.geometry.base.BaseGeometry,
                   geometry_crs : watershed_workflow.crs.CRS,
                   dataset : str = "DEM") -> xr.DataArray:
        """Gets the DEM data for a given geometry."""
        return py3dep.get_map(dataset, geometry, self._resolution, geometry_crs)
        
