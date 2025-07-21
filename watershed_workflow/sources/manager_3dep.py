"""Manager for downloading 3DEP data."""

from typing import Tuple

import shapely
import xarray
import py3dep

import watershed_workflow.crs

class Manager3DEP:
    name = '3DEP'
    
    def __init__(self, resolution):
        self._resolution = resolution

    def getData(self,
                   geometry : shapely.geometry.Polygon | \
                              shapely.geometry.MultiPolygon | \
                              Tuple[float,float,float,float],
                   geometry_crs : watershed_workflow.crs.CRS,
                   dataset : str = "DEM") -> xarray.DataArray:
        return py3dep.get_map(dataset, geometry, self._resolution, geometry_crs)
        
