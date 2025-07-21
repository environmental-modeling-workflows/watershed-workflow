"""Manager for interacting with NLCD datasets."""
import os, sys
import logging

import xarray as xr
import geopandas as gpd
import shapely.geometry

from typing import Tuple
import pygeohydro

from watershed_workflow.crs import CRS


colors = {
    0: ('None', (1., 1., 1.)),
    11: ('Open Water', (0.27843137255, 0.41960784314, 0.62745098039)),
    12: ('Perrenial Ice/Snow', (0.81960784314, 0.86666666667, 0.97647058824)),
    21: ('Developed, Open Space', (0.86666666667, 0.78823529412, 0.78823529412)),
    22: ('Developed, Low Intensity', (0.84705882353, 0.57647058824, 0.50980392157)),
    23: ('Developed, Medium Intensity', (0.92941176471, 0.00000000000, 0.00000000000)),
    24: ('Developed, High Intensity', (0.66666666667, 0.00000000000, 0.00000000000)),
    31: ('Barren Land', (0.69803921569, 0.67843137255, 0.63921568628)),
    41: ('Deciduous Forest', (0.40784313726, 0.66666666667, 0.38823529412)),
    42: ('Evergreen Forest', (0.10980392157, 0.38823529412, 0.18823529412)),
    43: ('Mixed Forest', (0.70980392157, 0.78823529412, 0.55686274510)),
    51: ('Dwarf Scrub', (0.64705882353, 0.54901960784, 0.18823529412)),
    52: ('Shrub/Scrub', (0.80000000000, 0.72941176471, 0.48627450980)),
    71: ('Grassland/Herbaceous', (0.88627450980, 0.88627450980, 0.75686274510)),
    72: ('Sedge/Herbaceous', (0.78823529412, 0.78823529412, 0.46666666667)),
    73: ('Lichens', (0.60000000000, 0.75686274510, 0.27843137255)),
    74: ('Moss', (0.46666666667, 0.67843137255, 0.57647058824)),
    81: ('Pasture/Hay', (0.85882352941, 0.84705882353, 0.23921568628)),
    82: ('Cultivated Crops', (0.66666666667, 0.43921568628, 0.15686274510)),
    90: ('Woody Wetlands', (0.72941176471, 0.84705882353, 0.91764705882)),
    95: ('Emergent Herbaceous Wetlands', (0.43921568628, 0.63921568628, 0.72941176471)),
    127: ('None', (1., 1., 1.)),
}

indices = dict([(pars[0], id) for (id, pars) in colors.items()])


class ManagerNLCD:
    """National Land Cover Database provides a raster for indexed land cover types
    [NLCD]_.

    .. note:: NLCD does not provide an API for subsetting the data, so the
       first time this is used, it WILL result in a long download time as it
       grabs the big file.  After that it will be much faster as the file is
       already local.

    TODO: Labels and colors for these indices should get moved here, but
    currently reside in watershed_workflow.colors.

    Parameters
    ----------
    layer : str, optional
      Layer of interest.  Default is `"cover`", should also be one for at
      least imperviousness, maybe others?
    year : int, optional
      Year of dataset.  Defaults to the most current available at the location.
    location : str, optional
      Location code.  Default is `"L48`" (lower 48), valid include `"AK`"
      (Alaska), `"HI`" (Hawaii, and `"PR`" (Puerto Rico).

    .. [NLCD] https://www.mrlc.gov/

    """
    colors = colors
    indices = indices

    def __init__(self, layer='cover', year=None, location='L48'):
        self.layer, self.year, self.location = self.validateInput(layer, year, location)
        self.layer_name = 'NLCD_{1}_{0}_{2}'.format(self.layer, self.year, self.location)
        self.name = 'National Land Cover Database (NLCD) Layer: {}'.format(self.layer_name)

    def validateInput(self, layer, year, location):
        """Validates input to the __init__ method."""
        valid_layers = ['cover', 'impervious', 'canopy', 'descriptor']
        if layer not in valid_layers:
            raise ValueError('NLCD invalid layer "{}" requested, valid are: {}'.format(
                layer, valid_layers))

        valid_locations = ['L48', 'AK', 'HI', 'PR']
        if location not in valid_locations:
            raise ValueError('NLCD invalid location "{}" requested, valid are: {}'.format(
                location, valid_locations))

        valid_years = {
            'L48': [2021, 2019, 2016, 2013, 2011, 2008, 2006, 2004, 2001],
            'AK': [2016, 2011, 2001],
            'HI': [2001, ],
            'PR': [2001, ],
        }
        if year is None:
            year = valid_years[location][0]
        else:
            if year not in valid_years[location]:
                raise ValueError(
                    'NLCD invalid year "{}" requested for location {}, valid are: {}'.format(
                        year, location, valid_years[location]))

        return layer, year, location

    def getDataset(self,
                   geometry : shapely.geometry.base.BaseGeometry,
                   geometry_crs : CRS) -> xr.DataArray:
        """
        Retrieves the NLCD dataset for a given geometry.

        Parameters
        ----------
        geometry : shapely.geometry.base.BaseGeometry
            The geometry for which the dataset is to be retrieved. 
        geometry_crs : str, optional
            The coordinate reference system of the geometry. If not provided, it defaults
            to the CRS of the geometry if available, otherwise assumes 'epsg:4326'.

        Returns
        -------
        xarray.DataArray
            The NLCD dataset corresponding to the provided geometry.
        """

        from_tuple = False
        
        geom_df = gpd.GeoDataFrame(geometry=[geometry,], crs=geometry_crs)

        dataset = pygeohydro.nlcd_bygeom(
            geom_df, 
            resolution=30, 
            years={self.layer: self.year},
            region=self.location,
        )

        assert len(dataset) == 1
        dset = dataset[0]
        assert len(dset) == 1
        return dset[next(k for k in dset.keys())]
    


