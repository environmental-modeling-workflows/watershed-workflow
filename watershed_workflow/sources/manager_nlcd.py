"""Manager for interacting with NLCD datasets."""
import os, sys
import logging

import xarray as xr
import rioxarray # needed to get rio, even though not used.
import geopandas as gpd
import shapely.geometry
import cftime

from typing import Tuple, List, Optional
import pygeohydro
import pygeohydro.helpers

from watershed_workflow.crs import CRS
from watershed_workflow.sources.manager_dataset import ManagerDataset


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


class ManagerNLCD(ManagerDataset):
    """National Land Cover Database manager for single-year snapshots.

    Supports variables: cover, impervious, canopy, descriptor.
    Each manager instance represents a single year of NLCD data.

    Parameters
    ----------
    location : str, optional
        Location code ('L48', 'AK', 'HI', 'PR'). Default 'L48'.
    year : int, optional
        NLCD data year. If None, uses most recent available for location.

    .. [NLCD] https://www.mrlc.gov/

    """
    colors = colors
    indices = indices

    def __init__(self, location='L48', year=None):
        """Initialize NLCD manager for specific location and year.
        
        Parameters
        ----------
        location : str, optional
            Location code ('L48', 'AK', 'HI', 'PR'). Default 'L48'.
        year : int, optional
            NLCD data year. If None, uses most recent available for location.
        """
        self.location = self._validateLocation(location)
        self.year = self._validateYear(year, location)
        
        # NLCD is non-temporal - each instance represents one year snapshot
        native_crs = CRS.from_epsg(4326)  # WGS84 Geographic
        super().__init__(
            name=f'NLCD {self.year} {self.location}',
            source='pygeohydro',
            native_resolution=0.00027,  # ~30m in degrees (approximately 30m at mid-latitudes)
            native_crs_in=native_crs,    # Expected input CRS
            native_crs_out=native_crs,   # Output data CRS
            native_start=None,           # Non-temporal data
            native_end=None,             # Non-temporal data
            valid_variables=['cover', 'impervious', 'canopy', 'descriptor'],
            default_variables=['cover']
        )

    def _validateLocation(self, location):
        """Validate location parameter."""
        valid_locations = ['L48', 'AK', 'HI', 'PR']
        if location not in valid_locations:
            raise ValueError(f'NLCD invalid location "{location}", valid are: {valid_locations}')
        return location

    def _validateYear(self, year, location):
        """Validate year for given location."""
        valid_years = {
            'L48': [2021, 2019, 2016, 2013, 2011, 2008, 2006, 2004, 2001],
            'AK': [2016, 2011, 2001],
            'HI': [2001],
            'PR': [2001]
        }
        
        if year is None:
            return valid_years[location][0]  # Most recent
        
        if year not in valid_years[location]:
            raise ValueError(f'NLCD invalid year "{year}" for location {location}, '
                            f'valid are: {valid_years[location]}')
        return year

    def _requestDataset(self, request: ManagerDataset.Request) -> ManagerDataset.Request:
        """Request NLCD data - ready immediately.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request with preprocessed parameters.
            
        Returns
        -------
        ManagerDataset.Request
            Updated request marked as ready.
        """
        request.is_ready = True
        return request

    def _fetchDataset(self, request: ManagerDataset.Request) -> xr.Dataset:
        """Fetch NLCD data for the request.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            Dataset request with preprocessed parameters.
            
        Returns
        -------
        xr.Dataset
            Dataset with requested NLCD variables for the specified year.
        """
        # Extract parameters from request
        geometry = request.geometry
        variables = request.variables
        
        assert variables is not None, "Variables should not be None for multi-variable NLCD data"
        
        # Create GeoDataFrame with native CRS (geometry is already in native_crs_in)
        geom_df = gpd.GeoDataFrame(geometry=[geometry], crs=self.native_crs_in)
        
        # Build years dict for pygeohydro - single year for all variables
        years_dict = {var: [self.year] for var in variables}
        
        # Fetch data using pygeohydro
        data_dict = pygeohydro.nlcd_bygeom(
            geom_df,
            resolution=30,  # Use 30 meters (pygeohydro expects meters)
            years=years_dict,
            region=self.location,
        )
        
        # Extract the dataset (dict key is GeoDataFrame index, we have index 0)
        raw_dataset = data_dict[0]
        
        # Create final dataset with variable names as keys (not prefixed)
        final_dataset = xr.Dataset()
        for var in variables:
            # pygeohydro returns variables as 'var_year'
            source_key = f'{var}_{self.year}'
            if source_key in raw_dataset:
                final_dataset[var] = raw_dataset[source_key]
            else:
                raise ValueError(f"Variable {var} for year {self.year} not found in pygeohydro response")
        
        # Add metadata attributes
        final_dataset.attrs['nlcd_year'] = self.year
        final_dataset.attrs['nlcd_location'] = self.location
        
        return final_dataset
    


