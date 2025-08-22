"""Manager for downloading SoilGrids250m-2017 products."""

from typing import List
import os
import logging
import numpy as np
import shapely
import rasterio
import rasterio.mask
import xarray as xr
import rioxarray
import cftime

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.sources.names
import watershed_workflow.warp
import watershed_workflow.crs
from watershed_workflow.sources.manager_dataset import ManagerDataset
from watershed_workflow.crs import CRS


class ManagerSoilGrids2017(ManagerDataset):
    """SoilGrids 250m (2017) datasets.

    SoilGrids 2017 maintains, to date, the only complete
    characterization of all soil properties needed for a hydrologic
    model.  The resolution is decent, and the accuracy is ok, but most
    importantly it is complete.

    .. [SoilGrids2017] https://www.isric.org/explore/soilgrids/faq-soilgrids-2017

    .. [hengl2014soilgrids] Hengl, Tomislav, et al. "SoilGrids1km—global soil information based on automated mapping." PloS one 9.8 (2014): e105992.

    .. [hengl2017soilgrids] Hengl, Tomislav, et al. "SoilGrids250m: Global gridded soil information based on machine learning." PLoS one 12.2 (2017): e0169748.
    
    See the above link for a complete listing of potential variable
    names; included here are a subset used by this code.  That said,
    any 2017 filename can be used with this source manager.

    Variables available with layer information:
    
    - BLDFIE_layer_1 through BLDFIE_layer_7: Bulk density of fine earth [kg m^-3]
    - CLYPPT_layer_1 through CLYPPT_layer_7: Percent clay [%]
    - SLTPPT_layer_1 through SLTPPT_layer_7: Percent silt [%] 
    - SNDPPT_layer_1 through SNDPPT_layer_7: Percent sand [%]
    - WWP_layer_1 through WWP_layer_7: Soil water capacity % at wilting point [%]
    - BDTICM: Absolute depth to continuous, unfractured bedrock [cm]
    """
    URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/"
    DEPTHS = [0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0]
    BASE_VARIABLES = ['BLDFIE', 'CLYPPT', 'SLTPPT', 'SNDPPT', 'WWP']
    BEDROCK_VARIABLE = 'BDTICM'
    LAYERS = list(range(1, 8))  # 1 through 7

    def __init__(self, variant: str | None = None):
        # Create all valid variable combinations
        valid_variables = [self.BEDROCK_VARIABLE]  # BDTICM has no layers
        for base_var in self.BASE_VARIABLES:
            for layer in self.LAYERS:
                valid_variables.append(f'{base_var}_layer_{layer}')
        
        # Set up names helper
        if variant == 'US':
            name = 'SoilGrids2017_US'
            self.names = watershed_workflow.sources.names.Names(
                name, 'soil_structure', name, '{variable}_M_{soillevel}250m_ll_us.tif')
        else:
            name = 'SoilGrids2017'
            self.names = watershed_workflow.sources.names.Names(
                name, 'soil_structure', name, '{variable}_M_{soillevel}250m_ll.tif')
        
        # Initialize ManagerDataset
        super().__init__(
            name=name,
            source=self.URL,
            native_resolution=250.0 / 111320.0,  # 250m converted to degrees (approx)
            native_crs_in=watershed_workflow.crs.from_epsg(4326),
            native_crs_out=watershed_workflow.crs.from_epsg(4326),
            native_start=None,
            native_end=None,
            valid_variables=valid_variables,
            default_variables=[self.BEDROCK_VARIABLE]
        )

    def _requestDataset(self, request: ManagerDataset.Request) -> ManagerDataset.Request:
        """Download all required files for the request.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            The dataset request
            
        Returns
        -------
        ManagerDataset.Request
            The request with is_ready set to True
        """
        # Download all required files
        for var_name in request.variables:
            base_var, layer = self._parseVariable(var_name)
            self._downloadFile(base_var, layer)
        
        # Set ready since downloads are immediate
        request.is_ready = True
        return request
    
    def _fetchDataset(self, request: ManagerDataset.Request) -> xr.Dataset:
        """Fetch the dataset for the request.
        
        Parameters
        ----------
        request : ManagerDataset.Request
            The dataset request
            
        Returns
        -------
        xr.Dataset
            Dataset containing the requested variables
        """
        data_arrays = {}
        bounds = request.geometry.bounds
        
        for var_name in request.variables:
            base_var, layer = self._parseVariable(var_name)
            
            # Get filename using the names helper
            if layer is None:
                soillevel = ''
            else:
                soillevel = f'sl{layer}_'
            filename = self.names.file_name(variable=base_var, soillevel=soillevel)
            
            # Load raster and clip to bounds
            dataset = rioxarray.open_rasterio(filename, cache=False)
            dataset = dataset.rio.clip_box(*bounds, crs=watershed_workflow.crs.to_rasterio(self.native_crs_out))
            
            # Convert to DataArray (remove band dimension if single band)
            if len(dataset.shape) > 2:
                da = dataset[0, :, :]  # Take first band
            else:
                da = dataset
            
            # Set variable name
            da.name = var_name
            
            # Add units and description as attributes
            if base_var == 'BDTICM':
                da.attrs['units'] = 'cm'
                da.attrs['description'] = 'Absolute depth to continuous, unfractured bedrock'
            elif base_var == 'BLDFIE':
                da.attrs['units'] = 'kg m^-3'
                da.attrs['description'] = 'Bulk density of fine earth'
            elif base_var in ['CLYPPT', 'SLTPPT', 'SNDPPT', 'WWP']:
                da.attrs['units'] = '%'
                if base_var == 'CLYPPT':
                    da.attrs['description'] = 'Percent clay'
                elif base_var == 'SLTPPT':
                    da.attrs['description'] = 'Percent silt'
                elif base_var == 'SNDPPT':
                    da.attrs['description'] = 'Percent sand'
                elif base_var == 'WWP':
                    da.attrs['description'] = 'Soil water capacity % at wilting point'
            
            if layer is not None:
                da.attrs['layer'] = layer
            
            data_arrays[var_name] = da
        
        # Create Dataset
        result_dataset = xr.Dataset(data_arrays)
        
        return result_dataset

    def _parseVariable(self, var_name: str) -> tuple[str, int | None]:
        """Parse a variable name to extract base variable and layer.
        
        Parameters
        ----------
        var_name : str
            Variable name (e.g., 'BLDFIE_layer_3' or 'BDTICM')
            
        Returns
        -------
        tuple[str, int | None]
            Base variable name and layer number (None for BDTICM)
        """
        if var_name == self.BEDROCK_VARIABLE:
            return var_name, None
        
        if '_layer_' in var_name:
            parts = var_name.split('_layer_')
            if len(parts) == 2 and parts[0] in self.BASE_VARIABLES:
                try:
                    layer = int(parts[1])
                    if layer in self.LAYERS:
                        return parts[0], layer
                except ValueError:
                    pass
        
        raise ValueError(f"Invalid variable name: {var_name}")

    def _downloadFile(self, base_variable: str, layer: int | None, force: bool = False) -> str:
        """Download a file if it doesn't exist.
        
        Parameters
        ----------
        base_variable : str
            Base variable name
        layer : int | None
            Layer number (None for bedrock)
        force : bool
            Force re-download if True
            
        Returns
        -------
        str
            Path to the downloaded file
        """
        os.makedirs(self.names.folder_name(), exist_ok=True)
        
        if layer is None:
            soillevel = ''
        else:
            soillevel = f'sl{layer}_'
        
        filename = self.names.file_name(variable=base_variable, soillevel=soillevel)
        
        if not os.path.exists(filename) or force:
            filename_base = self.names.file_name_base(variable=base_variable, soillevel=soillevel)
            url = self.URL + filename_base
            source_utils.download(url, filename, force)
        
        return filename

