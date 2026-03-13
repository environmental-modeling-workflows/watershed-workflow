"""Manager for downloading SoilGrids250m-2017 products."""

import os
import logging
import xarray as xr
import rioxarray

import watershed_workflow.crs
import watershed_workflow.utils.config

from . import utils as source_utils
from . import manager_dataset


#: Layered variables — each has 7 depth layers (sl1–sl7).
#: Requesting the base name (e.g. ``'CRFVOL'``) expands to all 7 layers.
LAYERED_VARIABLES = {
    'AWCh1':  'Available water capacity at pF 2.0 [cm3 cm-3]',
    'AWCh2':  'Available water capacity at pF 2.3 [cm3 cm-3]',
    'AWCh3':  'Available water capacity at pF 2.5 [cm3 cm-3]',
    'AWCtS':  'Available water capacity at saturation [cm3 cm-3]',
    'BLDFIE': 'Bulk density of fine earth [kg m-3]',
    'CECSOL': 'Cation exchange capacity of soil [cmolc kg-1]',
    'CLYPPT': 'Clay content [% weight]',
    'CRFVOL': 'Coarse fragments volumetric [% volume]',
    'OCDENS': 'Organic carbon density [kg m-3]',
    'ORCDRC': 'Organic carbon content [g kg-1]',
    'PHIHOX': 'pH measured in water [pH]',
    'PHIKCL': 'pH measured in KCl [pH]',
    'SLTPPT': 'Silt content [% weight]',
    'SNDPPT': 'Sand content [% weight]',
    'TEXMHT': 'Texture class (USDA system) [-]',
    'WWP':    'Volumetric water content at wilting point [% volume]',
}

#: Single-layer (whole-profile) variables — no depth suffix.
SINGLE_VARIABLES = {
    'ACDWRB': 'Acid subsoil gravel content (WRB) [%]',
    'BDRICM': 'Depth to bedrock (R horizon) [m]',
    'BDRLOG': 'Probability of bedrock within 2 m [%]',
    'BDTICM': 'Absolute depth to bedrock [m]',
    'HISTPR': 'Probability of being a Histosol [%]',
    'SLGWRB': 'Sodicity (WRB) [-]',
}

#: Variables stored in cm in the raw files that should be converted to m.
_CM_TO_M_VARIABLES = {'BDRICM', 'BDTICM'}


class ManagerSoilGrids2017(manager_dataset.ManagerDataset):
    """SoilGrids 250m (2017) datasets.

    SoilGrids 2017 [SoilGrids2017]_ maintains, to date, the only complete
    characterization of all soil properties needed for a hydrologic
    model.  The resolution is decent, and the accuracy is ok, but most
    importantly it is complete [hengl2014soilgrids]_ [hengl2017soilgrids]_.

    .. [SoilGrids2017] https://www.isric.org/explore/soilgrids/faq-soilgrids-2017

    .. [hengl2014soilgrids] Hengl, Tomislav, et al. "SoilGrids1km—global soil
       information based on automated mapping." PloS one 9.8 (2014): e105992.

    .. [hengl2017soilgrids] Hengl, Tomislav, et al. "SoilGrids250m: Global
       gridded soil information based on machine learning." PLoS one 12.2
       (2017): e0169748.

    Variables may be requested by base name or by explicit layer name.
    Requesting a layered base name (e.g. ``'CRFVOL'``) expands to all 7 depth
    layers (``CRFVOL_layer_1`` … ``CRFVOL_layer_7``).  Single-layer variables
    are returned as-is.

    Layered variables are returned as a 3D ``(depth_to_horizon, y, x)`` DataArray
    where ``depth_to_horizon`` is the depth to the **top** of each soil horizon
    in metres:

    ====  =======  ===============
    Layer  sl code  depth_to_horizon
    ====  =======  ===============
    1      sl1        0.00 m
    2      sl2        0.05 m
    3      sl3        0.15 m
    4      sl4        0.30 m
    5      sl5        0.60 m
    6      sl6        1.00 m
    7      sl7        2.00 m  (aggregated 0–200 cm)
    ====  =======  ===============

    **Available variables**

    Layered variables (request base name or ``{name}_layer_{1..7}``):

    ========  ===  =====================================================
    Name      Lyr  Description
    ========  ===  =====================================================
    AWCh1     yes  Available water capacity at pF 2.0 [cm3 cm-3]
    AWCh2     yes  Available water capacity at pF 2.3 [cm3 cm-3]
    AWCh3     yes  Available water capacity at pF 2.5 [cm3 cm-3]
    AWCtS     yes  Available water capacity at saturation [cm3 cm-3]
    BLDFIE    yes  Bulk density of fine earth [kg m-3]
    CECSOL    yes  Cation exchange capacity [cmolc kg-1]
    CLYPPT    yes  Clay content [% weight]
    CRFVOL    yes  Coarse fragments volumetric [% volume]
    OCDENS    yes  Organic carbon density [kg m-3]
    ORCDRC    yes  Organic carbon content [g kg-1]
    PHIHOX    yes  pH in water [pH]
    PHIKCL    yes  pH in KCl [pH]
    SLTPPT    yes  Silt content [% weight]
    SNDPPT    yes  Sand content [% weight]
    TEXMHT    yes  Texture class (USDA) [-]
    WWP       yes  Volumetric water content at wilting point [% volume]
    ========  ===  =====================================================

    Single-layer variables (whole-profile):

    ========  ===  =====================================================
    Name      Lyr  Description
    ========  ===  =====================================================
    ACDWRB    no   Acid subsoil gravel content (WRB) [%]
    BDRICM    no   Depth to bedrock (R horizon) [m]
    BDRLOG    no   Probability of bedrock within 2 m [%]
    BDTICM    no   Absolute depth to bedrock [m]
    HISTPR    no   Probability of being a Histosol [%]
    SLGWRB    no   Sodicity (WRB) [-]
    ========  ===  =====================================================
    """

    URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/"
    #: Depth to the top of each soil horizon [m], corresponding to sl1–sl7.
    DEPTHS = [0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0]
    LAYERS = list(range(1, 8))  # sl1 through sl7

    def __init__(self, variant: str | None = None):
        # Build full valid_variables list: all explicit layer names + all single names
        valid_variables = []
        for base_var in LAYERED_VARIABLES:
            for layer in self.LAYERS:
                valid_variables.append(f'{base_var}_layer_{layer}')
        valid_variables.extend(SINGLE_VARIABLES.keys())

        if variant == 'US':
            self._name = 'SoilGrids2017_US'
            self._file_suffix = '250m_ll_us.tif'
        else:
            self._name = 'SoilGrids2017'
            self._file_suffix = '250m_ll.tif'

        super().__init__(
            name=self._name,
            source=self.URL,
            native_resolution=250.0 / 111320.0,
            native_crs_in=watershed_workflow.crs.from_epsg(4326),
            native_crs_out=watershed_workflow.crs.from_epsg(4326),
            native_start=None,
            native_end=None,
            valid_variables=valid_variables,
            default_variables=['BDTICM'],
        )

    def getDataset(self, geometry, geometry_crs, start=None, end=None,
                   variables=None, out_crs=None, temporal_resampling=None):
        """Get SoilGrids2017 data, expanding layered base names to all 7 layers.

        Parameters
        ----------
        geometry : shapely.geometry
            Region of interest.
        geometry_crs : CRS
            CRS of the geometry.
        start : ignored
        end : ignored
        variables : list of str, optional
            Variable names to request.  Layered base names (e.g. ``'CRFVOL'``)
            are expanded to all 7 layers.  Default is ``['BDTICM']``.
        out_crs : CRS, optional
        temporal_resampling : ignored

        Returns
        -------
        xr.Dataset
        """
        variables = self._expandVariables(variables)
        return super().getDataset(geometry, geometry_crs, start, end,
                                  variables, out_crs, temporal_resampling)

    def requestDataset(self, geometry, geometry_crs, start=None, end=None,
                       variables=None, out_crs=None, temporal_resampling=None):
        """Request SoilGrids2017 data, expanding layered base names.

        Parameters
        ----------
        geometry : shapely.geometry
        geometry_crs : CRS
        start : ignored
        end : ignored
        variables : list of str, optional
            Layered base names are expanded to all 7 layers.
        out_crs : CRS, optional
        temporal_resampling : ignored

        Returns
        -------
        ManagerDataset.Request
        """
        variables = self._expandVariables(variables)
        return super().requestDataset(geometry, geometry_crs, start, end,
                                      variables, out_crs, temporal_resampling)

    def _expandVariables(self, variables):
        """Expand layered base names to explicit layer names."""
        if variables is None:
            return None
        expanded = []
        for var in variables:
            if var in LAYERED_VARIABLES:
                expanded.extend(f'{var}_layer_{l}' for l in self.LAYERS)
            else:
                expanded.append(var)
        return expanded

    def _requestDataset(self, request: manager_dataset.ManagerDataset.Request
                        ) -> manager_dataset.ManagerDataset.Request:
        """Return the request unchanged — no async step."""
        return request

    def _isServerReady(self, request: manager_dataset.ManagerDataset.Request) -> bool:
        """Return True — SoilGrids2017 downloads are synchronous."""
        return True

    def _folder(self):
        data_dir = watershed_workflow.utils.config.rcParams['DEFAULT']['data_directory']
        return os.path.join(data_dir, 'soil_structure', self._name)

    def _filename(self, base_var, layer):
        soillevel = '' if layer is None else f'sl{layer}_'
        fname = f'{base_var}_M_{soillevel}{self._file_suffix}'
        return os.path.join(self._folder(), fname)

    def _downloadDataset(self, request: manager_dataset.ManagerDataset.Request) -> None:
        """Download all required server-assigned files if not present."""
        os.makedirs(self._folder(), exist_ok=True)
        for var_name in request.variables:
            base_var, layer = self._parseVariable(var_name)
            filename = self._filename(base_var, layer)
            if not os.path.exists(filename):
                url = self.URL + os.path.basename(filename)
                source_utils.download(url, filename, False)

    def _loadDataset(self, request: manager_dataset.ManagerDataset.Request) -> xr.Dataset:
        """Load variables from server-assigned fixed-path files.

        Layered variables are stacked into a 3D ``(depth, y, x)`` DataArray
        using the ``DEPTHS`` coordinate.  Single-layer variables are 2D ``(y, x)``.

        Parameters
        ----------
        request : ManagerDataset.Request
            The dataset request.

        Returns
        -------
        xr.Dataset
            Dataset containing the requested variables.
        """
        # Clip bounds in the native CRS for early spatial subsetting
        bounds = request.geometry.bounds  # already in native_crs_in (WGS84)

        # Collect 2D arrays grouped by base variable
        layers_by_base = {}  # base_var -> {layer: DataArray}
        singles = {}         # var_name -> DataArray

        for var_name in request.variables:
            base_var, layer = self._parseVariable(var_name)
            filename = self._filename(base_var, layer)

            raw = rioxarray.open_rasterio(filename, cache=False)
            raw = raw.rio.clip_box(*bounds)  # subset to request bounds before loading
            da = raw[0, :, :] if len(raw.shape) > 2 else raw

            if layer is not None:
                layers_by_base.setdefault(base_var, {})[layer] = da
            else:
                da.name = var_name
                da.attrs['description'] = SINGLE_VARIABLES.get(base_var, '')
                singles[var_name] = da

        data_arrays = dict(singles)

        # Stack layered variables along a depth-to-horizon coordinate
        for base_var, layer_dict in layers_by_base.items():
            sorted_layers = sorted(layer_dict.keys())
            depths = [self.DEPTHS[l - 1] for l in sorted_layers]
            depth_coord = xr.DataArray(
                depths,
                dims='depth_to_horizon',
                name='depth_to_horizon',
                attrs={'units': 'm', 'description': 'Depth to the top of each soil horizon'},
            )
            stacked = xr.concat([layer_dict[l] for l in sorted_layers], dim=depth_coord)
            stacked.name = base_var
            data_arrays[base_var] = stacked

        return xr.Dataset(data_arrays)

    def _postprocessDataset(self, request, dataset):
        """Apply nodata masking and float conversion, then clip."""
        # Mask nodata BEFORE the base-class clip so that reprojection interpolates
        # NaN (not -32768) into border pixels.
        for var_name in list(dataset.data_vars):
            da = dataset[var_name].astype('float32')
            da = da.where(da != -32768)  # SoilGrids2017 nodata value → NaN
            da.attrs.pop('_FillValue', None)
            if var_name in _CM_TO_M_VARIABLES:
                da = da / 100.0  # cm → m
            desc = LAYERED_VARIABLES.get(var_name) or SINGLE_VARIABLES.get(var_name, '')
            da.attrs['description'] = desc
            dataset[var_name] = da
        return super()._postprocessDataset(request, dataset)

    def _parseVariable(self, var_name: str) -> tuple[str, int | None]:
        """Parse ``var_name`` into ``(base_var, layer)``."""
        if var_name in SINGLE_VARIABLES:
            return var_name, None
        if '_layer_' in var_name:
            base, _, num = var_name.rpartition('_layer_')
            if base in LAYERED_VARIABLES:
                try:
                    layer = int(num)
                    if layer in self.LAYERS:
                        return base, layer
                except ValueError:
                    pass
        raise ValueError(f'Invalid SoilGrids2017 variable: {var_name!r}')
