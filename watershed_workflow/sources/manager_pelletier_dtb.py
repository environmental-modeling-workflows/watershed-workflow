"""Manager for the Pelletier et al. (2016) global depth-to-bedrock dataset."""
import os
import numpy as np
import xarray as xr

import watershed_workflow.crs

from . import manager_raster
from . import cache_info
from .manager import ManagerAttributes


_URL = 'https://daac.ornl.gov/SOILS/guides/Global_Soil_Regolith_Sediment.html'

_FILENAME = os.path.join(
    'Global_Soil_Regolith_Sediment_1304', 'data',
    'average_soil_and_sedimentary-deposit_thickness.tif',
)


class ManagerPelletierDTB(manager_raster.ManagerRaster):
    """Global 1 km depth-to-bedrock from Pelletier et al. (2016) [PelletierDTB]_.

    .. note:: This dataset has no download API and is a large (~1 GB) file.
       Download it from the DOI below and unzip into::

           <data_directory>/soil_structure/Pelletier_DTB/ornl_daac_https/

       which should yield::

           <data_directory>/soil_structure/Pelletier_DTB/ornl_daac_https/Global_Soil_Regolith_Sediment_1304/data/average_soil_and_sedimentary-deposit_thickness.tif

    The returned ``band_1`` variable is depth to bedrock in **metres**.

    .. [PelletierDTB] Pelletier, J.D., et al. 2016. Global 1-km Gridded
       Thickness of Soil, Regolith, and Sedimentary Deposit Layers. ORNL
       DAAC, Oak Ridge, Tennessee, USA.
       http://dx.doi.org/10.3334/ORNLDAAC/1304
    """

    def __init__(self):
        attrs = ManagerAttributes(
            category='soil_structure',
            product='Pelletier Depth-to-Bedrock',
            product_short='Pelletier_DTB',
            source='ORNL DAAC',
            source_short='ornl_daac_https',
            url='https://doi.org/10.3334/ORNLDAAC/1304',
            license='public domain',
            citation='Pelletier et al. 2016',
            description='Global 1-km depth to bedrock from Pelletier et al. (2016).',
        )
        filename = cache_info.localFilePath(attrs, _FILENAME)
        super().__init__(
            filename,
            native_crs=watershed_workflow.crs.from_epsg(4326),
            native_resolution=1000.0 / 111320.0,  # 1 km in degrees
            bands=['band_1'],
            attrs=attrs,
        )

    def _downloadDataset(self, request: manager_raster.ManagerRaster.Request) -> None:
        """Validate that the file exists; raise a helpful error if not."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f'PelletierDTB file not found: {self.filename}\n'
                f'Download from {_URL} and unzip into '
                f'<data_directory>/soil_structure/Pelletier_DTB/ornl_daac_https/'
            )

    def _postprocessDataset(self, request, dataset):
        """Clip first, then convert dtype/nodata on the small clipped result."""
        dataset = super()._postprocessDataset(request, dataset)
        da = dataset['band_1'].astype('float32')
        da = da.rio.write_nodata(-1, encoded=True, inplace=True)
        da = da.where(da != -1)  # -1 -> NaN; raw values are integer metres
        da.attrs['units'] = 'm'
        da.attrs['description'] = 'Depth to bedrock'
        dataset['band_1'] = da
        return dataset
