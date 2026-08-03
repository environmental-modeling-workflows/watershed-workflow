"""Manager for the Shangguan et al. (2017) global depth-to-bedrock dataset."""
import os
import xarray as xr
import numpy as np

import watershed_workflow.crs

from . import manager_raster
from . import cache_info
from .manager import ManagerAttributes


_URL = 'http://globalchange.bnu.edu.cn/research/dtbd.jsp'
_FILENAME = 'BDTICM_M_250m_ll.tif'


class ManagerShangguanDTB(manager_raster.ManagerRaster):
    """Global 250 m depth-to-bedrock from Shangguan et al. (2017) [ShangguanDTB]_.

    .. note:: This dataset has no download API.  Download ``BDTICM_M_250m_ll.zip``
       from the URL below, unzip it, and place the resulting TIF at::

           <data_directory>/soil_structure/Shangguan_DTB/bnu_globalchange_https/BDTICM_M_250m_ll.tif

    The returned ``band_1`` variable is depth to bedrock in **metres**.

    .. [ShangguanDTB] Shangguan, W., T. Hengl, J. Mendes de Jesus, H. Yuan,
       and Y. Dai. 2017. Mapping the global depth to bedrock for land surface
       modeling. Journal of Advances in Modeling Earth Systems, 9, 65-88.
       https://doi.org/10.1002/2016MS000686

       Data available at: http://globalchange.bnu.edu.cn/research/dtbd.jsp
    """

    def __init__(self):
        attrs = ManagerAttributes(
            category='soil_structure',
            product='Shangguan Depth-to-Bedrock',
            product_short='Shangguan_DTB',
            source='BNU GlobalChange',
            source_short='bnu_globalchange_https',
            url='https://doi.org/10.1002/2016MS000686',
            license=None,
            citation='Shangguan et al. 2017',
            description='Global 250 m depth to bedrock from Shangguan et al. (2017).',
        )
        filename = cache_info.localFilePath(attrs, _FILENAME)
        super().__init__(
            filename,
            native_crs=watershed_workflow.crs.from_epsg(4326),
            native_resolution=0.002083333,  # ~250 m in degrees
            bands=['band_1'],
            attrs=attrs,
        )

    def _downloadDataset(self, request: manager_raster.ManagerRaster.Request) -> None:
        """Validate that the file exists; raise a helpful error if not."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f'ShangguanDTB file not found: {self.filename}\n'
                f'Download BDTICM_M_250m_ll.zip from {_URL}, unzip, and place '
                f'the TIF at <data_directory>/soil_structure/Shangguan_DTB/bnu_globalchange_https/{_FILENAME}'
            )

    def _postprocessDataset(self, request, dataset):
        """Clip first, then convert dtype/nodata/units on the small clipped result."""
        dataset = super()._postprocessDataset(request, dataset)
        da = dataset['band_1'].astype('float32')
        da = da.rio.write_nodata(-99999, encoded=True, inplace=True)
        da = da.where(da != -99999)  # -99999 -> NaN
        da = da / 100.0              # cm -> m
        da.attrs['units'] = 'm'
        da.attrs['description'] = 'Depth to bedrock'
        dataset['band_1'] = da
        return dataset
