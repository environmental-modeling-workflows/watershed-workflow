import os, sys
import logging

import pynhd
import geopandas

import watershed_workflow.crs
import watershed_workflow.sources.utils as src_utils

class FileManagerPyNHD:
    """Manager for interacting with USGS NLDI datasets.
    """
    def __init__(self):
        import pynhd
        self._crs = watershed_workflow.crs.from_epsg('4326')
        self.nhd = pynhd.WaterData("nhdflowline_network")

    def get_hucs(self, hucstr, level=None):
        if level is None:
            level = len(hucstr)
        assert(level == len(hucstr))
        logging.info(f'NLDI: Downloading huc {hucstr}')
        if len(hucstr) > 9:
            wbdstr = f'wbd{len(hucstr)}'
        else:
            wbdstr = f'wbd0{len(hucstr)}'

        wd = pynhd.WaterData(wbdstr, crs=self._crs)
        huc_df = wd.byid(f'huc{len(hucstr)}', hucstr)
        logging.info(f' ... done')
        return huc_df

    def get_hydro(self,
                  huc,
                  bounds=None,
                  bounds_crs=None,
                  in_network=True,
                  properties=None,
                  include_catchments=False,
                  force_download=False):
        if bounds is None:
            huc_shp = self.get_huc(huc)
            bounds = huc_shp.iloc[0].geometry.bounds
        else:
            bounds = watershed_workflow.warp.bounds(
                bounds, bounds_crs, self._crs)
        
        # get the reaches
        logging.info(f'NLDI: Downloading reaches in {bounds}')
        reaches = self.nhd.bybox(bounds)
        logging.info(f' ... done')
        return reaches
