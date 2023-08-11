import os, sys
import geopandas
import logging
import pynhd

class FileManagerPyNHD:
    """Manager for interacting with USGS NLDI datasets.
    """
    def __init__(self):
        import pynhd

        self.name = 'NLDI'
        self.file_level = 12
        self.lowest_level = 12
        self._crs = '5070'
        self.nldi = pynhd.NLDI()

    def get_huc12_upstream(self, huc, distance=500):
        # get all reaches upstream of the pourpoint within one perimeter of the huc
        pp = self.nldi.getfeature_byid('huc12pp', hucstr, 'upstreamTributary', 'huc12pp',
                                         distance=distance)
        return geopandas.concat([self.get_huc(h) for h in list(pp.name)])
        
    def get_huc(self, hucstr):

        if len(hucstr) > 9:
            wbdstr = f'wbd{len(hucstr)}'
        else:
            wbdstr = f'wbd0{len(hucstr)}'
        wd = pynhd.WaterData(wbdstr, crs=self._crs)

        huc_df = wd.byid(f'huc'+len(hucstr), hucstr)
        return huc_df

    def get_hydro_upstream(self, hucstr, distance=500):
        # get all reaches upstream of the pourpoint within one perimeter of the huc
        reaches = self.nldi.getfeature_byid('huc12pp', hucstr, 'upstreamTributary', 'flowlines',
                                         distance=distance).to_crs(self._crs)
        reaches['IDs'] = [reaches.loc[index].comid for index in reaches.index]
        return reaches


    def get_hydro(self, hucstr):
        # get the huc shape
        huc = self._get_huc(hucstr)
        assert(len(huc) == 1)
        huc = huc.loc[0]

        # get the reaches
        reaches = self._get_hydro_upstream(hucstr, huc.geometry.length * 1e-3)

        # intersect
        return reaches[reaches.intersects(huc.geometry)]
        

                                            
        


        
        
        
