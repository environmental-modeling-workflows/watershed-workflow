import watershed_workflow.sources.manager_nhd as nhd
import watershed_workflow.sources.utils as source_utils

class FileManagerNHDPlusAccumulator:
    """NHDPlus is organized by HUC4s, but sometimes we need them on 2s."""
    lowest_level = 2 # we can accumulate to anything!
    def __init__(self):
        self.nhd_plus = nhd.FileManagerNHDPlus()
        self.wbd = nhd.FileManagerWBD()
        self.name = self.nhd_plus.name

    def get_huc(self, huc):
        huc = source_utils.huc_str(huc)
        if len(huc) > 2:
            return self.nhd_plus.get_huc(huc)
        else:
            return self.wbd.get_huc(huc)

    def get_hucs(self, huc, level):
        huc = source_utils.huc_str(huc)
        if len(huc) > 2:
            return self.nhd_plus.get_hucs(huc, level)
        else:
            prof, wbd_hucs = self.wbd.get_hucs(huc, 4)
            contained = []
            for hu in wbd_hucs:
                if 'HUC4' in hu['properties']:
                    prof, subhucs = self.nhd_plus.get_hucs(hu['properties']['HUC4'], level)
                elif 'huc4' in hu['properties']:
                    prof, subhucs = self.nhd_plus.get_hucs(hu['properties']['huc4'], level)
                else:
                    raise RuntimeError(f'Cannot find HUC4 property name in HUC properties for {huc}')
                contained.extend(subhucs)
            return prof, contained

                
