import workflow
import workflow.sources.manager_nhd
import workflow.crs

import shapely.geometry

def test_split_real():
    wbd = workflow.sources.manager_nhd.FileManagerWBD()
    profile,h8s1 = workflow.get_hucs(wbd, '1407', 8, crs=workflow.crs.default_crs())
    profile,h8s2 = workflow.get_hucs(wbd, '1408', 8, crs=workflow.crs.default_crs())

    h8s = h8s1 + h8s2
    spl = workflow.split_hucs.SplitHUCs(h8s)
    assert(type(spl.exterior()) is shapely.geometry.Polygon)
