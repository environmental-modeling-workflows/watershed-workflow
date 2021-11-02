import watershed_workflow
import watershed_workflow.sources.manager_nhd
import watershed_workflow.crs

import shapely.geometry

def test_split_real():
    wbd = watershed_workflow.sources.manager_nhd.FileManagerWBD()
    profile,h8s1 = watershed_workflow.get_hucs(wbd, '1407', 8, out_crs=watershed_workflow.crs.default_crs())
    profile,h8s2 = watershed_workflow.get_hucs(wbd, '1408', 8, out_crs=watershed_workflow.crs.default_crs())

    h8s = h8s1 + h8s2
    spl = watershed_workflow.split_hucs.SplitHUCs(h8s)
    assert(type(spl.exterior()) is shapely.geometry.Polygon)
