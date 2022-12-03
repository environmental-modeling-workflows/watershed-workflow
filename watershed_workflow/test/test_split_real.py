import watershed_workflow
import watershed_workflow.sources.manager_nhd
import watershed_workflow.crs
import shapely.geometry

from source_fixtures import sources


def test_split_real(sources):
    huc = sources['HUC']
    profile, h8s1 = watershed_workflow.get_hucs(huc,
                                                '060101',
                                                8,
                                                out_crs=watershed_workflow.crs.default_crs())
    profile, h8s2 = watershed_workflow.get_hucs(huc,
                                                '060102',
                                                8,
                                                out_crs=watershed_workflow.crs.default_crs())

    h8s = h8s1 + h8s2
    spl = watershed_workflow.split_hucs.SplitHUCs(h8s)
    assert (type(spl.exterior()) is shapely.geometry.Polygon)
    assert (len(spl) == 16)
