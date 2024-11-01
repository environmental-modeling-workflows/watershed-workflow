import pytest
import watershed_workflow.crs
import watershed_workflow

from source_fixtures import sources, sources_download


# def test_river_tree_properties(sources_download):
#     crs = watershed_workflow.crs.default_crs
#     nhd = sources_download['hydrography']
#     _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
#     _, reaches = watershed_workflow.get_reaches(nhd,
#                                                 '060102020103',
#                                                 cc.exterior(),
#                                                 crs,
#                                                 crs,
#                                                 properties=True)

#     rivers = watershed_workflow.construct_rivers(reaches, method='hydroseq')
#     assert (len(rivers) == 1)
#     assert (rivers[0].is_consistent())
#     assert (len(rivers[0]) == 94)


# def test_river_tree_properties_prune(sources_download):
#     crs = watershed_workflow.crs.default_crs
#     nhd = sources_download['hydrography']
#     _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
#     _, reaches = watershed_workflow.get_reaches(nhd,
#                                                 '060102020103',
#                                                 cc.exterior(),
#                                                 crs,
#                                                 crs,
#                                                 properties=True)

#     rivers = watershed_workflow.construct_rivers(reaches,
#                                                  method='hydroseq',
#                                                  prune_by_area=0.03 * cc.exterior().area * 1.e-6)
#     assert (len(rivers) == 1)
#     assert (rivers[0].is_consistent())
#     assert (len(rivers[0]) == 49)


# def test_river_tree_geometry(sources):
#     crs = watershed_workflow.crs.default_crs
#     nhd = sources['HUC']
#     _, cc = watershed_workflow.get_split_form_hucs(nhd, '060102020103', 12, crs)
#     _, reaches = watershed_workflow.get_reaches(nhd,
#                                                 '060102020103',
#                                                 cc.exterior(),
#                                                 crs,
#                                                 crs,
#                                                 properties=False)

#     rivers = watershed_workflow.construct_rivers(reaches)
#     assert (len(rivers) == 1)
#     assert (rivers[0].is_consistent())
#     assert (len(rivers[0]) == 98)
