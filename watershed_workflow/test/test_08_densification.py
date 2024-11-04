import watershed_workflow
import watershed_workflow.densification
import watershed_workflow.river_tree
import watershed_workflow.split_hucs

from watershed_workflow.test.shapes import *

#
# NEED WAY MORE TESTS!  With and without reference river...
#

def test_densification(watershed_poly, watershed_reaches):
    watershed = watershed_workflow.split_hucs.SplitHUCs(watershed_poly)
    rivers = watershed_workflow.river_tree.createRiverTrees(watershed_reaches)

    watershed_workflow.densification.densifyHUCs(huc=watershed,
                                                 huc_raw=watershed,
                                                 rivers=rivers,
                                                 limit_scales=[0, 25, 100, 50])
    watershed_workflow.densification.densifyRivers(rivers, rivers, limit=14)

    assert (53 == len(watershed.exterior.exterior.coords))
    assert (16 == len(rivers[0].segment.coords))
    assert (12 == len(rivers[1].segment.coords))
