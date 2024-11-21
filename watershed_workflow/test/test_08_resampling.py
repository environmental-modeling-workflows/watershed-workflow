import watershed_workflow
import watershed_workflow.resampling
import watershed_workflow.river_tree
import watershed_workflow.split_hucs

from watershed_workflow.test.shapes import *

#
# NEED WAY MORE TESTS!  With and without reference river...
#

def test_resampling(watershed_poly, watershed_reaches):
    watershed = watershed_workflow.split_hucs.SplitHUCs(watershed_poly)
    rivers = watershed_workflow.river_tree.createRiverTrees(watershed_reaches)

    strat = watershed_workflow.resampling.createStrategyByDistance([0, 25, 100, 50],
                                                                      shapely.ops.unary_union([r.to_mls() for r in rivers]))
    
    watershed_workflow.resampling.resampleHUCs(watershed, strat)
                                                  

    strat2 = watershed_workflow.resampling.createStrategyFixed(14)
    watershed_workflow.resampling.resampleRivers(rivers, strat2)

    # old resampling algorithm
    #assert (53 == len(watershed.exterior.exterior.coords))
    assert (44 == len(watershed.exterior.exterior.coords))

    assert (16 == len(rivers[0].linestring.coords))
    assert (12 == len(rivers[1].linestring.coords))
