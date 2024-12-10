import pytest
import shapely
import geopandas
import numpy as np
from matplotlib import pyplot as plt

from watershed_workflow.test.shapes import two_boxes
import watershed_workflow.test.shapes
import watershed_workflow.angles

def test_bad_angle():
    ls1 = np.array([ (0,0), (1,0), (1,1) ])
    ls2 = np.array([ (0,0), (1,0), (0,0.01) ])
    ls3 = np.array([ (0,0), (1,0), (0,-0.01) ])

    assert watershed_workflow.angles._isInternalSharpAngle(ls2, 1, 10)
    assert watershed_workflow.angles._isInternalSharpAngle(ls3, 1, 10)
    assert not watershed_workflow.angles._isInternalSharpAngle(ls1, 1, 10)
    assert watershed_workflow.angles._isInternalSharpAngle(ls1, 1, 95)        


    
def testInternalAngleLen3Null():
    """Tests that same ls is returned if no bad angles"""
    # angle at 1 is 90 degrees, nothing to do
    ls = shapely.geometry.LineString([ (0,0), (1,1), (2,1) ])
    assert not watershed_workflow.angles._isInternalSharpAngle(ls, 1, 20)
    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert count == 0
    assert watershed_workflow.utils.isClose(ls, ls2)


def testInternalAngleLen3RemovePoint():
    """Tests length 3 -- center point is removed"""
    # angle at 1 is bad, remove point
    ls = shapely.geometry.LineString([ (0,0), (1,10), (2,0) ])
    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert count == 1

    # endpoints don't move
    assert watershed_workflow.angles._isInternalSharpAngle(ls, 1, 20)
    assert len(ls2.coords) == 2
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])


def testInternalAngleLen3Recenter():
    """Tests length 3 -- center point is moved"""
    # angle at 1 is bad degrees, nothing to do
    ls = shapely.geometry.LineString([ (0,0), (1,1.5), (2,0) ])
    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 90)
    assert count == 1
    
    assert watershed_workflow.angles._isInternalSharpAngle(ls, 1, 90)
    assert len(ls2.coords) == len(ls.coords)

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # middle point moves down
    assert ls2.coords[1][1] < ls.coords[1][1]
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 90)


def testInternalAngleNull():
    """Tests len > 3, all are fine."""
    ls = shapely.geometry.LineString([(0,0), (1,1), (2,0), (3,-1), (4,0)])
    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 45)
    assert count == 0
    assert watershed_workflow.utils.isClose(ls, ls2)    


def testInternalAngleOneBad():
    """Tests len > 3, one bad angle."""
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,10), (3,0), (4,0)])
    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert count == 1
    
    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 2 -- neighboring points
    assert len(ls.coords) == len(ls2.coords) + 2
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 20)


def testInternalAngleEndpointBad():
    """Tests len > 3, one bad angle."""
    ls = shapely.geometry.LineString([(0,0), (1,10), (2,0), (3,0), (4,0)])
    assert watershed_workflow.angles.isInternalSharpAngle(ls, 20)

    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert count == 1

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 1 -- neighboring points
    assert len(ls.coords) == len(ls2.coords) + 1
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 20)


def testInternalAngleEndpointTwoBad():
    """Tests len > 3, two bad angles."""
    ls = shapely.geometry.LineString([(0,0), (1,10), (2,0), (3,10), (4,0)])
    assert watershed_workflow.angles.isInternalSharpAngle(ls, 20)

    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert count == 2

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 1 -- neighboring points
    assert len(ls.coords) == len(ls2.coords) + 1
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 20)


def testTwoSuccessiveBadAngles():
    # really not clear what this should do!
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,10), (3,-10), (4,0), (5,0)])
    assert watershed_workflow.angles.isInternalSharpAngle(ls, 20)

    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    print(list(ls.coords))
    print(list(ls2.coords))
    assert count == 1

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 2, the second point isn't considered
    assert len(ls.coords) == len(ls2.coords) + 2

    # at the end of the day, this is what matters!  But it may be oversmoothing?
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 20)


def testThreeSuccessiveBadAngles():
    # really not clear what this should do!
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,10), (3,-10), (4,10), (5,0), (6,0)])
    assert watershed_workflow.angles.isInternalSharpAngle(ls, 20)

    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    print(list(ls.coords))
    print(list(ls2.coords))
    assert count == 2

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 2, the second point isn't considered
    assert len(ls.coords) == len(ls2.coords) + 3
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 20)


def testL():
    """Deal with a kink in an internal angle."""
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,0), (1, -.2), (0,-1), (0,-2)])
    assert watershed_workflow.angles.isInternalSharpAngle(ls, 20)

    count, ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert count == 1
    print(list(ls.coords))
    print(list(ls2.coords))
    assert not watershed_workflow.angles.isInternalSharpAngle(ls2, 20)


def testJunctionAngles():
    r1 = shapely.geometry.LineString([(3,0), (2,0), (1,0), (0,0)])
    r2 = shapely.geometry.LineString(reversed([(3,0), (4,0), (5,0), (6,0)]))
    r3 = shapely.geometry.LineString(reversed([(3,0), (4,1), (5,2), (6,2)]))
    
    df = geopandas.GeoDataFrame({'index':range(3),
                                 'geometry':[r1,r2,r3]}).set_index('index')
    rivers = watershed_workflow.river_tree.createRivers(df, 'geometry')
    assert len(rivers) == 1
    river = rivers[0]
    assert len(river) == 3

    linestrings = [watershed_workflow.utils.reverseLineString(river.linestring),] \
        + [c.linestring for c in river.children]
    angles = watershed_workflow.angles._getAngles(linestrings)
    assert len(angles) == 3
    assert abs(angles[0] - (90+45)) < 1.e-10
    assert abs(angles[1] - 45) < 1.e-10
    assert abs(angles[2] - 180) < 1.e-10
    assert not watershed_workflow.angles.isUpstreamSharpAngle(None, river, 20)


def testJunctionIsBadAngle():
    r1 = shapely.geometry.LineString([(3,0), (2,0), (1,0), (0,0)])
    r2 = shapely.geometry.LineString(reversed([(3,0), (4,0), (5,0), (6,0)]))
    r3 = shapely.geometry.LineString(reversed([(3,0), (4,0.1), (5,0.5), (6,2)]))

    df = geopandas.GeoDataFrame({'index':range(3),
                                 'geometry':[r1,r2,r3]}).set_index('index')
    rivers = watershed_workflow.river_tree.createRivers(df, 'geometry')
    assert len(rivers) == 1
    river = rivers[0]
    assert len(river) == 3

    assert watershed_workflow.angles.isUpstreamSharpAngle(None, river, 20)
    river_orig = river.deepcopy()
    count = watershed_workflow.angles.smoothUpstreamSharpAngles(None, river, 20)
    assert count == 1
    assert not watershed_workflow.angles.isUpstreamSharpAngle(None, river, 20)
    

    assert watershed_workflow.utils.isClose(river_orig.linestring, river.linestring)
    print(river_orig.children[0].linestring)
    print(river.children[0].children[0].linestring)
    print('-----')
    print(river_orig.children[1].linestring)
    print(river.children[0].children[1].linestring)

    # fig, ax = plt.subplots(1,1)
    # river_orig.plot(color='b', marker='x', ax=ax)
    # river.plot(color='r', marker='x', ax=ax)
    # plt.show()
    # assert False


def testHUCOutletIsBad():
    huc_shp = shapely.geometry.Polygon([ (0,0), (1,0), (1,1), (0,1) ])
    huc_shp_df = geopandas.GeoDataFrame({'geometry':[huc_shp,]})
    hucs = watershed_workflow.split_hucs.SplitHUCs(huc_shp_df)
    hucs_copy = hucs.deepcopy()

    reach_shp = shapely.geometry.LineString([ (0.5,0.2), (0.05,0.2), (0,0.5)])
    reach_shp_df = geopandas.GeoDataFrame({'index':[0,], 'geometry':[reach_shp,]}).set_index('index')
    river = watershed_workflow.river_tree.createRivers(reach_shp_df, 'geometry')[0]
    river_copy = river.deepcopy()

    watershed_workflow.hydrography.snapEndpoints(hucs, river, 0.01)

    assert watershed_workflow.angles.isOutletSharpAngle(hucs, river, 20)
    count = watershed_workflow.angles.smoothOutletSharpAngles(hucs, river, 20)
    assert count == 1
    assert not watershed_workflow.angles.isOutletSharpAngle(hucs, river, 20)

    # river is unchanged
    assert watershed_workflow.utils.isClose(river.linestring, river_copy.linestring)

    # fig, ax = plt.subplots(1,1)
    # river_copy.plot(color='b', marker='x', ax=ax)
    # river.plot(color='c', marker='x', ax=ax)
    # hucs_copy.plot(color='k', marker='x', ax=ax)
    # hucs.plot(color='r', marker='x', ax=ax)
    # plt.show()
    # assert False
    

def testHUCIsBad_NoOp(two_boxes):
    tb = watershed_workflow.split_hucs.SplitHUCs(two_boxes)
    assert not watershed_workflow.angles.isHUCsSharpAngle(tb, 80)
    tb_copy = tb.deepcopy()
    print('---')
    for h,ls in tb.linestrings.items():
        print(ls)
    print('---')
    count = watershed_workflow.angles.smoothHUCsSharpAngles(tb, 80)
    assert count == 0
    assert all(watershed_workflow.utils.isClose(p1, p2) for p1,p2 in zip(tb.polygons(), tb_copy.polygons()))


def testHUCIsBad():
    huc_shp1 = shapely.geometry.Polygon([ (0,0), (1,0), (1,10), (0,1) ])
    huc_shp2 = shapely.geometry.Polygon([ (1,0), (2,0), (2,1), (1,10) ])
    huc_shp_df = geopandas.GeoDataFrame({'geometry':[huc_shp1, huc_shp2]})
    hucs = watershed_workflow.split_hucs.SplitHUCs(huc_shp_df)

    hucs_copy = hucs.deepcopy()
    assert watershed_workflow.angles.isHUCsSharpAngle(hucs, 30)
    count = watershed_workflow.angles.smoothHUCsSharpAngles(hucs, 30)
    assert count == 1
    assert not watershed_workflow.angles.isHUCsSharpAngle(hucs, 30)

    # note, shouldn't move the junction points, just the other one
    # fig, ax = plt.subplots(1,1)
    # hucs_copy.plot(color='k', marker='x', ax=ax)
    # hucs.plot(color='r', marker='x', ax=ax)
    # plt.show()
    # assert False
    


