import pytest
import shapely
import geopandas
import numpy as np

import watershed_workflow.test.shapes
import watershed_workflow.angles

def test_bad_angle():
    ls1 = np.array([ (0,0), (1,0), (1,1) ])
    ls2 = np.array([ (0,0), (1,0), (0,0.01) ])
    ls3 = np.array([ (0,0), (1,0), (0,-0.01) ])

    assert watershed_workflow.angles._badInternalAngle(ls2, 1, 10)
    assert watershed_workflow.angles._badInternalAngle(ls3, 1, 10)
    assert not watershed_workflow.angles._badInternalAngle(ls1, 1, 10)
    assert watershed_workflow.angles._badInternalAngle(ls1, 1, 95)        


    
def testInternalAngleLen3Null():
    """Tests that same ls is returned if no bad angles"""
    # angle at 1 is 90 degrees, nothing to do
    ls = shapely.geometry.LineString([ (0,0), (1,1), (2,1) ])
    assert not watershed_workflow.angles._badInternalAngle(ls, 1, 20)
    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    assert watershed_workflow.utils.isClose(ls, ls2)


def testInternalAngleLen3RemovePoint():
    """Tests length 3 -- center point is removed"""
    # angle at 1 is bad, remove point
    ls = shapely.geometry.LineString([ (0,0), (1,10), (2,0) ])
    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)

    # endpoints don't move
    assert watershed_workflow.angles._badInternalAngle(ls, 1, 20)
    assert len(ls2.coords) == 2
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])


def testInternalAngleLen3Recenter():
    """Tests length 3 -- center point is moved"""
    # angle at 1 is bad degrees, nothing to do
    ls = shapely.geometry.LineString([ (0,0), (1,1.5), (2,0) ])
    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 90)

    assert watershed_workflow.angles._badInternalAngle(ls, 1, 90)
    assert len(ls2.coords) == len(ls.coords)

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # middle point moves down
    assert ls2.coords[1][1] < ls.coords[1][1]
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 90)


def testInternalAngleNull():
    """Tests len > 3, all are fine."""
    ls = shapely.geometry.LineString([(0,0), (1,1), (2,0), (3,-1), (4,0)])
    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 45)
    assert watershed_workflow.utils.isClose(ls, ls2)    


def testInternalAngleOneBad():
    """Tests len > 3, one bad angle."""
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,10), (3,0), (4,0)])
    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 2 -- neighboring points
    assert len(ls.coords) == len(ls2.coords) + 2
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 20)


def testInternalAngleEndpointBad():
    """Tests len > 3, one bad angle."""
    ls = shapely.geometry.LineString([(0,0), (1,10), (2,0), (3,0), (4,0)])
    assert watershed_workflow.angles.isBadInternalAngle(ls, 20)

    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 1 -- neighboring points
    assert len(ls.coords) == len(ls2.coords) + 1
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 20)


def testInternalAngleEndpointTwoBad():
    """Tests len > 3, one bad angle."""
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,0), (3,10), (4,0)])
    assert watershed_workflow.angles.isBadInternalAngle(ls, 20)

    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 1 -- neighboring points
    assert len(ls.coords) == len(ls2.coords) + 1
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 20)


def testTwoSuccessiveBadAngles():
    # really not clear what this should do!
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,10), (3,-10), (4,0), (5,0)])
    assert watershed_workflow.angles.isBadInternalAngle(ls, 20)

    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    print(list(ls.coords))
    print(list(ls2.coords))

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 2, the second point isn't considered
    assert len(ls.coords) == len(ls2.coords) + 2

    # at the end of the day, this is what matters!  But it may be oversmoothing?
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 20)


def testThreeSuccessiveBadAngles():
    # really not clear what this should do!
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,10), (3,-10), (4,10), (5,0), (6,0)])
    assert watershed_workflow.angles.isBadInternalAngle(ls, 20)

    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    print(list(ls.coords))
    print(list(ls2.coords))

    # endpoints don't move
    assert watershed_workflow.utils.isClose(ls.coords[0], ls2.coords[0])
    assert watershed_workflow.utils.isClose(ls.coords[-1], ls2.coords[-1])

    # length reduced by 2, the second point isn't considered
    assert len(ls.coords) == len(ls2.coords) + 3
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 20)


def testL():
    """Deal with a kink in an internal angle."""
    ls = shapely.geometry.LineString([(0,0), (1,0), (2,0), (1, -.2), (0,-1), (0,-2)])
    assert watershed_workflow.angles.isBadInternalAngle(ls, 20)

    ls2 = watershed_workflow.angles._smoothInternalSharpAngles(ls, 20)
    print(list(ls.coords))
    print(list(ls2.coords))
    assert not watershed_workflow.angles.isBadInternalAngle(ls2, 20)


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

    angles = watershed_workflow.angles.getAnglesAtJunction(river)
    assert len(angles) == 3
    assert abs(angles[0] - (90+45)) < 1.e-10
    assert abs(angles[1] - 45) < 1.e-10
    assert abs(angles[2] - 180) < 1.e-10
    assert not watershed_workflow.angles.isBadJunctionSharpAngle(river, 20)


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

    assert watershed_workflow.angles.isBadJunctionSharpAngle(river, 20)
    river_orig = river.deepcopy()
    watershed_workflow.angles.smoothJunctionSharpAngles(river, 20)
    assert not watershed_workflow.angles.isBadJunctionSharpAngle(river, 20)

    assert watershed_workflow.utils.isClose(river_orig.linestring, river.linestring)
    print(list(river_orig.children[0].linestring.coords))
    print(list(river.children[0].linestring.coords))
    print('-----')
    print(list(river_orig.children[1].linestring.coords))
    print(list(river.children[1].linestring.coords))
    assert False
    
    

