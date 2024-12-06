import watershed_workflow
import watershed_workflow.resampling
import watershed_workflow.river_tree
import watershed_workflow.split_hucs

from watershed_workflow.test.shapes import *


def test_distance_functor():
    shp = shapely.geometry.Point((0,0))
    functor = watershed_workflow.resampling.ComputeTargetLengthByDistanceToShape((1,5,2,10),shp)

    assert functor(shp) == 5
    assert functor((10000, 100000)) == 10
    assert functor((1,0)) == 5
    assert functor((2,0)) == 10
    assert abs(functor((4./3,0)) - (5 + 5./3)) < 1.e-10
    assert functor((1.5,0)) == 7.5

#
# Uniform tests
#
# -- no old points
def test_uniform_smaller():
    ls_in = shapely.geometry.LineString([ (i,0) for i in range(0,21,10)])
    coords_out = watershed_workflow.resampling._resampleLineStringUniform(ls_in, 2.)
    ls_out = shapely.geometry.LineString(coords_out)
    ls_test = shapely.geometry.LineString([(i,0) for i in range(0,21,2)])
    assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

def test_uniform_bigger():
    ls_in = shapely.geometry.LineString([ (i,0) for i in range(0,21,1)])
    coords_out = watershed_workflow.resampling._resampleLineStringUniform(ls_in, 2.)
    ls_out = shapely.geometry.LineString(coords_out)
    ls_test = shapely.geometry.LineString([(i,0) for i in range(0,21,2)])
    assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)
    
def test_no_keep_corner():
    ls_in = shapely.geometry.LineString([(0,0), (5,0), (5,5)])
    coords_out = watershed_workflow.resampling._resampleLineStringUniform(ls_in, 2.)
    ls_out = shapely.geometry.LineString(coords_out)
    ls_test = shapely.geometry.LineString([(0,0), (2,0), (4,0), (5,1), (5,3), (5,5)])
    assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

# -- prefer old points
if watershed_workflow.resampling.PREFER_OLD_POINTS:
    def test_smaller():
        ls_in = shapely.geometry.LineString([ (i,0) for i in range(0,21,10)])
        ls_out = watershed_workflow.resampling.resampleLineStringUniform(ls_in, 2.)
        ls_test = shapely.geometry.LineString([(i,0) for i in range(0,21,2)])
        assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

    def test_bigger():
        ls_in = shapely.geometry.LineString([ (i,0) for i in range(0,21,1)])
        ls_out = watershed_workflow.resampling.resampleLineStringUniform(ls_in, 2.)
        ls_test = shapely.geometry.LineString([(i,0) for i in range(0,21,2)])
        assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

    def test_keep_corner():
        ls_in = shapely.geometry.LineString([(0,0), (5,0), (5,5)])
        ls_out = watershed_workflow.resampling.resampleLineStringUniform(ls_in, 2.)

        ls_test = shapely.geometry.LineString([(0,0), (5./3,0), (10./3,0), (5,0), (5,5./3), (5,10./3), (5,5)])
        assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

#
# Nonuniform tests
#
# -- no old points
def test_nonuniform_smaller():
    shp = shapely.geometry.Point((-1,0))
    ls_in = shapely.geometry.LineString([ (0,0), (22.5,0) ])
    strat = watershed_workflow.resampling.ComputeTargetLengthByDistanceToShape((1, 5, 11, 10), shp)
    coords_out = watershed_workflow.resampling._resampleLineStringNonuniform(ls_in, strat)
    ls_out = shapely.geometry.LineString(coords_out)
    ls_test = shapely.geometry.LineString([(0,0), (5,0), (12.5,0), (22.5,0)])
    assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)


def test_nonuniform_bigger():
    shp = shapely.geometry.Point((-1,0))
    ls_in = shapely.geometry.LineString([ (0,0), (2.0,0), (7.0,0), (11.0,0), (20.0,0), (22.5,0) ])
    strat = watershed_workflow.resampling.ComputeTargetLengthByDistanceToShape((1, 5, 11, 10), shp)
    coords_out = watershed_workflow.resampling._resampleLineStringNonuniform(ls_in, strat)
    ls_out = shapely.geometry.LineString(coords_out)
    ls_test = shapely.geometry.LineString([(0,0), (5,0), (12.5,0), (22.5,0)])
    assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

def test_nonuniform_no_keep_corner():
    shp = shapely.geometry.Point((-1,0))

    ls_in = shapely.geometry.LineString([ (0,0), (10.0,0), (10,12.5)])
    strat = watershed_workflow.resampling.ComputeTargetLengthByDistanceToShape((1, 5, 11, 10), shp)
    coords_out = watershed_workflow.resampling._resampleLineStringNonuniform(ls_in, strat)
    ls_out = shapely.geometry.LineString(coords_out)

    ls_test = shapely.geometry.LineString([(0,0), (5,0), (10,2.5), (10,12.5)])
    assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

if watershed_workflow.resampling.PREFER_OLD_POINTS:
    # -- keep old points
    def test_nonuniform_keep_corner():
        shp = shapely.geometry.Point((-1,0))

        ls_in = shapely.geometry.LineString([ (0,0), (10.0,0), (10,12.5)])
        strat = watershed_workflow.resampling.ComputeTargetLengthByDistanceToShape((1, 5, 11, 10), shp)
        ls_out = watershed_workflow.resampling.resampleLineStringNonuniform(ls_in, strat)

        points1 = np.array([(0,0), (5,0), (12.5,0)]) # shrunk -- to (10,0)
        points1 = points1 * (10 / 12.5)

        arclens = np.array([0, 10, 20]) # arclens from distance = 10
        arclens = arclens * (12.5 / 20) # rescale to 12.5, the endpoint
        ls_test = shapely.geometry.LineString(
            [p for p in points1]+  [(10,s) for s in arclens[1:]])
        assert watershed_workflow.utils.isClose(ls_out, ls_test, 1.e-10)

    
# with data
def test_resampling(watershed_poly, watershed_reaches):
    watershed = watershed_workflow.split_hucs.SplitHUCs(watershed_poly)
    rivers = watershed_workflow.river_tree.createRivers(watershed_reaches, method='geometry')

    watershed_workflow.resampling.resampleHUCs(watershed, (0,25,10,50), 
                                shapely.ops.unary_union([r.to_mls() for r in rivers]))
                                                  

    watershed_workflow.resampling.resampleRivers(rivers, 14)

    # old resampling algorithm
    #assert (53 == len(watershed.exterior.exterior.coords))
    if watershed_workflow.resampling.PREFER_OLD_POINTS:
        assert (37 == len(watershed.exterior.exterior.coords))
    else:
        assert (34 == len(watershed.exterior.exterior.coords))
    assert (16 == len(rivers[0].linestring.coords))
    assert (12 == len(rivers[1].linestring.coords))
