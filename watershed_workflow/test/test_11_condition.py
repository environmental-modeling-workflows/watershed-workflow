import pytest
import numpy as np
import logging
import geopandas as gpd
import shapely
import xarray
from matplotlib import pyplot as plt

import watershed_workflow.condition
from watershed_workflow.test.shapes import *

PLOT = False

def make_points_1D(elevs):
    points = {}
    for i, e in enumerate(elevs):
        coords = np.array([i, 0, e])
        if i == 0:
            neighbors = [1, ]
        elif i == len(elevs) - 1:
            neighbors = [i - 1, ]
        else:
            neighbors = [i - 1, i + 1]
        points[i] = watershed_workflow.condition._Point(coords, neighbors)
    return points


def run_test_1D(elev_in, elev_out, alg):
    points = make_points_1D(elev_in)
    if alg == 1:
        watershed_workflow.condition.fillPits1(points, 0)
    elif alg == 2:
        watershed_workflow.condition.fillPits2(points, 0)
    elif alg == 3:
        watershed_workflow.condition.fillPits3(points, 0)

    print("GOT COORDS:")
    print(([points[i].coords[2] for i in range(len(points))]))

    for i in range(len(elev_in)):
        assert (points[i].coords[2] == elev_out[i])


def test_null():
    run_test_1D([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 1)
    run_test_1D([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 2)
    run_test_1D([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 3)


def test_one_pit():
    run_test_1D([0, 1, 3, 2, 4, 5], [0, 1, 3, 3, 4, 5], 1)
    run_test_1D([0, 1, 3, 2, 4, 5], [0, 1, 3, 3, 4, 5], 2)
    run_test_1D([0, 1, 3, 2, 4, 5], [0, 1, 3, 3, 4, 5], 3)


def test_two_pit():
    run_test_1D([0, 1, 3, 2, 1, 4], [0, 1, 3, 3, 3, 4], 1)
    run_test_1D([0, 1, 3, 2, 1, 4], [0, 1, 3, 3, 3, 4], 2)
    run_test_1D([0, 1, 3, 2, 1, 4], [0, 1, 3, 3, 3, 4], 3)


def test_two_pit_backwards():
    run_test_1D([0, 1, 3, 1, 2, 4], [0, 1, 3, 3, 3, 4], 1)
    run_test_1D([0, 1, 3, 1, 2, 4], [0, 1, 3, 3, 3, 4], 2)
    run_test_1D([0, 1, 3, 1, 2, 4], [0, 1, 3, 3, 3, 4], 3)


def test_double_pit():
    run_test_1D([0, 1, 2, 1, 3, 1, 3, 5], [0, 1, 2, 2, 3, 3, 3, 5], 1)
    run_test_1D([0, 1, 2, 1, 3, 1, 3, 5], [0, 1, 2, 2, 3, 3, 3, 5], 2)
    run_test_1D([0, 1, 2, 1, 3, 1, 3, 5], [0, 1, 2, 2, 3, 3, 3, 5], 3)


def test_bad_outlet_pit():
    run_test_1D([0, 1, 3, -1, 4, 5], [0, 1, 3, 3, 4, 5], 1)
    run_test_1D([0, 1, 3, -1, 4, 5], [0, 1, 3, 3, 4, 5], 2)
    run_test_1D([0, 1, 3, -1, 4, 5], [0, 1, 3, 3, 4, 5], 3)


def make_reach(elevs):
    """Helper function to make a straight reach along y=0 given elevs."""
    linestring = shapely.geometry.LineString([(i,0,e) for (i,e) in enumerate(elevs)])
    df = gpd.GeoDataFrame(geometry=[linestring,])
    rivers = watershed_workflow.river_tree.createRivers(df)
    return rivers[0]

def plotLineString(reach, fmt, ax):
    coords = np.array(reach.linestring.coords)
    return ax.plot(coords[:,0], coords[:,2], fmt)
    
    
def test_smoothProfile():
    """Tests smoothing of reach elevations

    Note smoothing is very aggresive, at least at these length scales.
    May need to change this or provide options, which may break this
    test which assumes you end up with a straight line after
    smoothing.

    """
    if PLOT:
        fig, ax = plt.subplots(1,1)

    elevs = [5, 2, 4, 1, 3, 0]
    reach = make_reach(elevs)
    mean = np.array(elevs).mean()

    if PLOT:
        plotLineString(reach, 'k-x', ax)

    watershed_workflow.condition.smoothProfile(reach)
    if PLOT:
        plotLineString(reach, 'r-+', ax)
        plt.show()

    coords = np.array(reach.linestring.coords)
    dz = coords[1:,2] - coords[:-1,2]
    assert max(dz) < 0
    assert max(dz) - min(dz) < 0.03
    assert abs(np.mean(coords[:,2]) - mean) < 1.e-5


def test_smoothWithLowerProfile():
    """Tests smooth with the lower=True option, which prefers to not
    let smoothing raise up the elevation profile due to bad individual
    high elevations."""
    
    if PLOT:
        fig, ax = plt.subplots(1,1)

    elevs = [5, 2, 4, 1, 3, 0]
    reach = make_reach(elevs)
    mean = np.array(elevs).mean()

    if PLOT:
        plotLineString(reach, 'k-x', ax)

    watershed_workflow.condition.smoothProfile(reach, True)
    if PLOT:
        plotLineString(reach, 'r-+', ax)
        plt.show()

    coords = np.array(reach.linestring.coords)
    dz = coords[1:,2] - coords[:-1,2]
    assert max(dz) < 0
    assert max(dz) - min(dz) < 0.03
    assert np.mean(coords[:,2]) < mean - 0.1


def test_localMonotonicity1():
    """Local monotonicity of one reach"""
    if PLOT:
        fig, ax = plt.subplots(1,1)

    elevs = [5, 2, 4, 1, 3, 0]
    reach = make_reach(elevs)
    assert not reach.isLocallyMonotonic()
    coords1 = np.array(reach.linestring.coords)


    if PLOT:
        plotLineString(reach, 'k-x', ax)

    watershed_workflow.condition.enforceLocalMonotonicity(reach)
    if PLOT:
        plotLineString(reach, 'r-+', ax)
        plt.show()

    coords2 = np.array(reach.linestring.coords)
    assert all(coords2[:,2] <= coords1[:,2])
    assert max(coords2[1:,2] - coords2[:-1,2]) <= 0
    assert reach.isLocallyMonotonic()


def test_localMonotonicity2():
    """Local monotonicity of one reach with the moving='upstream' option."""
    if PLOT:
        fig, ax = plt.subplots(1,1)

    elevs = [5, 2, 4, 1, 3, 0]
    reach = make_reach(elevs)
    assert not reach.isLocallyMonotonic()
    coords1 = np.array(reach.linestring.coords)

    if PLOT:
        plotLineString(reach, 'k-x', ax)

    watershed_workflow.condition.enforceLocalMonotonicity(reach, moving='upstream')
    if PLOT:
        plotLineString(reach, 'r-+', ax)
        plt.show()

    coords2 = np.array(reach.linestring.coords)
    assert all(coords2[:,2] <= coords1[:,2])
    assert max(coords2[1:,2] - coords2[:-1,2]) <= 0
    assert reach.isLocallyMonotonic()


def test_monotonicity():
    """Multi-reach monotonicity."""
    if PLOT:
        fig, ax = plt.subplots(1,1)

    coords1 = np.array([(0,0,5), (1,0,2), (2,0,4), (3,0,1), (4,0,3), (5,0,0)], 'd')
    coords2 = coords1.copy()
    coords2[:,0] -= 5.0
    coords2[:,2] += 5.0

    coords3 = coords2.copy()
    coords3[:,1] = [5,4,3,2,1,0]
    coords3[:,2] = 5 + np.array([5, 2, 4, 1, -1, 0])

    print(coords1)
    print(coords2)
    print(coords3)
    l1 = shapely.geometry.LineString(coords1)
    l2 = shapely.geometry.LineString(coords2)
    l3 = shapely.geometry.LineString(coords3)
    df = gpd.GeoDataFrame(geometry=[l1,l2,l3])
    river = watershed_workflow.river_tree.createRivers(df)[0]
    assert len(river) == 3
    assert not river.isMonotonic()
    coords_pre = np.concatenate([np.array(r.linestring.coords)[:,2] for r in river])

    if PLOT:
        for reach in river:
            plotLineString(reach, 'k-x', ax)

    watershed_workflow.condition.enforceMonotonicity(river)
    if PLOT:
        for reach in river:
            plotLineString(reach, 'r-+', ax)
        plt.show()

    coords_post = np.concatenate([np.array(r.linestring.coords)[:,2] for r in river])
    assert all(coords_post <= coords_pre)
    assert river.isContinuous()
    assert river.isMonotonic()


def test_monotonicityKnownDepressions():
    if PLOT:
        fig, ax = plt.subplots(1,1)

    coords1 = np.array([(0,0,5), (1,0,2), (2,0,4), (3,0,1), (4,0,3), (5,0,0)], 'd')
    coords2 = coords1.copy()
    coords2[:,0] -= 5.0
    coords2[:,2] += 5.0

    coords3 = coords2.copy()
    coords3[:,1] = [5,4,3,2,1,0]
    coords3[:,2] = 5 + np.array([5, 2, 4, 1, -1, 0])

    print(coords1)
    print(coords2)
    print(coords3)
    l1 = shapely.geometry.LineString(coords1)
    l2 = shapely.geometry.LineString(coords2)
    l3 = shapely.geometry.LineString(coords3)
    df = gpd.GeoDataFrame(geometry=[l1,l2,l3])
    river = watershed_workflow.river_tree.createRivers(df)[0]
    assert len(river) == 3
    assert not river.isMonotonic()
    coords_pre = np.concatenate([np.array(r.linestring.coords)[:,2] for r in river])

    if PLOT:
        for reach in river:
            plotLineString(reach, 'k-x', ax)

    watershed_workflow.condition.enforceMonotonicity(river, known_depressions=[1,])
    if PLOT:
        for reach in river:
            plotLineString(reach, 'r-+', ax)
        plt.show()

    coords_post = np.concatenate([np.array(r.linestring.coords)[:,2] for r in river])
    assert all(coords_post <= coords_pre)
    assert river.isContinuous()
    assert not river.isMonotonic()
    assert river.isMonotonic([1,])
    

@pytest.fixture
def goalpost(watershed_rivers2):
    hucs, rivers = watershed_rivers2
    river = rivers[0]
    assert len(river)  == 3
    watershed_workflow.simplify(hucs, [river,], 50)
    # test precondition
    assert sum(len(r.linestring.coords) for r in river) == 19

    def computeWidth(a): return 10
    m2 = watershed_workflow.tessalateRiverAligned(hucs, [river,], computeWidth, refine_max_area=100)

    # test precondition
    # different tessalates on different machines...?!?
    # on mac this is 1200, on ubuntu 1193?
    assert len(m2.coords) in [1200, 1193]

    # test begins
    # -- elevate based on a simple ramp
    dem = xarray.DataArray(data=np.array([[0,1], [0,1]]),
                           dims=('x', 'y'),
                           coords={'x':[0,400], 'y':[0,500]})

    watershed_workflow.condition.setProfileByDEM([river,], dem)
    watershed_workflow.elevate(m2, dem)
    watershed_workflow.condition.distributeProfileToMesh(m2, river)

    # should be monotonic already
    assert river.isContinuous()
    assert river.isMonotonic()
    return m2, river


def test_withHeadwaterDepress(goalpost):
    """Note, this requires a visual check -- all streams should be lowered."""
    m2, river = goalpost

    if PLOT:
        fig, ax = plt.subplots(1,2)
        watershed_workflow.plot.mesh(m2, color='elevation', vmin=-2, vmax=1, ax=ax[0])
        river.plot(color='r', marker='x', ax=ax[0])
    coords1 = m2.coords.copy()

    # depress
    watershed_workflow.condition.enforceMonotonicity(river, 1)
    watershed_workflow.condition.distributeProfileToMesh(m2, river)
    m2.clearGeometryCache()

    coords2 = m2.coords

    if PLOT:
        watershed_workflow.plot.mesh(m2, color='elevation', vmin=-2, vmax=1, ax=ax[1])
        river.plot(color='r', marker='x', ax=ax[1])
        ax[1].set_title("Confirm river elems are all lower")
        plt.show()

    assert ((coords1 - coords2) >= 0).all()
    assert river.isContinuous()
    assert river.isMonotonic()
    

def test_withDepress(goalpost):
    """Note, this requires a visual check -- all streams should be lowered."""
    m2, river = goalpost

    if PLOT:
        fig, ax = plt.subplots(1,2)
        watershed_workflow.plot.mesh(m2, color='elevation', vmin=-2, vmax=1, ax=ax[0])
        river.plot(color='r', marker='x', ax=ax[0])
    coords1 = m2.coords.copy()

    # depress
    watershed_workflow.condition.enforceMonotonicity(river)
    def burnInDepth(reach): return 1
    watershed_workflow.condition.burnInRiver(river, burnInDepth)
    watershed_workflow.condition.distributeProfileToMesh(m2, river)
    m2.clearGeometryCache()

    coords2 = m2.coords

    if PLOT:
        watershed_workflow.plot.mesh(m2, color='elevation', vmin=-2, vmax=1, ax=ax[1])
        river.plot(color='r', marker='x', ax=ax[1])
        ax[1].set_title("Confirm river elems are all lower")
        plt.show()

    assert ((coords1 - coords2) >= 0).all()
    assert river.isContinuous()
    assert river.isMonotonic()

def test_withDepressAndBanks(goalpost):
    """Note, this requires a visual check -- all streams should be lowered."""
    m2, river = goalpost

    if PLOT:
        fig, ax = plt.subplots(1,2)
        watershed_workflow.plot.mesh(m2, color='elevation', vmin=0, vmax=2, ax=ax[0])
        river.plot(color='r', marker='x', ax=ax[0])
    coords1 = m2.coords.copy()

    # depress
    watershed_workflow.condition.enforceMonotonicity(river)
    watershed_workflow.condition.enforceBankIntegrity(m2, river, 1)
    watershed_workflow.condition.distributeProfileToMesh(m2, river)
    m2.clearGeometryCache()

    coords2 = m2.coords

    if PLOT:
        watershed_workflow.plot.mesh(m2, color='elevation', vmin=0, vmax=2, ax=ax[1])
        river.plot(color='r', marker='x', ax=ax[1])
        ax[1].set_title("Confirm nodes neighboring river are now higher")
        plt.show()

    assert ((coords1 - coords2) <= 0).all()
    assert river.isContinuous()
    assert river.isMonotonic()
