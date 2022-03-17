import pytest

import numpy as np
import shapely.geometry
import watershed_workflow.utils

_tol = 1.e-7
def assert_close(s1, s2, tol=_tol):
    assert(watershed_workflow.utils.close(s1,s2,tol))

# ===== river shapes ===== 

@pytest.fixture
def y():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1),(1,0)]]
    return list(shapely.geometry.MultiLineString(points).geoms)

@pytest.fixture
def y_with_extension():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1),(1,0)],
              [(2,-1),(1,-1)]]
    return list(shapely.geometry.MultiLineString(points).geoms)

@pytest.fixture
def two_ys():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1), (1,0)],
              [(12,0), (11,0)],
              [(12,1), (12,0)],
              [(12,-1), (12,0)]]
    mls = list(shapely.geometry.MultiLineString(points).geoms)
    hydroseqs = [1,2,3,4,5,6]
    dnstream = [-1,1,1,-1,4,4]
    for seg, hs_id, dn_hs_id in zip(mls, hydroseqs, dnstream):
        seg.properties = {'HydrologicSequence' : hs_id,
                          'DownstreamMainPathHydroSeq' : dn_hs_id}
    return mls


@pytest.fixture
def braided_stream():
    points = [[(1,0), (0,0)],
              [(2,1), (1,0)],
              [(3, 0), (2,1)],
              [(4,0), (3,0)],
              [(2,-1), (1,0)],
              [(3, 0), (2,-1)]]
    mls = list(shapely.geometry.MultiLineString(points).geoms)
    hydroseqs = [1,2,3,6,4,5]
    dnstream = [-1, 1, 2, 3, 1, 4]
    upstream = [2, 3, 6, -1, 5, 6]
    divergence = [0,0,1,0,0,2]
    for seg, hs_id, dn, up, div in zip(mls, hydroseqs, dnstream, upstream, divergence):
        seg.properties = {'HydrologicSequence' : hs_id,
                          'DownstreamMainPathHydroSeq' : dn,
                          'UpstreamMainPathHydroSeq' : up,
                          'DivergenceCode' : div}
    return mls


@pytest.fixture
def rivers():
    points = [[(5,0), (0,0)],
              [(8,3), (5,0)],
              [(12,-3), (8,-3), (5,0)],
              [(15,-3), (12,-3)],
              [(12,0), (12,-3)]]
    return list(shapely.geometry.MultiLineString(points).geoms)


#
# Note, this is not valid input for a River object, or at least may
# not return a single river!
#
@pytest.fixture
def y_with_junction():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1), (0.5,0)]]
    return list(shapely.geometry.MultiLineString(points).geoms)



# ===== polygons =====
@pytest.fixture
def two_boxes():
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5)]
    shps = []
    shps.append(shapely.geometry.Polygon(b1))
    shps.append(shapely.geometry.Polygon(b2))
    return shps


@pytest.fixture
def three_boxes():
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5)]
    b3 = [(20, -5), (30,-5), (30,5), (20,5)]
    shps = []
    shps.append(shapely.geometry.Polygon(b1))
    shps.append(shapely.geometry.Polygon(b2))
    shps.append(shapely.geometry.Polygon(b3))
    return shps
