import pytest

import numpy as np
import shapely.geometry
import workflow.utils

_tol = 1.e-7
def assert_close(s1, s2, tol=_tol):
    assert(type(s1) == type(s2))
    if type(s1) is shapely.geometry.LineString:
        assert(len(s1.coords) == len(s2.coords))
        for c1, c2 in zip(s1.coords, s2.coords):
            assert((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 <= tol**2)
    elif type(s1) is shapely.geometry.MultiLineString:
        assert(len(s1) == len(s2))
        cl = workflow.utils.close(s1,s2)
        if not cl:
            print("not close s1:")
            for l in s1:
                print(list(l.coords))
            print("not close s2:")
            for l in s2:
                print(list(l.coords))
        assert(cl)

    else:
        raise NotImplementedError("Not implemented for type '%r'"%type(s1))


# ===== river shapes ===== 

@pytest.fixture
def y():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1),(1,0)]]
    return shapely.geometry.MultiLineString(points)

@pytest.fixture
def y_with_extension():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1),(1,0)],
              [(2,-1),(1,-1)]]
    return shapely.geometry.MultiLineString(points)

@pytest.fixture
def two_ys():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1), (1,0)],
              [(12,0), (11,0)],
              [(12,1), (12,0)],
              [(12,-1), (12,0)]]
    return shapely.geometry.MultiLineString(points)


@pytest.fixture
def y_with_junction():
    points = [[(1,0), (0,0)],
              [(1,1), (1,0)],
              [(1,-1), (0.5,0)]]
    return shapely.geometry.MultiLineString(points)


@pytest.fixture
def rivers():
    points = [[(5,0), (0,0)],
              [(8,3), (5,0)],
              [(12,-3), (8,-3), (5,0)],
              [(15,-3), (12,-3)],
              [(12,0), (12,-3)]]
    return shapely.geometry.MultiLineString(points)


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
