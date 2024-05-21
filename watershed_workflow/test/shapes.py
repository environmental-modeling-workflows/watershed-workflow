import pytest

import numpy as np
import shapely.geometry
import geopandas
import watershed_workflow.utils

_tol = 1.e-7


def assert_close(s1, s2, tol=_tol):
    assert (watershed_workflow.utils.close(s1, s2, tol))


# ===== river shapes =====

def to_dataframe(shape_func):
    def _func():
        shapes = shape_func()
        df = geopandas.GeoDataFrame({'index':range(len(shapes)),
                                     'geometry':shapes}).set_index('index')
        return df
    return _func


@pytest.fixture
@to_dataframe
def y():
    points = [[(1, 0), (0, 0)], [(1, 1), (1, 0)], [(1, -1), (1, 0)]]
    return list(shapely.geometry.MultiLineString(points).geoms)


@pytest.fixture
@to_dataframe
def y_with_extension():
    points = [[(1, 0), (0, 0)], [(1, 1), (1, 0)], [(1, -1), (1, 0)], [(2, -1), (1, -1)]]
    return list(shapely.geometry.MultiLineString(points).geoms)


@pytest.fixture
def two_ys():
    points = [[(1, 0), (0, 0)], [(1, 1), (1, 0)], [(1, -1), (1, 0)], [(12, 0), (11, 0)],
              [(12, 1), (12, 0)], [(12, -1), (12, 0)]]
    mls = list(shapely.geometry.MultiLineString(points).geoms)
    hydroseqs = [1, 2, 3, 4, 5, 6]
    dnstream = [-1, 1, 1, -1, 4, 4]


    df = geopandas.GeoDataFrame({'index' : hydroseqs,
                                 'HydrologicSequence' : hydroseqs,
                                 'DownstreamMainPathHydroSeq' : dnstream,
                                 'geometry' : mls}).set_index('index')
    return df


@pytest.fixture
def braided_stream():
    points = [[(1, 0), (0, 0)], [(2, 1), (1, 0)], [(3, 0), (2, 1)], [(4, 0), (3, 0)],
              [(2, -1), (1, 0)], [(3, 0), (2, -1)]]
    mls = list(shapely.geometry.MultiLineString(points).geoms)
    hydroseqs = [1, 2, 3, 6, 4, 5]
    dnstream = [-1, 1, 2, 3, 1, 4]
    upstream = [2, 3, 6, -1, 5, 6]
    divergence = [0, 0, 1, 0, 0, 2]

    df = geopandas.GeoDataFrame({'index' : range(len(mls)),
                                 'HydrologicSequence' : hydroseqs,
                                 'DownstreamMainPathHydroSeq' : dnstream,
                                 'UpstreamMainPathHydroSeq' : upstream,
                                 'DivergenceCode' : divergence,
                                 'geometry' : mls}).set_index('index')
    return df


@pytest.fixture
@to_dataframe
def rivers():
    points = [[(5, 0), (0, 0)], [(8, 3), (5, 0)], [(12, -3), (8, -3), (5, 0)], [(15, -3), (12, -3)],
              [(12, 0), (12, -3)]]
    return list(shapely.geometry.MultiLineString(points).geoms)


#
# Note, this is not valid input for a River object, or at least may
# not return a single river!
#
@pytest.fixture
@to_dataframe
def y_with_junction():
    points = [[(1, 0), (0, 0)], [(1, 1), (1, 0)], [(1, -1), (0.5, 0)]]
    return list(shapely.geometry.MultiLineString(points).geoms)


# ===== polygons =====
@pytest.fixture
@to_dataframe
def two_boxes():
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    b2 = [(10, -5), (20, -5), (20, 5), (10, 5)]
    shps = []
    shps.append(shapely.geometry.Polygon(b1))
    shps.append(shapely.geometry.Polygon(b2))
    return shps


@pytest.fixture
@to_dataframe
def three_boxes():
    b1 = [(0, -5), (10, -5), (10, 5), (0, 5)]
    b2 = [(10, -5), (20, -5), (20, 5), (10, 5)]
    b3 = [(20, -5), (30, -5), (30, 5), (20, 5)]
    shps = []
    shps.append(shapely.geometry.Polygon(b1))
    shps.append(shapely.geometry.Polygon(b2))
    shps.append(shapely.geometry.Polygon(b3))
    return shps
