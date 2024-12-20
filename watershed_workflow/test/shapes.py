import pytest

import numpy as np
import shapely.geometry
import geopandas
import watershed_workflow.utils

_tol = 1.e-7


def assert_close(s1, s2, tol=_tol):
    assert (watershed_workflow.utils.isClose(s1, s2, tol))


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
                                 'hydroseq' : hydroseqs,
                                 'dnhydroseq' : dnstream,
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
                                 'hydroseq' : hydroseqs,
                                 'dnhydroseq' : dnstream,
                                 'uphydroseq' : upstream,
                                 'divergence' : divergence,
                                 'geometry' : mls}).set_index('index')
    return df


@pytest.fixture
@to_dataframe
def rivers():
    return [shapely.geometry.LineString([(5, 0), (0, 0)]),
            shapely.geometry.LineString([(8, 3), (5, 0)]),
            shapely.geometry.LineString([(12, -3), (8, -3), (5, 0)]),
            shapely.geometry.LineString([(15, -3), (12, -3)]),
            shapely.geometry.LineString([(12, 0), (12, -3)]),
            ]


#
# Note, this is not valid input for a River object, or at least may
# not return a single river!
#
@pytest.fixture
@to_dataframe
def y_with_junction():
    return [shapely.geometry.LineString([(1, 0), (0, 0)]),
            shapely.geometry.LineString([(1, 1), (1, 0)]),
            shapely.geometry.LineString([(1, -1), (0.5, 0)]),
            ]


# ===== polygons =====
@pytest.fixture
@to_dataframe
def two_boxes():
    return [shapely.geometry.Polygon([(0, -5), (10, -5), (10, 5), (0, 5)]),
            shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5)]),
            ]


@pytest.fixture
@to_dataframe
def three_boxes():
    return [shapely.geometry.Polygon([(0, -5), (10, -5), (10, 5), (0, 5)]),
            shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5)]),
            shapely.geometry.Polygon([(20, -5), (30, -5), (30, 5), (20, 5)]),
            ]


@pytest.fixture
@to_dataframe
def three_more_boxes():
    return [ shapely.geometry.Polygon([(0, -5), (10, -5), (10, 5), (0, 5)]),
             shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5)]),
             shapely.geometry.Polygon([(0, 5), (10, 5), (20, 5), (20, 10), (0, 10)]),
            ]


@pytest.fixture
@to_dataframe
def watershed_poly1():
    return [ shapely.geometry.Polygon([(0, -5), (10, -5), (10, 5), (0, 5)]),
             shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5)]),
             shapely.geometry.Polygon([(0, 5), (10, 5), (20, 5), (20, 10), (0, 10)]),
            ]


@pytest.fixture
@to_dataframe
def watershed_reaches1():
    return [
        shapely.geometry.LineString([(5., 0.), (10., 5), ]),
        shapely.geometry.LineString([(15., 0.), (10., 5), ]),
        shapely.geometry.LineString([(10., 5.), (10, 10)]),
    ]


@pytest.fixture
@to_dataframe
def watershed_poly2():
    """Create watershed polygon, mocking NHDPLus dataset"""
    return [shapely.geometry.Polygon(
        100 * np.array([[0, 0], [1, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [3, 4.5],
                        [2, 5], [1, 4.5], [0, 4], [0, 3], [0, 2], [0, 1]], 'd')),]


@pytest.fixture
@to_dataframe
def watershed_reaches2():
    """Create a list of reaches, mocking NHDPLus dataset"""
    reach1 = shapely.geometry.LineString([(200, 200), (200, 0)])
    reach2 = shapely.geometry.LineString([(50, 300), (100, 300), (100, 200), (200, 200)])
    reach3 = shapely.geometry.LineString([(350, 400), (350, 300), (300, 300), (300, 200),
                                          (200, 200)])
    reach4 = shapely.geometry.LineString([(100, 400), (200, 300)])
    reaches = [reach1, reach2, reach3, reach4]
    return reaches


@pytest.fixture
def watershed_rivers1(watershed_poly1, watershed_reaches1):
    if watershed_poly1 is not None:
        hucs = watershed_workflow.split_hucs.SplitHUCs(watershed_poly1)
    else:
        hucs = None
    if watershed_reaches1 is not None:
        rivers = watershed_workflow.river_tree.createRivers(watershed_reaches1)
    else:
        rivers = None
    return hucs, rivers




@pytest.fixture
def watershed_rivers2(watershed_poly2, watershed_reaches2):
    """The goalpost river network with two rivers."""
    if watershed_poly2 is not None:
        hucs = watershed_workflow.split_hucs.SplitHUCs(watershed_poly2)
    else:
        hucs = None
    if watershed_reaches2 is not None:
        rivers = watershed_workflow.river_tree.createRivers(watershed_reaches2)
    else:
        rivers = None
    return hucs, rivers

