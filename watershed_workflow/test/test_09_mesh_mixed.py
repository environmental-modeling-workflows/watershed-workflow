import pytest
import numpy as np
import math
import shapely
import geopandas

import watershed_workflow.mesh
import watershed_workflow.utils
import watershed_workflow.resampling
import watershed_workflow.river_tree
import watershed_workflow.river_mesh
import watershed_workflow.split_hucs

from watershed_workflow.test.shapes import *


def assert_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    assert (len(l1) == len(l2))
    for a, b in zip(l1, l2):
        assert (a == b)


@pytest.fixture
def river_small():
    reach1 = shapely.geometry.LineString([(4, 10), (4.01, 5.0), (4, 0)])
    reach2 = shapely.geometry.LineString([(1, 19), (2, 15.01), (4, 10)])
    reach3 = shapely.geometry.LineString([(8, 19), (6, 15.01), (4, 10)])
    reaches = geopandas.GeoDataFrame(geometry=[reach1, reach2, reach3])
    rivers = watershed_workflow.river_tree.createRiverTrees(reaches, method='geometry')
    return rivers[0]


@pytest.fixture
def corr_small():
    corr_coords = [(3.500000999997, 0.0009999980000059787), (3.5099990000009997, 5.0),
                   (3.5001921882905838, 9.903405855207968), (1.523171889094147, 14.855841704695042),
                   (1.0, 19.0), (2.476828110905853, 15.164158295304956), (4.0, 11.348612713124119),
                   (5.543501505461575, 15.215083984305364), (8.0, 19.0),
                   (6.456498494538425, 14.804916015694635), (4.500192262191668, 9.90436890366601),
                   (4.510000999999, 5.0), (4.499999000003, -0.0009999980000059787),
                   (3.500000999997, 0.0009999980000059787)]
    return shapely.geometry.Polygon(corr_coords)


@pytest.fixture
def watershed_small():
    seg1 = shapely.geometry.LineString([[16, 4], [16, 8], [16, 12], [16, 16], [16, 20], [12, 20],
                                        [8, 20], [4, 20], [0, 20], [0, 16], [0, 12], [0, 8], [0, 4],
                                        [0, 0], [4, 0]])
    seg2 = shapely.geometry.LineString([[4, 0], [8, 0], [12, 0], [16, 0], [16, 4]])

    ws = geopandas.GeoDataFrame(geometry=[shapely.geometry.Polygon(seg1.coords[:] + seg2.coords[:]),])
    watershed = watershed_workflow.split_hucs.SplitHUCs(ws)

    for i, seg in enumerate([seg1, seg2]):
        watershed.linestrings[i] = seg
    return watershed

#
# NEED MORE TESTS HERE -- there are 1000+ lines of code in river_mesh...
#


def test_create_river_corridor(river_small):
    corr = watershed_workflow.river_mesh.createRiverCorridor(river_small, 1)

    assert (14 == len(corr.exterior.coords))
    assert (0 == math.dist((1.0, 19.0), corr.exterior.coords[4]))
    assert (0 == math.dist((4.0, 11.348612713124119), corr.exterior.coords[6]))


def test_to_quads(river_small, corr_small):
    quads = watershed_workflow.river_mesh.createRiverQuads(river_small, corr_small, 1)
    assert (6 == len(quads))
    assert (4 == len(quads[0]))
    assert (5 == len(quads[1]))
    assert (3 == len(quads[3]))
    assert (3 == len(quads[3]))
    assert_list_same([1, 2, 6, 10, 11], quads[1])
    assert_list_same([7, 8, 9], quads[-1])


# def test_triangulate(watershed_small, river_small):
#     points, elems = watershed_workflow.tessalate_river_aligned(watershed_small, [river_small],
#                                                                river_width=1,
#                                                                tol=0.1,
#                                                                refine_min_angle=32,
#                                                                refine_distance=[2, 5, 5, 10],
#                                                                diagnostics=False)
#     areas = np.array([watershed_workflow.utils.triangle_area(points[e]) for e in elems])

#     assert (47 == len(points))
#     assert (68 == len(elems))
