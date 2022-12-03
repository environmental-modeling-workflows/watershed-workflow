import pytest
import numpy as np
import math
import shapely

import watershed_workflow.mesh
import watershed_workflow.utils
import watershed_workflow.densify_rivers_hucs
import watershed_workflow.river_tree
import watershed_workflow.create_river_mesh
import watershed_workflow.split_hucs


def assert_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    assert (len(l1) == len(l2))
    for a, b in zip(l1, l2):
        assert (a == b)


def watershed_poly():
    """Create watershed polygon, mocking NHDPLus dataset"""
    ws1 = shapely.geometry.Polygon(
        100 * np.array([[0, 0], [1, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [3, 4.5],
                        [2, 5], [1, 4.5], [0, 4], [0, 3], [0, 2], [0, 1]], 'd'))
    return ws1


def reaches():
    """Create a list of reaches, mocking NHDPLus dataset"""
    reach1 = shapely.geometry.LineString([(200, 200), (200, 0)])
    reach2 = shapely.geometry.LineString([(50, 300), (100, 300), (100, 200), (200, 200)])
    reach3 = shapely.geometry.LineString([(350, 400), (350, 300), (300, 300), (300, 200),
                                          (200, 200)])
    reach4 = shapely.geometry.LineString([(100, 400), (200, 300)])
    reaches = [reach1, reach2, reach3, reach4]
    return reaches


@pytest.fixture
def watershed_rivers():
    my_hucs = [watershed_poly()]
    watershed = watershed_workflow.split_hucs.SplitHUCs(my_hucs)
    rivers = watershed_workflow.construct_rivers(watershed, reaches(), method='geometry')
    return watershed, rivers


@pytest.fixture
def river_small():
    reach1 = shapely.geometry.LineString([(4, 10), (4.01, 5.0), (4, 0)])
    reach2 = shapely.geometry.LineString([(0, 20), (2, 15.01), (4, 10)])
    reach3 = shapely.geometry.LineString([(8, 20), (6, 15.01), (4, 10)])
    reaches = [reach1, reach2, reach3]
    rivers = watershed_workflow.construct_rivers(hucs=None, reaches=reaches, method='geometry')
    return rivers[0]


@pytest.fixture
def corr_small():
    corr_coords = [(3.500000999997, 0.0009999980000059787), (3.5099990000009997, 5.0),
                   (3.5001921882905838, 9.903405855207968), (1.535761764959111, 14.824304065653306),
                   (0.0, 20.0), (2.4642382350408334, 15.195695934346832), (4.0, 11.348612713124119),
                   (5.535761764959166, 15.195695934346832), (8.0, 20.0),
                   (6.464238235040834, 14.824304065653168), (4.500192262191668, 9.90436890366601),
                   (4.510000999999, 5.0), (4.499999000003, -0.0009999980000059787),
                   (3.500000999997, 0.0009999980000059787)]
    return shapely.geometry.Polygon(corr_coords)


@pytest.fixture
def watershed_small():
    edge1 = [[x, 0] for x in np.arange(0, 20, 4)]
    edge2 = [[16, y] for y in np.arange(4, 24, 4)]
    edge3 = [[x, 20] for x in np.arange(12, -4, -4)]
    edge4 = [[0, y] for y in np.arange(16, 0, -4)]
    ws = shapely.geometry.Polygon(edge1 + edge2 + edge3 + edge4)
    watershed = watershed_workflow.split_hucs.SplitHUCs([ws])
    return watershed


def test_densify_rivers_hucs(watershed_rivers):
    watershed, rivers = watershed_rivers
    watershed_workflow.simplify(watershed,
                                rivers,
                                simplify_rivers=10,
                                snap=False,
                                cut_intersections=True)

    watershed = watershed_workflow.densify_rivers_hucs.densify_hucs(huc=watershed,
                                                                    huc_raw=watershed,
                                                                    rivers=rivers,
                                                                    use_original=False,
                                                                    limit_scales=[0, 25, 100, 50])
    rivers = watershed_workflow.densify_rivers_hucs.densify_rivers(rivers,
                                                                   rivers,
                                                                   limit=14,
                                                                   use_original=False,
                                                                   treat_collinearity=True)

    assert (51 == len(watershed.segments[0].coords))
    assert (16 == len(rivers[0].segment.coords))
    assert (12 == len(rivers[1].segment.coords))


def test_create_river_corridor(river_small):

    corr = watershed_workflow.create_river_mesh.create_river_corridor(river_small, 1)

    assert (14 == len(corr.exterior.coords))
    assert (0 == math.dist((0.0, 20.0), corr.exterior.coords[4]))
    assert (0 == math.dist((4.0, 11.348612713124119), corr.exterior.coords[6]))


def test_to_quads(river_small, corr_small):

    quads = watershed_workflow.create_river_mesh.to_quads(river_small, corr_small, 1)
    assert (6 == len(quads))
    assert (4 == len(quads[0]))
    assert (5 == len(quads[1]))
    assert (3 == len(quads[3]))
    assert (3 == len(quads[3]))
    assert_list_same([1, 2, 6, 10, 11], quads[1])
    assert_list_same([7, 8, 9], quads[-1])


def test_traingulate(watershed_small, river_small, corr_small):
    points, elems = watershed_workflow.triangulate(watershed_small, [river_small], [corr_small],
                                                   mesh_rivers=True,
                                                   tol=0.1,
                                                   refine_min_angle=32,
                                                   refine_distance=[2, 5, 5, 10],
                                                   diagnostics=False)
    areas = np.array([watershed_workflow.utils.triangle_area(points[e]) for e in elems])

    assert (44 == len(points))
    assert (56 == len(elems))
    assert (math.isclose(8.895187, max(areas), rel_tol=1e-4))
    assert (math.isclose(2.655299, min(areas), rel_tol=1e-4))
