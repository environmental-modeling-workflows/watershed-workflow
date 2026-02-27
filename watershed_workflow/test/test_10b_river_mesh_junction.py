"""Tests for projectJunction() via createRiverMesh() on Y-junction river networks."""
import pytest
import numpy as np
import shapely.geometry
import geopandas
from matplotlib import pyplot as plt

import watershed_workflow.river_tree
import watershed_workflow.river_mesh
import watershed_workflow.utils
import watershed_workflow.sources.standard_names as names
from watershed_workflow.river_mesh import createRiverMesh


def _meshYJunction(trunk_coords, left_coords, right_coords,
                   trunk_width, left_width, right_width,
                   check_convexity=True):
    """Create a river mesh for a Y-junction (one trunk, two tributary children).

    All three reaches must share a common junction point: trunk_coords[-1]
    must equal left_coords[0] and right_coords[0].

    Parameters
    ----------
    trunk_coords : list of tuple
        Coordinate list for the downstream trunk reach, ordered upstream
        to downstream (i.e. coords[0] is the junction, coords[-1] is the
        outlet).
    left_coords : list of tuple
        Coordinate list for the left (paddler's right) tributary, ordered
        upstream to downstream so that coords[-1] is the junction.
    right_coords : list of tuple
        Coordinate list for the right (paddler's left) tributary, ordered
        upstream to downstream so that coords[-1] is the junction.
    trunk_width : float
        River width for the trunk reach.
    left_width : float
        River width for the left tributary.
    right_width : float
        River width for the right tributary.
    check_convexity : bool, optional
        Passed through to createRiverMesh.  Set to False to skip the
        convexity check/fix pass, allowing degenerate cases to return
        without raising.  Default is True.

    Returns
    -------
    river : River
        The constructed river tree.
    coords : np.ndarray
        Mesh coordinate array (N x 2).
    elems : list of list of int
        Element connectivity (each entry is a list of coordinate indices).
    """
    trunk = shapely.geometry.LineString(trunk_coords)
    left = shapely.geometry.LineString(left_coords)
    right = shapely.geometry.LineString(right_coords)

    reaches = geopandas.GeoDataFrame(geometry=[trunk, left, right])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')
    assert len(rivers) == 1, "Expected a single connected river tree"
    river = rivers[0]

    def computeWidth(reach):
        if reach.linestring.equals(trunk):
            return trunk_width
        elif reach.linestring.equals(left):
            return left_width
        else:
            return right_width

    coords, elems = createRiverMesh(river, computeWidth, check_convexity=check_convexity)
    return river, coords, elems


def _plot_junction(request, river, coords, elems, title=''):
    if not request.config.getoption('--plot'):
        return
    fig, ax = plt.subplots(1, 1)
    river.plot(ax=ax, color='b', marker='x')
    ax.scatter(coords[:, 0], coords[:, 1], marker='o', color='g', zorder=5)
    polys = geopandas.GeoDataFrame(
        geometry=[shapely.geometry.Polygon(coords[e]).exterior for e in elems])
    polys.plot(ax=ax, color='r', alpha=0.3)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def _assertConvex(coords, elems):
    """Assert that every element polygon is convex."""
    for i, e in enumerate(elems):
        assert watershed_workflow.utils.isConvex(coords[e]), \
            f"Element {i} is not convex: coords = {coords[e]}"


def _assertNonOverlapping(coords, elems, tol=1e-10):
    """Assert that no two element polygons have a non-trivial area of overlap."""
    polys = [shapely.geometry.Polygon(coords[e]) for e in elems]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            area = polys[i].intersection(polys[j]).area
            assert area <= tol, \
                f"Elements {i} and {j} overlap by area {area}"


def _meshTwoReaches(upstream_coords, downstream_coords,
                    upstream_width, downstream_width,
                    check_convexity=True):
    """Create a river mesh for two inline reaches sharing a junction point.

    upstream_coords[-1] must equal downstream_coords[0].

    Parameters
    ----------
    upstream_coords : list of tuple
        Coordinate list for the upstream reach, ordered upstream to downstream.
    downstream_coords : list of tuple
        Coordinate list for the downstream reach, ordered upstream to downstream.
    upstream_width : float
        River width for the upstream reach.
    downstream_width : float
        River width for the downstream reach.
    check_convexity : bool, optional
        Passed through to createRiverMesh.  Default is True.

    Returns
    -------
    river : River
        The constructed river tree.
    coords : np.ndarray
        Mesh coordinate array (N x 2).
    elems : list of list of int
        Element connectivity (each entry is a list of coordinate indices).
    """
    upstream = shapely.geometry.LineString(upstream_coords)
    downstream = shapely.geometry.LineString(downstream_coords)

    reaches = geopandas.GeoDataFrame(geometry=[upstream, downstream])
    rivers = watershed_workflow.river_tree.createRivers(reaches, method='geometry')
    assert len(rivers) == 1, "Expected a single connected river tree"
    river = rivers[0]

    def computeWidth(reach):
        if reach.linestring.equals(downstream):
            return downstream_width
        else:
            return upstream_width

    coords, elems = createRiverMesh(river, computeWidth, check_convexity=check_convexity)
    return river, coords, elems


#
# Two-reach tests: elbow junction exercising projectTwo()
#

def test_two_reaches_angled_same_width(request):
    """Two reaches at a significant angle (~90 deg), same width."""
    # downstream goes right, upstream comes from above -- clear elbow at origin
    downstream_coords = [(0, 0), (10, 0), (20, 0)]
    upstream_coords = [(0, 10), (0, 5), (0, 0)]

    river, coords, elems = _meshTwoReaches(upstream_coords, downstream_coords, 1.0, 1.0)

    _plot_junction(request, river, coords, elems, 'two reaches 90 deg, same width')

    assert len(elems) == sum(len(r.linestring.coords) - 1 for r in river)
    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_two_reaches_angled_different_width(request):
    """Two reaches at a significant angle (~90 deg), downstream 5x wider than upstream."""
    downstream_coords = [(0, 0), (10, 0), (20, 0)]
    upstream_coords = [(0, 10), (0, 5), (0, 0)]

    river, coords, elems = _meshTwoReaches(upstream_coords, downstream_coords,
                                           upstream_width=1.0, downstream_width=5.0,
                                           check_convexity=False)

    _plot_junction(request, river, coords, elems, 'two reaches 90 deg, different width')

    assert len(elems) == sum(len(r.linestring.coords) - 1 for r in river)
    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_two_reaches_acute_angle_same_width(request):
    """Two reaches at an acute bend angle (~15 deg), same width.

    The upstream reach approaches from the upper-left at 165 deg from the
    downstream direction, giving a 15 deg bend at the junction.
    """
    import math
    # upstream direction makes 165 deg with downstream (+x), so it comes from
    # upper-left: upstream coords point toward (0,0) from direction 165 deg
    theta = math.radians(165)
    upstream_coords = [(-20 * math.cos(theta), -20 * math.sin(theta)),
                       (-10 * math.cos(theta), -10 * math.sin(theta)),
                       (0, 0)]
    downstream_coords = [(0, 0), (10, 0), (20, 0)]

    river, coords, elems = _meshTwoReaches(upstream_coords, downstream_coords, 1.0, 1.0)

    _plot_junction(request, river, coords, elems, 'two reaches 15 deg bend, same width')

    assert len(elems) == sum(len(r.linestring.coords) - 1 for r in river)
    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_two_reaches_acute_angle_different_width(request):
    """Two reaches at an acute bend angle (~15 deg), downstream 5x wider than upstream."""
    import math
    theta = math.radians(165)
    upstream_coords = [(-20 * math.cos(theta), -20 * math.sin(theta)),
                       (-10 * math.cos(theta), -10 * math.sin(theta)),
                       (0, 0)]
    downstream_coords = [(0, 0), (10, 0), (20, 0)]

    river, coords, elems = _meshTwoReaches(upstream_coords, downstream_coords,
                                           upstream_width=1.0, downstream_width=3.0,
                                           check_convexity=False)

    _plot_junction(request, river, coords, elems, 'two reaches 15 deg bend, different width')

    assert len(elems) == sum(len(r.linestring.coords) - 1 for r in river)
    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_two_reaches_nearly_parallel_same_width(request):
    """Two reaches nearly parallel (shallow bend), same width."""
    # downstream goes right; upstream comes in at a shallow angle from upper-left
    downstream_coords = [(0, 0), (10, 0), (20, 0)]
    upstream_coords = [(-20, 2), (-10, 1), (0, 0)]

    river, coords, elems = _meshTwoReaches(upstream_coords, downstream_coords, 1.0, 1.0)

    _plot_junction(request, river, coords, elems, 'two reaches nearly parallel, same width')

    assert len(elems) == sum(len(r.linestring.coords) - 1 for r in river)
    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_two_reaches_nearly_parallel_different_width(request):
    """Two reaches nearly parallel (shallow bend), downstream 5x wider than upstream."""
    downstream_coords = [(0, 0), (10, 0), (20, 0)]
    upstream_coords = [(-20, 2), (-10, 1), (0, 0)]

    river, coords, elems = _meshTwoReaches(upstream_coords, downstream_coords,
                                           upstream_width=1.0, downstream_width=5.0,
                                           check_convexity=False)

    _plot_junction(request, river, coords, elems,
                   'two reaches nearly parallel, different width')

    assert len(elems) == sum(len(r.linestring.coords) - 1 for r in river)
    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


#
# Y-junction tests: exercising projectJunction()
#

def test_symmetric_y_junction(request):
    """Symmetric Y: two equal tributaries meeting at a right angle, uniform width."""
    # trunk goes from junction (0,0) downward to (0,-4)
    trunk_coords = [(0, 0), (0, -2), (0, -4)]
    # left tributary comes from upper-left (-3,3) down to junction
    left_coords = [(-3, 3), (-2, 2), (-1, 1), (0, 0)]
    # right tributary comes from upper-right (3,3) down to junction
    right_coords = [(3, 3), (2, 2), (1, 1), (0, 0)]

    river, coords, elems = _meshYJunction(trunk_coords, left_coords, right_coords,
                                          1.0, 1.0, 1.0)

    _plot_junction(request, river, coords, elems, 'symmetric Y junction')

    # basic sanity: correct element count
    expected_elems = sum(len(r.linestring.coords) - 1 for r in river)
    assert len(elems) == expected_elems

    # junction element (upstream element of trunk) must be a pentagon (5 nodes)
    junction_elem = river[names.ELEMS][0]
    assert len(junction_elem) == 5, \
        f"Junction element should be a pentagon, got {len(junction_elem)} nodes"

    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_asymmetric_y_junction(request):
    """Asymmetric Y: tributaries at different angles, equal width."""
    # trunk straight downward
    trunk_coords = [(0, 0), (0, -3)]
    # left tributary comes in at a steep angle
    left_coords = [(-1, 4), (-0.5, 2), (0, 0)]
    # right tributary comes in nearly horizontal
    right_coords = [(4, 1), (2, 0.5), (0, 0)]

    river, coords, elems = _meshYJunction(trunk_coords, left_coords, right_coords,
                                          1.0, 1.0, 1.0)

    _plot_junction(request, river, coords, elems, 'asymmetric Y junction')

    junction_elem = river[names.ELEMS][0]
    assert len(junction_elem) == 5

    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_variable_width_y_junction(request):
    """Y junction with different widths on trunk and tributaries."""
    trunk_coords = [(0, 0), (0, -2), (0, -4)]
    left_coords = [(-3, 3), (-1.5, 1.5), (0, 0)]
    right_coords = [(3, 3), (1.5, 1.5), (0, 0)]

    river, coords, elems = _meshYJunction(trunk_coords, left_coords, right_coords,
                                          trunk_width=2.0,
                                          left_width=0.8,
                                          right_width=0.8)

    _plot_junction(request, river, coords, elems, 'variable width Y junction')

    junction_elem = river[names.ELEMS][0]
    assert len(junction_elem) == 5

    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)


def test_extreme_width_ratio_y_junction(request):
    """Y junction where trunk and one tributary are ~10x wider than the other.

    Segment lengths are kept larger than all widths (ds > width) so that
    the only problem under test is the miter algorithm's handling of
    heterogeneous widths, not the separate ds < width degeneracy.

    The convexity check/fix pass is disabled so that the raw projection
    result is returned; the test then asserts the mesh is convex and
    non-overlapping, which is expected to fail until the algorithm is fixed.
    """
    # trunk goes straight down; ds = 20 >> max width 5
    trunk_coords = [(0, 0), (0, -20)]
    # left tributary at ~45 deg; ds ~21 >> 5
    left_coords = [(-30, 20), (-15, 10), (0, 0)]
    # right tributary at a shallow angle; ds ~20 >> 0.5
    right_coords = [(40, 4), (20, 2), (0, 0)]

    river, coords, elems = _meshYJunction(trunk_coords, left_coords, right_coords,
                                          trunk_width=5.0,
                                          left_width=5.0,
                                          right_width=0.5,
                                          check_convexity=False)

    _plot_junction(request, river, coords, elems, 'extreme width ratio Y junction')

    junction_elem = river[names.ELEMS][0]
    assert len(junction_elem) == 5

    _assertConvex(coords, elems)
    _assertNonOverlapping(coords, elems)
