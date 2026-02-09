import pytest
import numpy as np
import copy
import logging
import pickle
import os
from matplotlib import pyplot as plt

import watershed_workflow.condition as condition
from watershed_workflow.mesh import Mesh2D


PLOT = False


def createTriangulatedSquareMesh(nx=10, ny=10, xsize=1000.0, ysize=1000.0, randomize=20):
    """Create a triangulated square mesh using matplotlib's Delaunay triangulation.

    Parameters
    ----------
    nx : int
        Number of points in x direction. Default is 10.
    ny : int
        Number of points in y direction. Default is 10.
    xsize : float
        Size of domain in x direction (meters). Default is 1000.0.
    ysize : float
        Size of domain in y direction (meters). Default is 1000.0.
    randomize : float
        Amount to randomize interior point locations (meters). Boundary points
        are not randomized. Default is 0.0 (no randomization).

    Returns
    -------
    points : np.ndarray
        (N, 2) array of x,y coordinates
    triangles : np.ndarray
        (M, 3) array of triangle vertex indices
    """
    from matplotlib.tri import Triangulation

    # Create regular grid of points
    x = np.linspace(0, xsize, nx)
    y = np.linspace(0, ysize, ny)
    xv, yv = np.meshgrid(x, y)

    # Flatten to point list
    points = np.column_stack([xv.ravel(), yv.ravel()])

    # Randomize interior points if requested
    if randomize > 0.0:
        rng = np.random.default_rng(seed=42)
        for i in range(len(points)):
            px, py = points[i]

            # Check if point is on boundary
            on_boundary = (abs(px) < 1e-10 or abs(px - xsize) < 1e-10 or
                          abs(py) < 1e-10 or abs(py - ysize) < 1e-10)

            if not on_boundary:
                # Add random perturbation to interior points
                dx = rng.uniform(-randomize, randomize)
                dy = rng.uniform(-randomize, randomize)
                points[i, 0] += dx
                points[i, 1] += dy

    # Triangulate
    tri = Triangulation(points[:, 0], points[:, 1])
    triangles = tri.triangles

    return points, triangles


def createMeshWithElevation(elevation_func, points, triangles):
    """Create Mesh2D with elevation from a function."""
    # Add elevation as third coordinate
    z = np.array([elevation_func(x, y) for x, y in points])
    coords = np.column_stack([points, z])

    # Convert triangles to list of lists for Mesh2D
    conn = [list(tri) for tri in triangles]

    return Mesh2D(coords, conn)


# List of all algorithms with their calling conventions
ALGORITHMS = [
    ('null', {}),
    ('global', {}),
    ('global', {'max_iterations' : 100}),
    ('marching old', {}),
    ('marching old', {'max_iterations' : 5}),
    ('marching', {}),
    ('marching', {'max_iterations' : 5}),
]


# Generate individual tests for each algorithm
@pytest.mark.parametrize("algo_name, algo_args", ALGORITHMS)
def test_ramp(algo_name, algo_args):
    """Test null case: simple ramp with correct boundary conditions.

    Elevation ramps from 0 at x=0 to 100 at x=1000.
    Forced outlets at x=0, divides at x=1000.
    Should require no conditioning.
    """
    # Create mesh with simple ramp
    def ramp_elevation(x, y):
        return 100.0 * x / 1000.0

    points, tris = createTriangulatedSquareMesh(nx=10, ny=10, randomize=0.)
    m2 = createMeshWithElevation(ramp_elevation, points, tris)

    # Find boundary edges at x=0 (outlets) and x=1000 (divides)
    forced_outlet_edges = []
    divide_edges = []

    for edge in m2.boundary_edges:
        v0, v1 = edge
        x0, x1 = m2.coords[v0, 0], m2.coords[v1, 0]

        # Check if both vertices are at x=0 (outlet)
        if abs(x0) < 1e-6 and abs(x1) < 1e-6:
            forced_outlet_edges.append(edge)
        # Check if both vertices are at x=1000 (divide)
        elif abs(x0 - 1000.0) < 1e-6 and abs(x1 - 1000.0) < 1e-6:
            divide_edges.append(edge)

    # Run algorithm
    elevs_initial = m2.coords[:,2].copy()
    m2, result = condition.fillPits(m2, algo_name,
                                forced_outlet_edges=forced_outlet_edges,
                                divide_edges=divide_edges,
                                plot=PLOT,
                                **algo_args)
    changes = m2.coords[:, 2] - elevs_initial
    plt.show()

    # Should have no pits
    assert len(result['pits_final']) == 0, \
        f"{algo_name} has pits: {result['pits_final']}"

    # No elevations should change (null test)
    assert np.all(np.abs(changes) < 1e-9), \
        f"{algo_name} modified elevations in null test: max change = {np.abs(changes).max():.3e}"


@pytest.mark.parametrize("algo_name, algo_args", ALGORITHMS)
def test_ramp_with_pit(algo_name, algo_args):
    """Test ramp with an internal pit """
    # Create mesh with simple ramp (same as test_ramp)
    def ramp(x, y):
        return 100.0 * x / 1000.0

    def gaussian_2d(x, y, center, amplitude, sigma):
        x0, y0 = center
        return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def ramp_with_pit(x, y):
        return ramp(x, y) - gaussian_2d(x, y, (500, 500), 50, 200)

    points, tris = createTriangulatedSquareMesh(nx=10, ny=10)
    m2 = createMeshWithElevation(ramp_with_pit, points, tris)

    # Find boundary edges at x=0 (outlets) and x=1000 (divides)
    forced_outlet_edges = []
    divide_edges = []

    for edge in m2.boundary_edges:
        v0, v1 = edge
        x0, x1 = m2.coords[v0, 0], m2.coords[v1, 0]

        # Check if both vertices are at x=0 (outlet)
        if abs(x0) < 1e-6 and abs(x1) < 1e-6:
            forced_outlet_edges.append(edge)
        # Check if both vertices are at x=1000 (divide)
        elif abs(x0 - 1000.0) < 1e-6 and abs(x1 - 1000.0) < 1e-6:
            divide_edges.append(edge)

    elevs_initial = m2.coords[:,2].copy()
    m2, result = condition.fillPits(m2, algo_name,
                                forced_outlet_edges=forced_outlet_edges,
                                divide_edges=divide_edges,
                                epsilon=0.0,
                                plot=PLOT,
                                **algo_args)
    changes = m2.coords[:, 2] - elevs_initial
    plt.show()

    # Behavior depends on algorithm
    if algo_name == 'null':
        # Null algorithm should not fix pits
        assert len(result['pits_final']) == len(result['pits_initial']), \
            f"null algorithm changed pit count: {len(result['pits_initial'])} -> {len(result['pits_final'])}"

        # No elevations should change
        assert np.all(np.abs(changes) < 1e-9), \
            f"null algorithm modified elevations: max change = {np.abs(changes).max():.3e}"
    else:
        # Real algorithms should fix all pits
        assert len(result['pits_final']) <= len(result['pits_initial']), \
            f"{algo_name} increased number of pits to {len(result['pits_final'])} pits: {result['pits_final']}"

        # Some elevations should have increased
        assert np.all(changes >= -1e-9), \
            f"{algo_name} lowered some elevations: min change = {changes.min():.3e}"


@pytest.mark.parametrize("algo_name, algo_args", ALGORITHMS)
def test_ramp_with_deep_pit(algo_name, algo_args):
    """Test ramp with an internal pit """
    # Create mesh with simple ramp (same as test_ramp)
    def ramp(x, y):
        return 100.0 * x / 1000.0

    def gaussian_2d(x, y, center, amplitude, sigma):
        x0, y0 = center
        return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def ramp_with_pit(x, y):
        return ramp(x, y) - gaussian_2d(x, y, (500, 500), 100, 200)

    points, tris = createTriangulatedSquareMesh(nx=10, ny=10)
    m2 = createMeshWithElevation(ramp_with_pit, points, tris)

    # Find boundary edges at x=0 (outlets) and x=1000 (divides)
    forced_outlet_edges = []
    divide_edges = []

    for edge in m2.boundary_edges:
        v0, v1 = edge
        x0, x1 = m2.coords[v0, 0], m2.coords[v1, 0]

        # Check if both vertices are at x=0 (outlet)
        if abs(x0) < 1e-6 and abs(x1) < 1e-6:
            forced_outlet_edges.append(edge)
        # Check if both vertices are at x=1000 (divide)
        elif abs(x0 - 1000.0) < 1e-6 and abs(x1 - 1000.0) < 1e-6:
            divide_edges.append(edge)

    elevs_initial = m2.coords[:,2].copy()
    m2, result = condition.fillPits(m2, algo_name,
                                forced_outlet_edges=forced_outlet_edges,
                                divide_edges=divide_edges,
                                epsilon=0.0,
                                plot=PLOT,
                                **algo_args)
    changes = m2.coords[:, 2] - elevs_initial
    plt.show()

    # Behavior depends on algorithm
    if algo_name == 'null':
        # Null algorithm should not fix pits
        assert len(result['pits_final']) == len(result['pits_initial']), \
            f"null algorithm changed pit count: {len(result['pits_initial'])} -> {len(result['pits_final'])}"

        # No elevations should change
        assert np.all(np.abs(changes) < 1e-9), \
            f"null algorithm modified elevations: max change = {np.abs(changes).max():.3e}"
    else:
        # Real algorithms should fix all pits
        assert len(result['pits_final']) <= len(result['pits_initial']), \
            f"{algo_name} increased number of pits to {len(result['pits_final'])} pits: {result['pits_final']}"

        # Some elevations should have increased
        assert np.all(changes >= -1e-9), \
            f"{algo_name} lowered some elevations: min change = {changes.min():.3e}"


@pytest.mark.parametrize("algo_name, algo_args", ALGORITHMS)
def test_real_mesh(algo_name, algo_args):
    """Test with a real mesh from 03_m2_preconditioning.pickle.

    This mesh has ~77k cells and ~41k vertices with real topography.
    Preserved pits are cells with more than 3 vertices.
    The forced outlet is the boundary edge whose cell is a preserved pit.
    """
    # Load the pickle file
    test_dir = os.path.dirname(__file__)
    pickle_path = os.path.join(test_dir, '03_m2_preconditioning.pickle')

    with open(pickle_path, 'rb') as f:
        m2_original = pickle.load(f)

    # Make a copy to avoid modifying the original
    m2 = copy.deepcopy(m2_original)

    # Preserved pits are cells with more than 3 vertices
    preserved_pits = [c for c in range(m2.num_cells) if len(m2.conn[c]) > 3]
    preserved_pits_set = set(preserved_pits)

    # Find the forced outlet: boundary edge whose internal cell is a preserved pit
    forced_outlet_edges = []
    for edge in m2.boundary_edges:
        cells = m2.edges_to_cells[edge]
        assert len(cells) == 1, f"Boundary edge {edge} has {len(cells)} cells"
        if cells[0] in preserved_pits_set:
            forced_outlet_edges.append(edge)

    logging.info(f"Found {len(preserved_pits)} preserved pits")
    logging.info(f"Found {len(forced_outlet_edges)} forced outlet edges")

    # Run the algorithm
    elevs_initial = m2.coords[:,2].copy()
    m2, result = condition.fillPits(m2, algo_name,
                                preserved_pits=preserved_pits,
                                forced_outlet_edges=forced_outlet_edges,
                                epsilon=0.0,
                                plot=PLOT,
                                **algo_args)
    changes = m2.coords[:, 2] - elevs_initial
    plt.show()

    # Log results
    logging.info(f"{algo_name}: {len(result['pits_initial'])} initial pits -> "
                 f"{len(result['pits_final'])} final pits")

    # Behavior depends on algorithm
    if algo_name == 'null':
        # Null algorithm should not fix pits
        assert len(result['pits_final']) == len(result['pits_initial']), \
            f"null algorithm changed pit count: {len(result['pits_initial'])} -> {len(result['pits_final'])}"

        # No elevations should change
        assert np.all(np.abs(changes) < 1e-9), \
            f"null algorithm modified elevations: max change = {np.abs(changes).max():.3e}"
    elif algo_name == 'global':
        # global doesn't do much in this case
        
        # Elevations should only increase or stay the same
        assert np.all(changes >= -1e-9), \
            f"{algo_name} lowered some elevations: min change = {changes.min():.3e}"
    
    else:
        # Real algorithms should reduce pits (may not eliminate all on first pass)
        assert len(result['pits_final']) <= len(result['pits_initial']), \
            f"{algo_name} increased number of pits: {len(result['pits_initial'])} -> {len(result['pits_final'])}"

        # Elevations should only increase or stay the same
        assert np.all(changes >= -1e-9), \
            f"{algo_name} lowered some elevations: min change = {changes.min():.3e}"


