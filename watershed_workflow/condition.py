from typing import Optional, Iterable, Dict, Tuple, List, Literal, Callable, Set, Union, Any

import numpy as np
import attr
import sortedcontainers, collections
import logging
import math
import shapely
import xarray
import scipy.ndimage
import ipywidgets as widgets

import watershed_workflow.sources.standard_names as names
import watershed_workflow.utils
from watershed_workflow.mesh import Mesh2D, Edge
from watershed_workflow.river_tree import River
import watershed_workflow.data


def _isPit(depth, epsilon, tol):
    """Comparator to epsilon, tol"""
    return depth > -(epsilon - tol)


def _computePitDepth(m2 : Mesh2D,
                     c : int,
                     relative_to : Iterable[int] | None
                     ) -> Tuple[float, float]:
    """Compute pit depth for a cell relative to neighbors and boundary edges.

    A pit's depth is the difference in elevation between the cell's
    centroid and the minimum of its surroundings. For internal cells,
    this is the minimum of neighboring cell centroids. For boundary cells,
    also considers boundary edge elevations.

    Returns
    -------
    internal_pit_depth : float
        Pit depth relative to neighboring cells. Positive indicates
        the cell is lower than all neighbors (a pit). Computed as
        min(neighboring_cell_z) - cell_z.
    boundary_edge_pit_depth : float
        Pit depth relative to boundary edges. Positive indicates the
        cell is lower than all adjacent boundary edges. NaN if c is
        an internal cell. Computed as min(boundary_edge_z) - cell_z.

    Notes
    -----
    Both depths are computed independently and can differ for boundary cells.
    A depth is never negative - if a cell is higher than its surroundings,
    the depth will be negative or zero.
    """
    def isRelativeTo(nc):
        if relative_to is None:
            return True
        else:
            return nc in relative_to

    my_z = m2.computeCentroid(c)[2]

    try:
        other_z = min(m2.computeCentroid(nc)[2] for nc in m2.cell_to_cells[c] if isRelativeTo(nc))
    except ValueError:
        # no neighbors -- first cell in
        other_z = my_z

    pit_depth = other_z - my_z
    be_pit_depth = np.nan

    if len(m2.cell_to_cells[c]) < len(m2.conn[c]):
        # at least one boundary edge!
        be_z = min((m2.coords[e[0],2] + m2.coords[e[1],2]) / 2 \
                   for e in m2.cell_edges[c] if e in m2.boundary_edges)
        be_pit_depth = be_z - my_z

    return other_z - my_z, be_pit_depth


def _measurePit(m2, c,
                preserved_pits,
                forced_outlet_cells,
                optional_outlet_cells,
                divide_cells,
                epsilon,
                tol,
                relative_to : Optional[Iterable[int]] = None
                ):
    """Checks cell c"""

    is_pit = False
    cause = None
    internal_depth = np.nan
    boundary_depth = np.nan

    if c not in preserved_pits:
        internal_depth, boundary_depth = _computePitDepth(m2, c, relative_to)
        if np.isnan(boundary_depth):
            # internal cell, only compare to internal
            is_pit = _isPit(internal_depth, epsilon, tol)
            cause = 'internal'

        elif c in forced_outlet_cells:
            # must be an outlet
            is_pit = _isPit(boundary_depth, epsilon, tol)
            cause = 'forced outlet'

        elif c in divide_cells:
            
            if _isPit(internal_depth, epsilon, tol):
                is_pit = True
                cause = 'divide internal'
            else:
                is_pit = _isPit(-boundary_depth, epsilon, tol)
                cause = 'divide boundary'

        else:
            is_pit = _isPit(boundary_depth, epsilon, tol) and _isPit(internal_depth, epsilon, tol)
            cause = 'boundary internal'

    if is_pit:
        logging.debug(f'  ... {cause} pit {c} with depths {internal_depth}, {boundary_depth}')
        logging.debug(f'      at centroid ({m2.computeCentroid(c)})')
        logging.debug(f'      and coords:')
        for v in m2.conn[c]:
            logging.debug(f'        {m2.coords[v]}')

    return is_pit, cause, internal_depth, boundary_depth


def _metricPitDepth(pit):
    c, cause, internal, boundary = pit
    if cause == 'internal':
        return internal
    elif cause == 'forced outlet':
        return boundary
    elif cause == 'divide internal':
        return internal
    elif cause == 'divide boundary':
        return -boundary
    elif cause == 'boundary internal':
        return max(internal, boundary)
    raise ValueError(f'Unrecognized pit cause {cause}')


def findPits(m2: Mesh2D,
             preserved_pits: Optional[Iterable[int]] = None,
             forced_outlet_edges: Optional[Iterable[Tuple[int, int]]] = None,
             optional_outlet_edges: Optional[Iterable[Tuple[int, int]]] = None,
             divide_edges: Optional[Iterable[Tuple[int, int]]] = None,
             epsilon: float = 0.,
             tol : float = 1.e-8,
             ) -> List[Tuple[int, str, float, float]]:
    """Identify problematic pits (local minima) in the mesh.

    Finds cells whose centroid elevation is lower than their surroundings,
    preventing drainage. Uses sophisticated boundary edge categorization to
    handle outlet and divide edges correctly.

    Parameters
    ----------
    m2 : Mesh2D
        The 2D mesh containing vertex coordinates and cell connectivity.
        Elevations are read from m2.coords[:, 2].
    preserved_pits : Iterable[int], optional
        Cell indices that are intentionally pits (e.g., lakes) and should
        not be reported as problems. Default is the empty list.
    forced_outlet_edges : Iterable[Tuple[int, int]], optional
        Boundary edges that must be outlets. Cells touching these edges
        are pits only if water would flow inward across the boundary.
        Default is the empty list.
    optional_outlet_edges : Iterable[Tuple[int, int]], optional
        Boundary edges that may be outlets. Defaults to all boundary edges
        not in forced_outlet_edges or divide_edges. Cells touching these
        are pits only if trapped both internally and externally.
    divide_edges : Iterable[Tuple[int, int]], optional
        Boundary edges that must not be outlets (watershed divides). Cells
        touching these are pits if water would flow outward OR if trapped
        internally. Default is the empty list.
    epsilon : float, optional
        Minimum elevation increase required to not be a pit. Use the same
        epsilon as fillPits methods for consistency. Default is 0.0.
    tol : float, optional
        Numerical tolerance for roundoff errors. Default is 1.e-8.

    Returns
    -------
    pits : List[Tuple[int, str, float, float]]
        List of pits: [cell, cause, internal_pit_depth, boundary_pit_depth]
    
    Notes
    -----
    Pit detection uses: isPit(depth) = depth > -(epsilon - tol)
    where depth is from computePitDepth().

    Boundary edge partitioning:
    - All boundary edges are categorized into forced_outlet_edges,
      optional_outlet_edges, or divide_edges
    - Unspecified edges default to optional_outlet_edges
    - forced_outlet_edges take precedence in overlaps

    Pit criteria by cell type:
    - Internal cells: isPit(internal_depth)
    - Forced outlet cells: isPit(boundary_depth)
    - Divide edge cells: isPit(internal_depth) OR NOT isPit(boundary_depth)
    - Optional outlet cells: isPit(internal_depth) AND isPit(boundary_depth)
    """
    # process and partition inputs
    forced_outlet_edges, optional_outlet_edges, divide_edges = \
        _partitionOutletEdges(m2, forced_outlet_edges, optional_outlet_edges, divide_edges)

    # outlet edges --> cells
    forced_outlet_cells, optional_outlet_cells, divide_cells = \
        _partitionOutletCells(m2, forced_outlet_edges, optional_outlet_edges, divide_edges)

    # process input of preserved_pits
    preserved_pits = set(preserved_pits) if preserved_pits != None else set()

    return _findPits(m2, preserved_pits, forced_outlet_cells,
                     optional_outlet_cells, divide_cells, epsilon, tol)


def _findPits(m2: Mesh2D,
              preserved_pits: Set[int],
              forced_outlet_cells: Iterable[int],
              optional_outlet_cells: Iterable[int],
              divide_cells: Iterable[int],
              epsilon: float,
              tol : float,
             ) -> List[Tuple[int, str, float, float]]:
    logging.debug('Searching for pits...')

    # Find pits
    pits = []
    for c in range(m2.num_cells):
        res = _measurePit(m2, c, preserved_pits,
                          forced_outlet_cells, optional_outlet_cells, divide_cells,
                          epsilon, tol)
        if res[0]:
            pits.append((c,)+res[1:])
    return pits

def singlePitDepth(p):
    """Given a pit-tuple, returns a single depth used for debugging and info."""
    return np.nanmax(np.array([p[2], p[3]]))

def plotPitFilling(m2: Mesh2D,
                   old_pits : List[Tuple[int, str, float, float]],
                   new_pits : List[Tuple[int, str, float, float]],
                   old_z_verts: np.ndarray,
                   method_name: str,
                   ax : Optional[Any] = None,
                   metrics: Optional[Dict] = None,
                   pit_kwargs = None,
                   vertex_kwargs = None,
                   cell_kwargs = None,
                   ):
    """Plot before/after elevation comparison for pit filling algorithms.

    Creates 3x3 panel plot showing:
    - Row 0: Pit depths (old, new, delta)
    - Row 1: Vertex elevations (old, new, delta)
    - Row 2: Cell centroid elevations (old, new, delta)

    Includes dynamic colormap scaling that automatically adjusts color limits
    when zooming/panning the plot for detailed inspection of specific regions.

    Parameters
    ----------
    m2 : Mesh2D
        The mesh with new elevations in m2.coords[:,2]
    old_pits : List[Tuple[int, str, float, float]]
        List of pits before filling
    new_pits : List[Tuple[int, str, float, float]]
        List of pits after filling
    old_z_verts : np.ndarray
        Copy of original vertex elevations before modification
    method_name : str
        Name of the algorithm (for title)
    ax : array of matplotlib axes, optional
        If provided, should be 3x3 array of axes
    metrics : dict, optional
        Dictionary with metrics to display on plot. Expected keys:
        'pits_initial', 'pits_final', 'rmse', 'mae', 'max', 'num_modified'
    pit_kwargs : dict, optional
        Additional arguments passed to m2.plot() for pit depth plots
    vertex_kwargs : dict, optional
        Additional arguments passed to m2.plotVertices() for vertex plots
    cell_kwargs : dict, optional
        Additional arguments passed to m2.plot() for cell plots

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : np.ndarray
        Array of axes (3x3)

    Notes
    -----
    Dynamic colormap scaling groups related plots:
    - Pit depths (old/new) share linear scale, delta uses symmetric scale
    - Elevations (old/new vertex and cell) share linear scale
    - Delta elevations use symmetric scales around zero
    """
    from matplotlib import pyplot as plt
    import watershed_workflow.dynamic_colormaps

    if pit_kwargs is None: pit_kwargs = dict()
    if vertex_kwargs is None: vertex_kwargs = dict()
    if cell_kwargs is None: cell_kwargs = dict()

    if ax is None:
        fig, ax = plt.subplots(3, 3, figsize=(15, 9),  sharex=True, sharey=True)
    else:
        fig = ax[0, 0].figure

    # Convert sparse pit arrays to dense (full mesh size)
    old_pit_depths_dense = np.zeros(m2.num_cells, dtype=float)
    if len(old_pits) > 0:
        for p in old_pits:
            old_pit_depths_dense[p[0]] = _metricPitDepth(p)

    new_pit_depths_dense = np.zeros(m2.num_cells, dtype=float)
    if len(new_pits) > 0:
        for p in new_pits:
            new_pit_depths_dense[p[0]] = _metricPitDepth(p)

    # Extract vertex elevations
    new_z_verts = m2.coords[:, 2]
    dz_verts = new_z_verts - old_z_verts

    # Extract cell centroid elevations
    old_z_cells = np.array([
        watershed_workflow.utils.computeCentroid([old_z_verts[v] for v in m2.conn[c]])
        for c in range(m2.num_cells)])
    new_centroids = m2.centroids
    new_z_cells = new_centroids[:, 2]
    dz_cells = new_z_cells - old_z_cells

    # Create plots - vmin/vmax will be set by dynamic colormap scaler
    # Row 0 -- pit depths
    col1 = m2.plot(old_pit_depths_dense, ax=ax[0, 0],
            cmap='Reds', label='old pit depth [m]', alpha=1, edgecolors='none', **pit_kwargs)
    col2 = m2.plot(new_pit_depths_dense, ax=ax[0, 1],
            cmap='Reds', label='new pit depth [m]', alpha=1, edgecolors='none', **pit_kwargs)
    col3 = m2.plot(new_pit_depths_dense - old_pit_depths_dense, ax=ax[0, 2],
            cmap='RdBu_r', label='Δ pit depth [m]', alpha=1, edgecolors='none', **pit_kwargs)

    # Row 1 -- vertex elevations
    verts1 = m2.plotVertices(old_z_verts, ax=ax[1, 0],
                    cmap='gist_earth', label='old vertex elevation [masl]',  **vertex_kwargs)
    verts2 = m2.plotVertices(new_z_verts, ax=ax[1, 1],
                    cmap='gist_earth', label='new vertex elevation [masl]', **vertex_kwargs)
    verts3 = m2.plotVertices(dz_verts, ax=ax[1, 2],
                    cmap='RdBu_r', label='Δ vertex elevation [m]', **vertex_kwargs)

    # Row 2 -- cell elevations
    col4 = m2.plot(old_z_cells, ax=ax[2, 0],
            cmap='gist_earth', label='old cell elevation [masl]', edgecolors='none', **cell_kwargs)
    col5 = m2.plot(new_z_cells, ax=ax[2, 1],
            cmap='gist_earth', label='new cell elevation [masl]', edgecolors='none', **cell_kwargs)
    col6 = m2.plot(dz_cells, ax=ax[2, 2],
            cmap='RdBu_r', label='Δ cell elevation [m]', edgecolors='none', **cell_kwargs)

    # Add metrics text if provided
    title = f'Pit Filling: {method_name}\n'
    if metrics:
        metrics_text = []
        if 'pits_initial' in metrics:
            metrics_text.append(f"Pits: {metrics['pits_initial']} → {metrics.get('pits_final', '?')}")
        if 'pits_max_depth' in metrics:
            metrics_text.append(f"at max depth: {metrics['pits_max_depth']:.3f}")
        if 'rmse' in metrics:
            metrics_text.append(f"RMSE: {metrics['rmse']:.3f}")
        if 'mae' in metrics:
            metrics_text.append(f"MAE: {metrics['mae']:.3f}")
        if 'max' in metrics:
            metrics_text.append(f"Max Δz: {metrics['max']:.3f}")
        if 'num_modified' in metrics:
            metrics_text.append(f"Modified: {metrics['num_modified']} of {len(m2.coords)}")

        if metrics_text:
            title += '\n' + '  |  '.join(metrics_text)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    output = widgets.Output()
    scaler = watershed_workflow.dynamic_colormaps.DynamicColormapScaler(output=output)
    scaler.addGroup([col1, col2], False)
    scaler.addGroup([col3,], True)
    scaler.addGroup([verts1, verts2, col4, col5], False)
    scaler.addGroup([verts3,], True)
    scaler.addGroup([col6,], True)
    scaler.connect(ax[0,0])
    return output, fig, ax, scaler


def _partitionOutletEdges(m2 : Mesh2D,
                          forced_outlet_edges : Iterable[Edge] | None,
                          optional_outlet_edges : Iterable[Edge] | None,
                          divide_edges : Iterable[Edge] | None
                          ) -> Tuple[Set[Edge],
                                     Set[Edge],
                                     Set[Edge]]:
    """Process input and partition."""
    if forced_outlet_edges is None:
        forced_outlet_edges = set()
    else:
        forced_outlet_edges = set(Edge(e) for e in forced_outlet_edges)
    if optional_outlet_edges is None:
        optional_outlet_edges = set()
    else:
        optional_outlet_edges = set(Edge(e) for e in optional_outlet_edges)
    if divide_edges is None:
        divide_edges = set()
    else:
        divide_edges = set(Edge(e) for e in divide_edges)

    # make sure they are nonoverlapping, where forced_outlet_edges takes precedence
    divide_edges = divide_edges - forced_outlet_edges
    optional_outlet_edges = optional_outlet_edges - divide_edges - forced_outlet_edges
    
    # make sure all boundary edges appear somewhere
    for e in m2.boundary_edges:
        if e not in divide_edges and e not in forced_outlet_edges:
            optional_outlet_edges.add(e)
    return forced_outlet_edges, optional_outlet_edges, divide_edges


def _partitionOutletCells(m2: Mesh2D,
                          forced_outlet_edges : Set[Edge],
                          optional_outlet_edges : Set[Edge],
                          divide_edges : Set[Edge]
                          ) -> Tuple[Set[int],
                                     Set[int],
                                     Set[int]]:
    """Convert edges to cells."""
    def _toCell(e):
        cells = m2.edge_cells[e]
        assert len(cells) == 1
        return cells[0]

    forced_outlet_cells = set(_toCell(e) for e in forced_outlet_edges)
    optional_outlet_cells = set(_toCell(e) for e in optional_outlet_edges)
    divide_cells = set(_toCell(e) for e in divide_edges)

    # make sure they are nonoverlapping.  Note that here, divide
    # cannot take precedence because if it is both a divide cell and
    # an optional cell, the optional will win.
    optional_outlet_cells = optional_outlet_cells - forced_outlet_cells
    divide_cells = divide_cells - forced_outlet_cells - optional_outlet_cells
    return forced_outlet_cells, optional_outlet_cells, divide_cells


def computeChangeStatistics(original_vertex_elevations: np.ndarray,
                            new_vertex_elevations: np.ndarray,
                            metric: str = 'rmse') -> Union[float, Dict[str, float]]:
    """Quantify deviation between original and modified vertex elevations.

    Computes statistics describing how much vertex elevations changed,
    useful for assessing the impact of pit filling on the DEM.

    Parameters
    ----------
    original_vertex_elevations : np.ndarray
        Original vertex elevations, shape (num_vertices,).
    new_vertex_elevations : np.ndarray
        Modified vertex elevations, shape (num_vertices,).
    metric : str, optional
        Metric to compute. Options:

        - 'rmse': Root mean square error (default)
        - 'mae': Mean absolute error
        - 'max': Maximum absolute change (worst case)
        - 'total': Total (sum) of absolute changes
        - 'all': Returns dict with all metrics plus additional statistics

        Default is 'rmse'.

    Returns
    -------
    float or dict
        If metric != 'all', returns single float value in mesh coordinate units.

        If metric == 'all', returns dict with keys:

        - 'rmse': Root mean square error, measures overall goodness of
          fit; penalizes large changes.
        - 'mae': Mean absolute error across all vertices, a typical
          elevation change per vertex
        - 'max': Maximum (worst-case) elevation change
        - 'total': Sum of absolute elevation changes
        - 'num_modified': Count of vertices with elevation change > 1e-9
        - 'mean_modified': Mean change among modified vertices only
        - 'median_modified': Median change among modified vertices only
        - 'percent_modified': Percentage of vertices modified

    Notes
    -----
    All metrics are computed as (new - original), so positive values indicate
    vertices were raised. Pit filling should only raise elevations (or leave
    them unchanged), never lower them.

    """
    # Compute differences
    diffs = new_vertex_elevations - original_vertex_elevations
    abs_diffs = np.abs(diffs)

    # Identify modified vertices (change > tolerance)
    modified_mask = abs_diffs > 1e-9
    num_modified = np.sum(modified_mask)
    modified_diffs = abs_diffs[modified_mask]

    if metric == 'rmse':
        return np.sqrt(np.mean(diffs**2))

    elif metric == 'mae':
        return np.mean(abs_diffs)

    elif metric == 'max':
        return np.max(abs_diffs)

    elif metric == 'total':
        return np.sum(abs_diffs)

    elif metric == 'all':
        result = {
            'rmse': float(np.sqrt(np.mean(diffs**2))),
            'mae': float(np.mean(abs_diffs)),
            'max': float(np.max(abs_diffs)),
            'total': float(np.sum(abs_diffs)),
            'num_modified': int(num_modified),
            'percent_modified': float(100.0 * num_modified / len(diffs)),
        }

        # Compute statistics for modified vertices only
        if num_modified > 0:
            result['mean_modified'] = float(np.mean(modified_diffs))
            result['median_modified'] = float(np.median(modified_diffs))
        else:
            result['mean_modified'] = 0.0
            result['median_modified'] = 0.0

        return result

    else:
        raise ValueError(f"Invalid metric '{metric}'. Valid options are: 'rmse', 'mae', 'max', 'total', 'all'")


def raiseCellCentroid(m2: Mesh2D,
                      c: int,
                      target_raise: float,
                      fixed_vertices: Set[int]) -> Dict[int, float]:
    """Compute vertex raises to increase a cell's centroid by target amount.

    Raises only non-fixed vertices uniformly so that the cell's centroid
    elevation increases by exactly target_raise. The centroid is computed
    as the arithmetic mean of vertex coordinates.

    Parameters
    ----------
    m2 : Mesh2D
        The 2D mesh containing vertex coordinates and cell connectivity.
    c : int
        Cell index into m2.conn.
    target_raise : float
        Amount to raise the cell centroid elevation.
    fixed_vertices : Set[int]
        Set of vertex indices that cannot be raised (typically vertices
        on boundary edges that must remain at their current elevation).

    Returns
    -------
    vertex_raises : Dict[int, float]
        Dictionary mapping vertex_id -> raise_amount for free vertices only.
        Returns empty dict if no free vertices exist (all vertices fixed).

    Notes
    -----
    To raise centroid by target_raise with n_fixed fixed vertices:
    raise = target_raise * n_total / n_free

    This ensures that the mean of all vertex elevations increases by
    exactly target_raise when only the free vertices are modified.
    """
    vertex_raises = {}
    cell_vertices = list(m2.conn[c])

    free_vertices = [v for v in cell_vertices if v not in fixed_vertices]
    n_total = len(cell_vertices)
    n_free = len(free_vertices)

    if n_free == 0:
        # All vertices fixed - cannot raise
        logging.debug(f"Cell {c} cannot be raised (all {n_total} vertices "
                            f"are fixed), pit depth = "
                            f"{target_raise:.6f}")
        return vertex_raises  # Cannot raise, return empty

    raise_amount = target_raise * n_total / n_free

    for v in free_vertices:
        vertex_raises[v] = raise_amount

    return vertex_raises


def conditionCell(m2: Mesh2D,
                  c,
                  pit : Tuple[int | bool, str, float, float],
                  forced_outlet_edges: Set[Tuple[int, int]],
                  optional_outlet_edges: Set[Tuple[int, int]],
                  divide_edges: Set[Tuple[int, int]],
                  epsilon: float,
                  tol: float,
                  additional_fixed_vertices: Optional[Set[int]] = None,
                  relative_to : Optional[Iterable[int]] = None,
                  ) -> Dict[int, float]:
    """Compute vertex raises needed to condition a single pit cell.

    Determines which vertices should be fixed based on boundary edge
    categorization, then computes appropriate raises to eliminate the pit.

    Parameters
    ----------
    m2 : Mesh2D
        The 2D mesh containing vertex coordinates and cell connectivity.
    c : int
        Cell index into m2.conn.
    pit : Tuple[int or bool, str, float, float]
        Pit information tuple: (is_pit, cause, internal_depth, boundary_depth).
    forced_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that must be outlets.
    optional_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that may be outlets.
    divide_edges : Set[Tuple[int, int]]
        Boundary edges that must not be outlets (watershed divides).
    epsilon : float
        Minimum elevation increase to enforce drainage.
    tol : float
        Numerical tolerance for roundoff errors.
    additional_fixed_vertices : Set[int] or None, optional
        Additional vertices that must remain fixed beyond those determined
        by boundary edge logic. Default is None (no additional fixed vertices).
    relative_to : Iterable[int] or None, optional
        If provided, only consider these cells when measuring pits.
        Default is None (consider all neighbors).

    Returns
    -------
    vertex_raises : Dict[int, float]
        Dictionary mapping vertex_id -> raise_amount. Returns empty dict
        if cell is already conditioned.

    Notes
    -----
    Logic by boundary edge type:

    - Internal cells: Raise centroid by internal_depth + epsilon
    - Forced outlet edges: Fix vertices on forced edges, raise centroid
    - Divide edges with boundary_depth > 0: Raise only boundary vertices
    - Divide edges with boundary_depth <= 0: Raise centroid by internal_depth
    - Optional outlet edges: Fix vertices on optional edges, raise centroid

    Additional fixed vertices are respected in all cases.
    """
    if additional_fixed_vertices is None:
        additional_fixed_vertices = set()

    # Get pit depths for this cell
    _, cause, internal_depth, boundary_depth = pit

    logging.debug(f"Conditioning cell {c} ({cause} pit) with internal_depth {internal_depth:.5f} "
                  f"and boundary depth {boundary_depth:.5f}")

    if cause == 'internal':
        # Internal cell - no fixed vertices, raise centroid
        logging.debug(f" ... internal")
        vertex_raises = raiseCellCentroid(m2, c, internal_depth + epsilon,
                                fixed_vertices=additional_fixed_vertices)

    else:
        # Boundary cell - determine fixed vertices and raises
        cell_vertices = set(m2.conn[c])
        cell_edges = set(m2.cell_edges[c])
        boundary_cell_edges = cell_edges & set(m2.boundary_edges)

        if cause == 'forced outlet':
            logging.debug(f" ... forced -- raising internally")
            # Must drain outward - fix vertices on forced_outlet_edges, raise centroid
            fixed_vertices = set()
            for edge in boundary_cell_edges:
                if edge in forced_outlet_edges:
                    fixed_vertices.update(edge)
            fixed_vertices.update(additional_fixed_vertices)
            vertex_raises = raiseCellCentroid(m2, c, boundary_depth + epsilon, fixed_vertices)

        elif cause == 'divide boundary':
            logging.debug(f" ... divide -- raising boundary")
            # Water flowing outward - raise ONLY boundary vertices directly
            vertex_raises = {}
            raise_amount = -boundary_depth + epsilon
            for v in cell_vertices:
                if v not in additional_fixed_vertices:
                    if any(v in edge for edge in divide_edges if edge in cell_edges):
                        vertex_raises[v] = raise_amount

        elif cause == 'divide internal':
            # Internally trapped - fix nothing, raise centroid
            logging.debug(f" ... divide -- raising all")
            vertex_raises = raiseCellCentroid(m2, c, internal_depth + epsilon,
                                    fixed_vertices=additional_fixed_vertices)

        elif cause == 'boundary internal':
            logging.debug(f" ... optional -- raising all")
            # May drain either way - fix vertices on optional_outlet_edges, raise centroid
            fixed_vertices = set()
            for edge in optional_outlet_edges:
                if edge in cell_edges:
                    fixed_vertices.update(edge)
            fixed_vertices.update(additional_fixed_vertices)
            vertex_raises = raiseCellCentroid(m2, c, boundary_depth + epsilon, fixed_vertices)

        else:
            raise ValueError('conditionCell() called on non-pit cell or unrecognized cause.')

    logging.debug(f"      {vertex_raises}")
    return vertex_raises


def _fillPits_iterator_decorator(func, method_name):
    def iterated_func(m2: Mesh2D,
                      pits : List[Tuple[int,str,float,float]],
                      preserved_pits: Set[int],
                      forced_outlet_edges : Set[Edge],
                      optional_outlet_edges : Set[Edge],
                      divide_edges : Set[Edge],
                      epsilon: float,
                      tol: float,
                      max_iterations: int = 1000,
                      increase_ok : bool = False,
                      **kwargs,
                      ) -> Mesh2D:
        for itr in range(max_iterations):
            n_pits = len(pits)
            if n_pits == 0:
                logging.info(f" ... done iterating in {itr} iterations, 0 pits.")
                return m2

            m2 = func(m2, pits, preserved_pits, forced_outlet_edges, optional_outlet_edges, divide_edges,
                 epsilon, tol, **kwargs)
            pits_final = _findPits(m2, preserved_pits, forced_outlet_edges, optional_outlet_edges, divide_edges,
                                epsilon, tol)
            if max_iterations < 10 or itr%10 == 0:
                logging.info(f" ... iteration {itr} of {method_name}: {len(pits_final)} pits")

            if not increase_ok and len(pits_final) > n_pits:
                logging.info(f" ... done iterating in {itr} iterations due to increasing number of pits.")
                return m2

            pits = pits_final

        logging.info(f" ... done iterating, max iterations.")
        return m2

    return iterated_func


def fillPits_global(m2: Mesh2D,
                    pits : List[Tuple[int,str,float,float]],
                    preserved_pits: Set[int],
                    forced_outlet_edges : Set[Edge],
                    optional_outlet_edges : Set[Edge],
                    divide_edges : Set[Edge],
                    epsilon: float,
                    tol: float,
                    boundary_only: bool = False
                    ) -> Mesh2D:
    """Fill pits in the mesh using iterative cell-based algorithm.

    This algorithm iteratively identifies and eliminates pits by raising
    vertex elevations using boundary edge categorization to determine which
    vertices should be fixed. Guarantees no pits remain (within tolerance)
    and only raises elevations, never lowering them.

    Modifies m2.coords[:, 2] in place.

    Parameters
    ----------
    m2 : Mesh2D
        The 2D mesh containing vertex coordinates and cell connectivity.
        Elevations in m2.coords[:, 2] will be modified in place.
    pits : List[Tuple[int, str, float, float]]
        List of pits to fill: (cell, cause, internal_depth, boundary_depth).
    preserved_pits : Set[int]
        Cell indices that should be preserved as pits (e.g., lakes,
        playas, or other real depressions). These cells are never filled.
    forced_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that must be outlets. Cells touching these edges
        will have these edge vertices fixed while raising the cell centroid.
    optional_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that may be outlets.
    divide_edges : Set[Tuple[int, int]]
        Boundary edges that must not be outlets (watershed divides). Cells
        touching these may have divide edge vertices raised to prevent
        outward flow.
    epsilon : float
        Minimum elevation increase per cell in flow direction. Units are
        same as mesh coordinates.
    tol : float
        Numerical tolerance for roundoff errors.
    boundary_only : bool, optional
        If True, only process boundary pits (pits with cause != 'internal').
        If False, process all pits. Default is False.

    Notes
    -----
    Algorithm iteratively:

    1. Identifies pit cells using findPits() with boundary edge categorization
    2. For each pit, calls computePitDepth() to get internal and boundary depths
    3. Calls conditionCell() to determine which vertices to raise based on
       boundary edge type (forced outlet, divide, or optional outlet)
    4. Accumulates vertex raises (taking max for vertices shared by multiple cells)
    5. Applies vertex elevation changes to mesh
    6. Repeats until no changes exceed tolerance or max_iterations reached

    Boundary edge categorization ensures:
    - Forced outlet edges: Interior vertices raised, boundary vertices fixed
    - Divide edges: Boundary vertices raised to prevent outward flow
    - Optional outlet edges: Interior vertices raised, boundary vertices fixed

    Elevations are only raised, never lowered.
    """
    args_edges = [forced_outlet_edges, optional_outlet_edges, divide_edges,
                  epsilon, tol]

    # parse input
    forced_outlet_cells, optional_outlet_cells, divide_cells = \
        _partitionOutletCells(m2, forced_outlet_edges, optional_outlet_edges, divide_edges)

    # Filter pits if boundary_only
    if boundary_only:
        pits = [pit for pit in pits if pit[1] != 'internal']

    # Compute vertex raises using conditionCell for each pit
    vertex_raises = {}  # Dict to accumulate raises
    for pit in pits:
        # Use conditionCell to determine proper vertex raises based on boundary edge type
        cell_raises = conditionCell(m2, pit[0], pit, *args_edges,
                                   additional_fixed_vertices=None)

        # Accumulate raises (take max for shared vertices)
        for v, raise_amount in cell_raises.items():
            vertex_raises[v] = max(vertex_raises.get(v, 0), raise_amount)

    # Apply raises to mesh
    for v, raise_amount in vertex_raises.items():
        m2.coords[v, 2] += raise_amount

    # Compute convergence metrics
    num_vertices_changed = len(vertex_raises)
    max_change = 0.0 if num_vertices_changed == 0 else max(vertex_raises.values())
    return m2
    

fillPits_global_iterative = _fillPits_iterator_decorator(fillPits_global, "global")
    
    
def fillPits_marching_old(m2: Mesh2D,
                          pits : List[Tuple[int,str,float,float]],
                          preserved_pits: Set[int],
                          forced_outlet_edges : Set[Edge],
                          optional_outlet_edges : Set[Edge],
                          divide_edges : Set[Edge],
                          epsilon: float,
                          tol: float,
                          ) -> Mesh2D:
    """Fill pits using a greedy marching algorithm.

    The goal of this algorithm is to ensure that, starting with an
    outlet cell and a list of known pits, there is a path to every
    cell by way of faces that is monotonically increasing in
    elevation.

    A cell is called reachable if such a path exists.  Cells are
    incrementally added to the "waterway," or the set of cells with
    identified paths.

    Starting from an outlet, it adds cells to the waterway by picking
    the lowest elevation cell that currently borders the existing
    waterway.  It conditions upon adding the cell to the _boundary_.

    Conditioning a cell requires that this cell is higher that at
    least one of its neighbors that is _already in the waterway_.
    Note that this is more aggressive because it requires the lower
    cell to be in the waterway, not just any neighbor.

    If a cell has a lower cell that is NOT in the waterway, and has a
    valid, monotonically decreasing pathway to the waterway, then that
    pathway has a cell that _is_ in the boundary of the current
    waterway, and that cell's elevation is lower than this cell, and
    therefore would have been selected before this cell.
    Contradiction; therefore the downhill path must lead to a pit or
    the boundary, but not to the waterway.

    Parameters
    ----------
    m2 : Mesh2D
        The mesh to condition.
    pits : List[Tuple[int, str, float, float]]
        List of pits to fill: (cell, cause, internal_depth, boundary_depth).
    preserved_pits : Set[int]
        Cell indices to preserve as pits (e.g., lakes) that may be a depression.
        This is important for reservoirs/lakes/etc where bathymetry is known and
        pits are physical.
    forced_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that are forced outlets.
    optional_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that may be outlets.
    divide_edges : Set[Tuple[int, int]]
        Boundary edges that are watershed divides.
    epsilon : float
        Minimum slope parameter.
    tol : float
        Numerical tolerance for roundoff errors.
    """
    class Waterway:
        """Waterway is the set of cells that are already conditioned and can be reached."""
        def __init__(self):
            self.cells = set()

            # Waterway edges is the set of edges whose cells are all in waterway
            self.edges = set()

        def add(self, be):
            """Add BoundaryEntry object to the waterway"""
            logging.debug(f"adding cell {be.cell} (z = {be.z}) to the waterway")
            self.cells.add(be.cell)
            for e in be.edges:
                self.edges.add(e)

    waterway = Waterway()

    class BoundaryEntry:
        """A cell that is not yet in the waterway, but has at least one edge whose other cell is in the waterway."""
        def __init__(self, cell, edges):
            assert type(cell) is int
            assert 0 <= cell < m2.num_cells
            assert type(edges) is list
            for e in edges:
                assert isinstance(e, Edge)

            self.cell = cell
            self.edges = edges
            self.z = m2.computeCentroid(self.cell)[2]

    # Seed boundary with all outlet cells (maintaining cell-edge correspondence)
    boundary_entries = []
    for edge in forced_outlet_edges:
        cells = m2.edge_cells[edge]
        assert len(cells) == 1
        boundary_entries.append(BoundaryEntry(cells[0], [edge,]))
    boundary = sortedcontainers.SortedList(boundary_entries, key=lambda be: be.z)

    # preserved cells are always in the boundary, allowing them to be picked up as we reach that elevation.
    if preserved_pits:
        masked_cells = [BoundaryEntry(c, list()) for c in preserved_pits]
        boundary.update(masked_cells)

    while len(boundary) > 0:
        # pop the lowest boundary cell and stick its edge and cell
        next_be = boundary.pop(0)
        waterway.add(next_be)

        # find all other edges of the cell just added
        for other_e in m2.cell_edges[next_be.cell]:
            if other_e in waterway.edges: continue

            # find the cell on the other side of other_e
            other_e_cells = m2.edge_cells[other_e]
            if len(other_e_cells) == 1:
                # boundary edge, add it to the waterway
                assert next_be.cell == other_e_cells[0]
                waterway.edges.add(other_e)
                continue

            assert len(other_e_cells) == 2, \
                f"Edge {other_e} has {len(other_e_cells)} cells, expected 2"
            assert next_be.cell in other_e_cells, \
                f"Cell {next_be.cell} not in edge {other_e} cells: {other_e_cells}"
            if next_be.cell == other_e_cells[0]:
                other_c = other_e_cells[1]
            else:
                assert (next_be.cell == other_e_cells[1])
                other_c = other_e_cells[0]

            # this would break assumption of what it means to be
            # in boundary.
            assert other_c not in waterway.cells, \
                f"Cell {other_c} already in waterway but being added to boundary"

            # now we have an other_e, other_c pair to add into
            # boundary.  But first we may need to condition.
            other_c_centroid = m2.computeCentroid(other_c)

            # Find waterway neighbors and compute local max elevation
            waterway_neighbors = [n for n in m2.cell_to_cells[other_c] if n in waterway.cells]
            assert len(waterway_neighbors) > 0, "Cell in boundary must have waterway neighbors"
            max_neighbor_elev = max(m2.computeCentroid(n)[2] for n in waterway_neighbors)
            target_elev = max_neighbor_elev + epsilon

            # Only condition if below target; otherwise cell is already high enough
            if other_c_centroid[2] < target_elev:
                other_c_vertices = m2.conn[other_c]

                # for this to be possible, there must be at least
                # one free vertex in the vertices of next_c.  By free,
                # we mean that its elevation can be changed
                # without breaking everything.  This means that
                # neither of that vertex's edges can be in
                # waterway.edges or boundary.
                #
                # we also need the fixed (non-free) vertex elevations
                fixed_vertex_elevs = dict()

                for e in m2.cell_edges[other_c]:
                    if (e == other_e) or (e in waterway.edges) or any(
                        (e == i) for be in boundary for i in be.edges):
                        if e[0] not in fixed_vertex_elevs:
                            fixed_vertex_elevs[e[0]] = m2.coords[e[0], 2]
                        if e[1] not in fixed_vertex_elevs:
                            fixed_vertex_elevs[e[1]] = m2.coords[e[1], 2]
                free_vertices = [n for n in other_c_vertices if n not in fixed_vertex_elevs]

                # should not be possible to be both lower
                # elevation and not have a free vertex, or it would
                # already be in boundary, and therefore have no
                # free vertices

                # If no free vertices, cannot raise - log and skip conditioning
                if len(free_vertices) == 0:
                    logging.debug(f"Cell {other_c} has no free vertices but needs conditioning "
                                f"(target={target_elev:.3f}, current={other_c_centroid[2]:.3f}), "
                                f"leaving as pit")
                else:
                    # calculate the z of the free vertices required to
                    # make the cell's centroid == target_elev
                    # Want: (sum_fixed + n_free * z_free) / n_total = target_elev
                    z_free = (target_elev * len(other_c_vertices)
                              - sum(fixed_vertex_elevs.values())) / len(free_vertices)

                    # Raise all free vertices to z_free (only raise, never lower)
                    for v in free_vertices:
                        vertex_current_elev = m2.coords[v, 2]
                        if z_free > vertex_current_elev:
                            logging.debug(
                                f'  moving vertex {v} from {vertex_current_elev:.6f} to {z_free:.6f}'
                            )
                            m2.coords[v, 2] = z_free

                    logging.debug(f"Cell {other_c} increased {len(free_vertices)} free vertices to {z_free}")

            # now add it to the boundary (whether conditioned or not)
            try:
                # is it already in the boundary?
                other_be = next(be for be in boundary if be.cell == other_c)
            except StopIteration:
                # no, add it
                # logging.debug(f'  adding to boundary: edge: {other_e}  cell: {other_c}')
                boundary.add(BoundaryEntry(other_c, [other_e, ]))
            else:
                # yes, just add this edge to that entry
                if other_e not in other_be.edges:
                    other_be.edges.append(other_e)

    # when this is done, all cells should be in waterway
    assert len(waterway.cells) == m2.num_cells, \
        f"Not all cells processed: {len(waterway.cells)}/{m2.num_cells} in waterway"
    assert len(waterway.edges) == m2.num_edges, \
        f"Not all edges processed: {len(waterway.edges)}/{m2.num_edges} in waterway"

    # delete the centroid info to force recalculation
    m2.clearGeometryCache()
    return m2


fillPits_marching_old_iterative = _fillPits_iterator_decorator(fillPits_marching_old, "marching old")


def fillPits_marching(m2: Mesh2D,
                      pits : List[Tuple[int,str,float,float]],
                      preserved_pits: Set[int],
                      forced_outlet_edges : Set[Edge],
                      optional_outlet_edges : Set[Edge],
                      divide_edges : Set[Edge],
                      epsilon: float,
                      tol: float,
                      seed_policy : Optional[str | List[str]] = None,
                      seed_to : Optional[str] = 'waterway',
                      conditioning_policy : str = 'waterway',
                      fixing_policy : str = 'waterway',
                      preserved_pits_are_fixed : bool = True,
                      replace_upon_conditioning : bool = True,
                      ) -> Mesh2D:
    """Fill pits using a greedy marching algorithm.

    The goal of this algorithm is to ensure that, starting from an
    outlet cell and a list of known pits, there is a path to every
    cell by way of faces that are monotonically increasing in
    elevation.

    A cell is called reachable if such a path exists.  Cells are
    incrementally added to the "waterway," or the set of cells with
    identified, unchanging paths.

    This algorithm starts with all preserved pits and outlets.  These
    cells are fixed and placed in the waterway.  Cells are added to
    the waterway by picking the lowest elevation cell that currently
    borders the existing waterway.

    A cell must be conditioned before it can be added to the waterway.
    Conditioning a cell enforces that this cell has a higher elevation
    than at least one of its neighbors that is _already in the
    waterway_.

    If a cell has a lower cell that is NOT in the waterway, and has a
    valid, monotonically decreasing pathway to the waterway, then that
    pathway has a cell that _is_ in the border of the current
    waterway, and that cell's elevation is lower than this cell, and
    therefore would have been selected before this cell.
    Contradiction; therefore the downhill path must lead to a pit or
    the boundary, but not to the waterway.  This observation is key to
    enforcing the condition.

    To ensure that conditioning one cell does not break other, already
    conditioned cells, we must choose a set of vertices to fix.
    Clearly all cells in the waterway should have their vertices
    fixed; these are not revisited.  The obvious approach is to fix
    vertices upon conditioning -- then once a cell is conditioned, it
    need not be revisited.

    The only known failure mechanism of this approach is that a cell
    cannot be conditioned because the vertices of that cell are all a
    part of previously conditioned cells and therefore have been
    fixed.  In that case the cell is added anyway and left
    unconditioned (becoming a pit).

    Note there are lot of options for the algorithm, but the defaults
    for all policies are expected to be the most robust.

    Parameters
    ----------
    m2 : Mesh2D
        The mesh to condition.
    pits : List[Tuple[int, str, float, float]]
        List of pits to fill: (cell, cause, internal_depth, boundary_depth).
    preserved_pits : Set[int]
        Cell indices to preserve as pits (e.g., lakes) that may be a depression.
        This is important for reservoirs/lakes/etc where bathymetry is known and
        pits are physical.
    forced_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that are forced outlets.
    optional_outlet_edges : Set[Tuple[int, int]]
        Boundary edges that may be outlets.
    divide_edges : Set[Tuple[int, int]]
        Boundary edges that are watershed divides.
    epsilon : float
        Minimum slope parameter.
    tol : float
        Tolerance for numeric roundoff.
    seed_policy : str or List[str], optional
        What elements are included in the initial seed? Default is
        ['forced outlets', 'preserved pits', 'optional outlets']. Valid entries
        include these values.
    seed_to : str, optional
        Where are seeded elements put, into the 'waterway' or into the
        'border'. Default is 'waterway'.
    conditioning_policy : str, optional
        When to condition -- upon entering 'waterway' or 'border'.
        Default is 'waterway'.
    fixing_policy : str, optional
        When to fix vertices -- upon entering 'waterway' or 'border'.
        Default is 'waterway'.
    preserved_pits_are_fixed : bool, optional
        If True, fixes all vertices of preserved pits. Default is True.
    replace_upon_conditioning : bool, optional
        If cells are conditioned on placement in the waterway,
        conditioning may raise the cell elevation, meaning it is no
        longer the lowest elevation cell in the border. If this is
        True, conditioned cells are placed back into the border and not
        put in the waterway. Default is True.
    """
    args_edges = [forced_outlet_edges, optional_outlet_edges, divide_edges, epsilon, tol]

    # parse input
    forced_outlet_cells, optional_outlet_cells, divide_cells = \
        _partitionOutletCells(m2, forced_outlet_edges, optional_outlet_edges, divide_edges)
    args_cells = [preserved_pits, forced_outlet_cells, optional_outlet_cells, divide_cells, epsilon, tol]

    if seed_policy is None:
        seed_policy = ['forced outlets', 'preserved pits', 'optional outlets']
    elif isinstance(seed_policy, str):
        seed_policy = [seed_policy,]

    if fixing_policy not in ['none', 'waterway', 'border']:
        raise ValueError(f'Invalid fixing_policy "{fixing_policy}", must be one of "none", "waterway", or "border."')

    logging.debug('Seeding marching')
    logging.debug('---------------------------')
    
    waterway = set()  # set of cells that are known reachable
    fixed_vertices = set()  # Vertices in waterway cells (incrementally updated)
    border = sortedcontainers.SortedList(key=lambda entry : entry[0])

    # pre-fix preserved pits so they cannot change when conditioning
    if preserved_pits_are_fixed:
        for c in preserved_pits:
            fixed_vertices.update(m2.conn[c])
    
    
    def measureAndConditionCell(c, rel_to_waterway=True) -> bool:
        """Returns whether the elevation was raised or not."""
        raised = False
        if rel_to_waterway:
            pit = _measurePit(m2, c, *args_cells, relative_to=waterway)
        else:
            pit = _measurePit(m2, c, *args_cells)

        if pit[0]:
            # condition c
            to_raise = conditionCell(m2, c, pit, *args_edges,
                                     additional_fixed_vertices=fixed_vertices,
                                     relative_to=waterway)

            if to_raise:
                # raise all coordinates
                for v, val in to_raise.items():
                    m2.coords[v,2] += val

                # recompute elevation of affected cells in the border
                to_replace = []
                for i,(z,bc) in enumerate(border): # make a copy
                    if any(v in m2.conn[bc] for v in to_raise.keys()):
                        to_replace.append(i)

                for i in reversed(to_replace):
                    (z, bc) = border.pop(i)
                    z_new = m2.computeCentroid(bc)[2]
                    border.add((z_new, bc))
                    logging.debug(f'REPLACING bc {bc}, recomputed elev from {z} to {z_new}')
                raised = True
        return raised

    def addToBorder(c):
        if c in waterway:
            return
        if any(c == entry[1] for entry in border):
            return
        
        # condition upon adding
        if conditioning_policy == 'border':
            measureAndConditionCell(c, True)

        border.add((m2.computeCentroid(c)[2], c))

        if fixing_policy == 'border':
            if c not in preserved_pits:
                fixed_vertices.update(m2.conn[c])


    def addToWaterway(c, replace_upon_conditioning2=False) -> bool:
        logging.debug(f'Adding to waterway {c} at {m2.computeCentroid(c)}')

        # condition upon adding
        if conditioning_policy == 'waterway':
            conditioned = measureAndConditionCell(c, True)

            if conditioned and replace_upon_conditioning2:
                # stick back in the border, it is no longer lowest elevation!
                addToBorder(c)
                return False

        waterway.add(c)

        # fix vertices
        if fixing_policy == 'waterway':
            if c not in preserved_pits:
                fixed_vertices.update(m2.conn[c])
        return True

    # the seed is the initial waterway.  These are a valid sinks of
    # water.
    for sp in seed_policy:
        if sp == 'forced outlets':
            for c in forced_outlet_cells:
                # condition to make sure they are outlets
                if seed_to == 'waterway':
                    if conditioning_policy == 'border':
                        measureAndConditionCell(c, False)
                    addToWaterway(c)
                else:
                    addToBorder(c)

        elif sp == 'optional outlets':
            for c in optional_outlet_cells:
                # condition to make sure they are outlets
                if seed_to == 'waterway':
                    if conditioning_policy == 'border':
                        measureAndConditionCell(c, False)
                    addToWaterway(c)
                else:
                    addToBorder(c)

        elif sp == 'preserved pits':
            for c in preserved_pits:
                # no conditioning of preserved pits
                if seed_to == 'waterway':
                    addToWaterway(c)
                else:
                    addToBorder(c)
        else:
            raise ValueError(f"Invalid seed policy: '{sp}'")

    # now that all seeds are in the waterway, we can add all their
    # neighbors to the border, conditioning as we go
    logging.debug('Setting up border')
    logging.debug('---------------------------')
    for c in waterway:
        for nc in m2.cell_to_cells[c]:
            addToBorder(nc)

    logging.debug('Marching')
    logging.debug('---------------------------')
    # start marching
    while len(border) > 0:
        # pop the lowest border cell and add it to the waterway
        (z,bc) = border.pop(0)
        added = addToWaterway(bc, replace_upon_conditioning)
        if added:
            for nc in m2.cell_to_cells[bc]:
                addToBorder(nc)

    return m2


fillPits_marching_iterative = _fillPits_iterator_decorator(fillPits_marching, "marching")


def fillPits(m2: Mesh2D,
             method_name: str = 'marching',
             preserved_pits: Optional[Iterable[int]] = None,
             forced_outlet_edges : Optional[Iterable[Edge]] = None,
             optional_outlet_edges : Optional[Iterable[Edge]] = None,
             divide_edges : Optional[Iterable[Edge]] = None,
             epsilon: float = 0.0,
             tol : float = 1.e-8,
             plot: bool = False,
             max_iterations : Optional[int] = None,
             **kwargs,
             ) -> Tuple[Mesh2D, Dict[str, Any]]:
    """Fill pits in mesh using specified method(s) with optional plotting and metrics.

    User-friendly wrapper that runs pit filling method(s), computes
    metrics, and optionally plots results. Note that the returned
    Mesh2D may be the input mesh, or may be different, depending upon
    the algorithm.  The user should assume that m2 is modified in
    place, but should use the returned mesh.

    Parameters
    ----------
    m2 : Mesh2D
        The mesh to condition (modified in place).
    method_name : str, optional
        Method to use. Can be: 'recommended', 'global', 'marching', 'marching old',
        'null', or 'boundary cleanup'. Default is 'recommended'.
    preserved_pits : Iterable[int], optional
        Cell indices to preserve as pits (e.g., lakes). Default is the empty list.
    forced_outlet_edges : Iterable[Tuple[int, int]], optional
        Boundary edges that must be outlets. Default is the empty list.
    optional_outlet_edges : Iterable[Tuple[int, int]], optional
        Boundary edges that may be outlets. Defaults to all boundary edges not in
        forced_outlet_edges or divide_edges.
    divide_edges : Iterable[Tuple[int, int]], optional
        Boundary edges that must not be outlets (watershed divides).
        Default is the empty list.
    epsilon : float, optional
        Minimum slope parameter. Default is 0.0 (no enforced slope).
    tol : float, optional
        Numerical tolerance for roundoff errors. Default is 1.e-8.
    plot : bool, optional
        If True, creates before/after elevation comparison plots. Default is False.
    max_iterations : int, optional
        Maximum iterations for iterative methods. If specified, uses iterative
        version of the method. Default is None (use non-iterative version).
    kwargs : dict, optional
        Additional keyword arguments passed to fillPits algorithm.

    Returns
    -------
    m2 : Mesh2D
        The conditioned mesh.
    result : Dict[str, Any]
        Statistics dictionary with keys:
        - 'method_name': Name of method used
        - 'pits_initial': List of initial pits
        - 'pits_final': List of remaining pits
        - 'pits_removed': Number of pits removed
        - 'elevation_stats': Dict with 'rmse', 'mae', 'max', 'num_modified', etc.

    """
    preserved_pits = set(preserved_pits) if preserved_pits else set()

    # set defaults, standardize input
    forced_outlet_edges, optional_outlet_edges, divide_edges = \
        _partitionOutletEdges(m2, forced_outlet_edges, optional_outlet_edges, divide_edges)
    args_edges = [preserved_pits, forced_outlet_edges, optional_outlet_edges, divide_edges,
                  epsilon, tol]

    # outlet edges --> cells
    forced_outlet_cells, optional_outlet_cells, divide_cells = \
        _partitionOutletCells(m2, forced_outlet_edges, optional_outlet_edges, divide_edges)
    args_cells = [preserved_pits, forced_outlet_cells, optional_outlet_cells, divide_cells, epsilon, tol]

    
    # Map of method_name names to functions
    if max_iterations is not None:
        # pick an iterative method
        if method_name == 'null':
            method = None
        elif method_name == 'global':
            method = fillPits_global_iterative
        elif method_name == 'marching':
            method = fillPits_marching_iterative
        elif method_name == 'marching old':
            method = fillPits_marching_old_iterative
        else:
            raise ValueError(f'Unrecognized method {method_name}')
        method_name = method_name+"_iterative"
    else:
        # pick a non-iterative method
        if method_name == 'null':
            method = None
        elif method_name == 'global':
            method = fillPits_global
        elif method_name == 'marching':
            method = fillPits_marching
        elif method_name == 'marching old':
            method = fillPits_marching_old
        elif method_name == 'boundary cleanup':
            method = fillPits_boundary_cleanup
        else:
            raise ValueError(f'Unrecognized method {method_name}')
        

    # Store state before running the first algorithm
    pits_initial = _findPits(m2, *args_cells)
    elevs_initial = m2.coords[:,2].copy()

    logging.info("")
    logging.info(f"Running {method_name}: {len(pits_initial)} initial pits")
    logging.info("==============================================================================")

    # Run method with standard calling convention (unless null)
    if method is None:
        pits_final = pits_initial
    elif max_iterations is None:
        m2 = method(m2, pits_initial, *args_edges, **kwargs)
        pits_final = _findPits(m2, *args_cells)
    else:
        m2 = method(m2, pits_initial, *args_edges, max_iterations=max_iterations, **kwargs)
        pits_final = _findPits(m2, *args_cells)
    m2.clearGeometryCache()

    # Compute elevation change statistics
    elev_stats = computeChangeStatistics(elevs_initial, m2.coords[:,2], metric='all')

    # Store results
    result = {
        'method_name' : method_name,
        'pits_initial' : pits_initial,
        'pits_final': pits_final,
        'pits_removed': len(pits_initial) - len(pits_final),
        'elevation_stats': elev_stats,
    }

    pits_classified = collections.defaultdict(list)
    for p in pits_final:
        pits_classified[p[1]].append(p)    
    
    logging.info(f"  completed: {len(pits_final)} final pits, {result['pits_removed']} removed")
    for pit_type, pits_of_type in pits_classified.items():
        pits_of_type_depths = np.array([singlePitDepth(p) for p in pits_of_type])
        logging.info(f"    - {len(pits_of_type)} '{pit_type}' pits with max depth "
                     f"{max(pits_of_type_depths)} and median {np.median(pits_of_type_depths)}")

    logging.info(f"  RMSE of dz: {elev_stats['rmse']}")
    logging.info(f"  MAE of dz: {elev_stats['mae']}")
    logging.info(f"  MAX of dz: {elev_stats['max']}")
    
    # Plot if requested
    if plot:
        plot_metrics = {
            'pits_initial': len(pits_initial),
            'pits_final': len(pits_final),
            'pits_max_depth': max(_metricPitDepth(p) for p in pits_final) if pits_final else 0.,
            }
        plot_metrics.update(elev_stats)
        output, fig, ax, scaler = plotPitFilling(m2, pits_initial, pits_final,
                                         elevs_initial, method_name, metrics=plot_metrics)
        result['output'] = output
        result['fig'] = fig
        result['ax'] = ax
        result['scaler'] = scaler

    return m2, result


def conditionMesh(m2 : Mesh2D,
                  preserved_pits: Optional[Iterable[int]] = None,
                  forced_outlet_edges : Optional[Iterable[Edge]] = None,
                  optional_outlet_edges : Optional[Iterable[Edge]] = None,
                  divide_edges : Optional[Iterable[Edge]] = None,
                  epsilon: float = 0.0,
                  tol : float = 1.e-8,
                  plot: bool = False,
                  ) -> Tuple[Mesh2D, List[Dict[str,Any]]]:
    """The recommended algorithm for filling pits away from the river."""
    args = [preserved_pits, forced_outlet_edges, optional_outlet_edges,
            divide_edges, epsilon, tol]
    m2, res1 = fillPits(m2, 'marching', *args, max_iterations=3)
    pits = res1['pits_final']

    # find all triangular, interior pits with nonoverlapping neighbors
    nonoverlapping = []
    affected = []
    for p in pits:
        if p[1] == 'internal':
            c = p[0]
            if len(m2.conn[c]) == 3 and \
               all(len(m2.conn[n]) == 3 for n in m2.cell_to_cells[c]):
                # refine the triangle
                local_affected = [p[0],] + m2.cell_to_cells[p[0]]
                if not any(la in affected for la in local_affected):
                    nonoverlapping.append(p[0])
                    affected.extend(local_affected)    
    # refine
    m2_r, removed_cells = watershed_workflow.mesh.refineTriangles(m2, nonoverlapping)

    # remap preserved_pits.  Note that edges are fine
    #
    # removed_cells provides a list of cells, in the original numbering, that were removed
    preserved_pits_old = list(preserved_pits)
    preserved_pits_new = []
    for c in preserved_pits_old:
        # count the number of cells in removed_cells that have an id less than preserved_pits
        new_c = c - sum(1 for r in removed_cells if r < c)
        assert m2.conn[c] == m2_r.conn[new_c]
        preserved_pits_new.append(new_c)

    args = [set(preserved_pits_new), forced_outlet_edges, optional_outlet_edges,
            divide_edges, epsilon, tol]

    # marching
    m2_r, res2 = fillPits(m2_r, 'marching', *args, max_iterations=3)

    # global iterative
    m2_r, res3 = fillPits(m2_r, 'global', *args, max_iterations=200, increase_ok=True)
    return m2_r, [res1, res2, res3]


def conditionRiverMeshes(m2 : Mesh2D,
                         rivers : List[River],
                         *args, **kwargs) -> None:
    """For multiple rivers, condition, IN PLACE, the elevations of
    stream-corridor elements to ensure connectivity throgh culverts,
    skips ponds, maintain monotonicity, or otherwise enforce depths of
    constructed channels.
    """
    for river in rivers:
        conditionRiverMesh(m2, river, *args, **kwargs)

        
def conditionRiverMesh(m2 : Mesh2D,
                       river : River,
                       smooth : bool = False,
                       lower : bool = False,
                       bank_integrity_elevation : float = 0.0,
                       depress_headwaters_by : Optional[float] = None,
                       network_burn_in_depth : Optional[Callable[[River,], float]] = None,
                       known_depressions : Optional[List[int]] = None) -> None:
    """Condition, IN PLACE, the elevations of stream-corridor elements
   to ensure connectivity throgh culverts, skips ponds, maintain
   monotonicity, or otherwise enforce depths of constructed channels.

    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
        2D mesh with 3D coordinates.
    river: watershed_workflow.river_tree.River object
        River tree with reach['elems'] added for quads
    smooth: boolean, optional
        If true, smooth the profile of each reach using a gaussian
        filter (mainly to pass through railroads and avoid reservoirs).
    lower: boolean, optional
        If true, lower the smoothed bed profile to match the lower
        points on the raw bed profile. This is useful particularly for
        narrow ag. ditches where NHDPLus flowlines often do not
        coincide with the DEM depressions and so stream-elements
        intermitently fall into them.
    bank_integrity_elevation: float, optional
        Where the river is passing right next to the reservoir or
        NHDline is misplaced into the reservoir, banks may fall into
        the reservoir. If true, this will enforce that the bank vertex
        is at a higher elevation than the stream bed elevation.
    depress_headwaters_by: float, optional
        If the depression is not captured well in the DEM, the
        river-mesh elements (streambed) headwater reaches may be
        lowered by this number.  The effect is propogated downstream
        only up to where it is needed to maintain topographic gradients
        on the network scale in the network sweep step.
    network_burn_in_depth: Callable[[River,], float], optional
        A function that takes a reach (River object) as input and returns the burn-in 
        depth for that specific reach. This depth specifies how much to lower the 
        river-mesh elements below their original elevation. The callable allows for 
        dynamic calculation based on reach properties, stream order, or custom logic.
    known_depressions: list, optional
        If provided, a list of IDs to not be burned in via the network
        sweep.

    """
    # conditioning of stream-bed profiles to enforce typical channel
    # depths, large-scale topographic gradients in the streambeds, and
    # connectivity through culverts that pass under road and railway
    # embankments
    if smooth:
        for reach in river:
            # smooth the reach profile
            smoothProfile(reach, lower=lower)

    # network-wide conditioning
    enforceMonotonicity(river, depress_headwaters_by, known_depressions)

    # potentially burn in the network using a depression function
    if network_burn_in_depth is not None:
        burnInRiver(river, network_burn_in_depth)

        # note this breaks continuity vertically -- fix it
        river.makeContinuous()

    # map new profile to mesh
    distributeProfileToMesh(m2, river)

    # ensure that a diked channel passing over/around
    # a pond or reservoirs does not have bank-vertices fall into
    # the depression
    if bank_integrity_elevation > 0.:
        enforceBankIntegrity(m2, river, bank_integrity_elevation)
        
    m2.clearGeometryCache()


def setProfileByDEM(rivers : List[River],
                    dem : xarray.DataArray,
                    **kwargs) -> None:
    """Set the z-coordinate of the reach linestring from a DEM dataset."""
    assert len(rivers) > 0
    points = np.array([c for river in rivers for reach in river for c in reach.linestring.coords])
    elevs = watershed_workflow.data.interpolateValues(points, rivers[0].crs, dem, **kwargs)

    if points.shape[1] == 3:
        new_points = points
        new_points[:,2] = elevs
    else:
        new_points = np.empty((len(points), 3), 'd')
        new_points[:, :2] = points
        new_points[:,2] = elevs

    i = 0
    for river in rivers:
        for reach in river:
            count = len(reach.linestring.coords)
            reach.linestring = shapely.geometry.LineString(new_points[i:i+count])
            i += count

        
def smoothProfile(reach : River,
                  lower : bool = False) -> None:
    """Applies gaussian filter smoothing to the bed-profile obtained from DEM.

    This option becomes important in ag. watersheds when NHDPLus is
    inconsistent with the depression in the DEM.

    """
    ls = reach.linestring
    s = watershed_workflow.utils.computeArclengths(ls)

    coords = np.array(ls.coords)
    new_z = scipy.ndimage.gaussian_filter(coords[:,2], 5, mode='nearest')

    if lower:
        # NHDPlus flowlines may not fall on the DEM depression of the
        # narrow ditch, hence the smoothed bed profile will
        # underestimate the depression In this step, the smoothed bed
        # propfile is depressed by a median of one-sided difference
        # between the raw and smoothed profile
        diffs = new_z - coords[:,2]
        if any(diffs > 0):
            new_z = new_z - np.median(diffs[diffs > 0])

    coords[:,2] = new_z
    reach.linestring = shapely.geometry.LineString(coords)


def enforceLocalMonotonicity(reach : River,
                             moving : Literal['downstream', 'upstream'] = 'downstream') -> None:
    """Ensures that the streambed-profile elevations are monotonically
    increasing as we move upstream, or decreasing as we move
    downstream.

    """
    coords = np.array(reach.linestring.coords)
    if moving == 'upstream':
        for i in range(len(coords) - 1, 0, -1):
            if coords[i, 2] > coords[i - 1, 2]:
                coords[i, 2] = coords[i - 1, 2]

    elif moving == 'downstream':
        for i in range(len(coords) - 1):
            if coords[i + 1, 2] > coords[i, 2]:
                coords[i + 1, 2] = coords[i, 2]

    else:
        raise ValueError(f"Invalid value '{moving}' for enforceMonotonicity()")

    reach.linestring = shapely.geometry.LineString(coords)

    
def enforceMonotonicity(river : River,
                        depress_headwaters_by : Optional[float] = None,
                        known_depressions : Optional[List[int]] = None) -> None:
    """Sweep the river network from each headwater reach (leaf node)
    to the watershed outlet (root node), removing aritificial
    obstructions in the river mesh and enforcing depths of constructed
    channels.

    """
    if known_depressions is None:
        known_depressions = []

    # starting from one of the leaf nodes providing extra depression at the upstream end
    for leaf in river.leaf_nodes:
        if leaf.index not in known_depressions:
            if depress_headwaters_by is not None:
                assert depress_headwaters_by >= 0.
                coords = np.array(leaf.linestring.coords)
                coords[:,2] = coords[:,2] - depress_headwaters_by
                leaf.linestring = shapely.geometry.LineString(coords)

        for reach in leaf.pathToRoot():
            if not reach.index in known_depressions:
                # traversing from leaf reach (headwater) catchment to the root reach
                enforceLocalMonotonicity(reach)
                if reach.parent is not None:
                    junction_elevs = [r.linestring.coords[-1][2] for r in reach.parent.children] \
                        + [reach.parent.linestring.coords[0][2],]
                    new_coord = (reach.linestring.coords[-1][0], reach.linestring.coords[-1][1],
                                 min(junction_elevs))

                    for r in reach.parent.children:
                        r.moveCoordinate(-1, new_coord)
                    reach.parent.moveCoordinate(0, new_coord)

    assert river.isContinuous()
    assert river.isMonotonic(known_depressions)

    
def burnInRiver(river : River,
                network_burn_in_depth : Callable[[River,], float]) -> None:
    """Reduce reach elevations by a float or function."""
    for reach in river:
        coords = np.array(reach.linestring.coords)
        coords[:,2] = coords[:,2] - network_burn_in_depth(reach)
        reach.linestring = shapely.geometry.LineString(coords)

        
def distributeProfileToMesh(m2 : Mesh2D,
                            river : River) -> None:
    """Take reach profile elevations and move them out to the mesh vertices."""
    for reach in river:
        for i, elem in enumerate(reach['elems']):
            m2.coords[elem[1:-1], 2] = reach.linestring.coords[i][2]

        m2.coords[elem[0], 2] = reach.linestring.coords[i+1][2]
        m2.coords[elem[-1], 2] = reach.linestring.coords[i+1][2]

        
def enforceBankIntegrity(m2 : Mesh2D,
                         river : River,
                         bank_integrity_elevation : float) -> None:
    """Forces banks at least bank_integrity_elevation higher than the channel elevation."""
    # collecting IDs of all vertices in the river/stream
    river_corr_ids = set(vertex_id for reach in river for elem in reach['elems'] for vertex_id in elem)

    for reach in river:
        for i, elem in enumerate(reach['elems']):
            bank_vertex_ids = _findBankVerticesFromElem(m2, elem)
            for vertex_id in bank_vertex_ids:
                if vertex_id not in river_corr_ids:
                    midp = (reach.linestring.coords[i][2] + reach.linestring.coords[i+1][2]) / 2
                    if m2.coords[vertex_id][2] < midp + bank_integrity_elevation:
                        logging.info(f"raised vertex {vertex_id} for bank integrity")
                        m2.coords[vertex_id][2] = midp + bank_integrity_elevation

        
def _findBankVerticesFromElem(m2 : Mesh2D,
                              elem : List[int]) -> Edge:
    """For a given m2 mesh and id of river-corridor element, returns
    longitudinal edges of the river-corridor element.

    """
    # 1st and 2nd-to-last edges -- the last is the downstream, cross-stream edge
    elem_edges = [Edge(elem[i], elem[(i+1) % len(elem)]) for i in range(len(elem))]
    edge_r = elem_edges[0]
    edge_l = elem_edges[-2]
    return _findBankVerticesFromEdge(m2, elem, edge_r), _findBankVerticesFromEdge(m2, elem, edge_l)


def _findBankVerticesFromEdge(m2 : Mesh2D,
                              elem : List[int],
                              edge : Edge) -> int:
    """For a given m2 mesh, id of river-corridor element, and edge,
    returns the bank-vertex id, i.e., for the triangle attached to the
    river-corridor, vertex that does not form the river corridor.

    """
    cell_ids = m2.edge_cells[edge]
    cells_to_edge = [m2.conn[cell_id] for cell_id in cell_ids]
    try:
        cells_to_edge.remove(elem)
    except ValueError:
        # could be flipped due to handedness
        cells_to_edge.remove(list(reversed(elem)))
    bank_tri = cells_to_edge[0]
    non_edge_verts = set(bank_tri) - set(edge)
    if len(non_edge_verts) != 1:
        raise RuntimeError('Expected to find a triangle, found a polygon?')
    return non_edge_verts.pop()



