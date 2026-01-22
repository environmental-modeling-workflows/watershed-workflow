from typing import Optional, Iterable, Dict, Tuple, List, Literal, Callable, Set

import numpy as np
import attr
import sortedcontainers
import logging
import math
import shapely
import xarray
import scipy.ndimage

import watershed_workflow.sources.standard_names as names
import watershed_workflow.utils
from watershed_workflow.mesh import Mesh2D
from watershed_workflow.river_tree import River
import watershed_workflow.data


def fillPits_marching(m2 : Mesh2D,
                 is_waterbody : Optional[np.ndarray] = None,
                 outlet_edge : Optional[Tuple[int,int]] = None,
                 eps : float = 1e-3,
                 debug : Optional[List[int]] = None,
                 ) -> None:
    """Conditions the dual of the mesh, IN PLACE, by filling pits.

    This ensures the property that, starting with an outlet cell,
    there is a path to every cell by way of faces that is
    monotonically increasing in elevation (except in cells which are a
    part of waterbodies).

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to condition.
    is_waterbody : np.ndarray(int, shape=ncells), optional
      A boolean/integer mask of length m2.num_cells, with True
      values indicating that the given cell is a part of a waterbody
      that may be a depression.  This is important for
      reservoirs/lakes/etc where bathymetry is known and pits are
      physical.
    outlet_edge : Tuple[int,int], optional
      If provided, the point to start conditioning from.  If not
      provided, will use the edge on the boundary of m2 with the
      lowest elevation.
    eps : float, optional=1e-3
      A small vertical displacement for soft tolerances.
    debug : List[int]
      A list of cells to debug

    """
    # input processing
    if debug is None:
        debug = []

    # place the outlet 
    if outlet_edge is None:
        # determine the outlet edge -- the lowest edge point
        def computeEdgeCentroid(e):
            return (m2.coords[[0], 2] + m2.coords[e[1], 2]) / 2. 
        outlet_edge = min(m2.boundary_edges, key=computeEdgeCentroid)
    outlet_edge = m2.edge_hash(*outlet_edge)

    if is_waterbody is not None:
        is_waterbody = np.zeros(len(m2.num_cells), 'i')
        # assert len(is_waterbody) == m2.num_cells
        # if len(debug):
        #     logging.info('fillPitDual: putting waterways in boundary...')

        # for c in range(len(is_waterbody)):
        #     if is_waterbody[c]:
        #         # note this calls the private add, which adds without
        #         # conditioning or putting neighbors in the boundary
        #         waterway._add(c)

    
    # Algorithmic design
    #
    # The goal is to start from a list of "waterway" cells, which have
    # fixed node elevations and cannot be changed, and a list of
    # "boundary" cells, which are options for picking up at any point.
    #
    # Cells in the boundary:
    # - are not conditioned yet
    # - have at least one edge in the waterway
    # - are stored in such a way to make getting the lowest elevation
    #   cell easily
    # NOTE: cannot be stored in a sortedcontainer because the centroid
    # may change.  So every time we get the min elevation cell, we
    # must recompute.
    class Boundary:
        def __init__(self):
            self.cells = set()

        def __len__(self):
            return len(self.cells)

        def pop(self):
            min_c = min(self.cells, key=lambda c : m2.computeCentroid(c)[2])
            self.cells.remove(min_c)
            return min_c

        def add(self, c):
            self.cells.add(c)

    #
    # Cells in the waterway:
    # - all cell edges are in the waterway
    # - all cell nodes are in the waterway
    # - all cell nodes are fixed -- elevation will not change
    # - all cell centroids are either conditioned, e.g. they have a
    #   downhill-flowing path to an outlet, or 
    class Waterway:
        """Waterway is the set of cells that are already conditioned and can be reached."""
        def __init__(self, eps):
            # cells in the waterway
            self.cells = set()

            # a set of nodes with at least one edge in the waterway.
            #
            # These nodes have already been touched, and therefore
            # cannot have their elevation changed without breaking
            # things.
            self.nodes = set()
            
            # set the current high water mark
            self.max_z = -1e10

            # incremental amount each pathway must go up
            self.eps = eps

            # the boundary of the waterway
            self.boundary = Boundary()

        def _add(self, c):
            """Add cell to the waterway, but bypass the usual logic.

            Used only for waterbodies.
            """
            self.cells.add(c)

            for n in m2.conn[c]:
                self.nodes.add(n)
                
        def add(self, c):
            self.condition(c)
            self._add(c)
            c_z = m2.computeCentroid(c)[2]
            assert c_z >= self.max_z
            logging.debug(f"adding cell {c} to waterway (conditioned z = {c_z})")
            self.max_z = c_z

            # also add neighbors to the boundary
            for other_c in m2.cell_to_cells[c]:
                if other_c not in self.cells:
                    self.boundary.add(other_c)
                    
        def condition(self, c):
            cc = m2.computeCentroid(c)
            target_z = self.max_z + self.eps
            if cc[2] < target_z:
                c_nodes = m2.conn[c]

                # for this to be possible, there must be at least
                # one free vertex in the vertices of next_c.  By free,
                # we mean that its elevation can be changed
                # without breaking everything.  This means that
                # none of that vertex's edges can be in
                fixed_vertex_elevs = dict()
                free_vertices = []
                for v in c_nodes:
                    if v in self.nodes:
                        fixed_vertex_elevs[v] = m2.coords[v,2]
                    else:
                        free_vertices.append(v)

                # should not be possible to be both lower
                # elevation and not have a free vertex, or it would
                # already be in boundary, and therefore have no
                # free vertices
                if len(free_vertices) == 0:
                    logging.info(f"No free vertices when conditioning cell {c} at z = {cc[2]}")
                    cell_neighbors = m2.cell_to_cells[c]
                    for c2c in cell_neighbors:
                        logging.info(f"  cell_neighbor : {c2c} at z = {m2.computeCentroid(c2c)[2]}")
                    
                    raise RuntimeError(f"No free vertices when conditioning cell {c}")

                # this formula is triangle-only -- this means there
                # can be at most one free vertex, since we just put an
                # edge in the waterway
                #
                # it could probably be relaxed to non-triangle, but
                # then we would have to think about what we wanted the
                # free_nodes to do -- do they all go up an equal
                # amount?  All go up to a common, min required
                # elevation?  What happens when that min required
                # elevation is less than the max free elevation?
                assert len(free_vertices) == 1

                # calculate the z of the free vertices required to
                # make the triangle's centroid == waterway_max
                z_free = (target_z * (len(free_vertices) + len(fixed_vertex_elevs))
                          - sum(fixed_vertex_elevs.values())) / len(free_vertices)

                logging.debug(
                    f'  moving z vertex {free_vertices[0]} from {m2.coords[free_vertices[0],2]} to {z_free}'
                )
                m2.coords[free_vertices[0], 2] = z_free

        
    # first put all waterway cells in the waterway.  This will fix the
    # nodes and edges, but not alter the max_z, nor do any
    # conditioning.
    waterway = Waterway(eps)

    # add outlet cell to the waterway
    outlet_cells = m2.edges_to_cells[outlet_edge]
    assert len(outlet_cells) == 1
    outlet_cell = outlet_cells[0]
    waterway.add(outlet_cell)

    while len(waterway.boundary) > 0:
        waterway.add(waterway.boundary.pop())

    # when this is done, all cells should be in waterway
    assert len(waterway.cells) == m2.num_cells

    # delete the centroid info to force recalculation
    m2.clearGeometryCache()
    return

def isLocalMinima(m2, c):
    cc = m2.computeCentroid(c)
    for e in m2.cell_edges(m2.conn[c]):
        e_cells = m2.edges_to_cells[e]
        if len(e_cells) > 1:
            other_c = next(c2 for c2 in e_cells if c2 != c)
            if m2.computeCentroid(other_c)[2] <= cc[2]:
                return False
    return True

def isBoundaryCell(m2, c):
    return any(len(m2.edges_to_cells[e]) == 1 for e in m2.conn[c])

        
def identifyLocalMinima(m2 : Mesh2D) -> np.ndarray:
    """For all cells, identify if their centroid elevation is lower than
    the elevation of all neighbors, and it is not on the boundary.

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to check.

    Returns
    -------
    np.array
      Array of 0s and 1s, where 1 indicates a local minima.

    """
    return np.array([isLocalMinima(m2, c) and not isBoundaryCell(m2, c) for c in range(m2.num_cells)])


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
                       use_parent : bool = False,
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
                              elem : List[int]) -> Tuple[int,int]:
    """For a given m2 mesh and id of river-corridor element, returns
    longitudinal edges of the river-corridor element.

    """
    # 1st and 2nd-to-last edges -- the last is the downstream, cross-stream edge
    edge_r = list(m2.cell_edges(elem))[0]
    edge_l = list(m2.cell_edges(elem))[-2]
    return _findBankVerticesFromEdge(m2, elem, edge_r), _findBankVerticesFromEdge(m2, elem, edge_l)


def _findBankVerticesFromEdge(m2 : Mesh2D,
                              elem : List[int],
                              edge : Tuple[int,int]) -> int:
    """For a given m2 mesh, id of river-corridor element, and edge,
    returns the bank-vertex id, i.e., for the triangle attached to the
    river-corridor, vertex that does not form the river corridor.

    """
    cell_ids = m2.edges_to_cells[edge]
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



