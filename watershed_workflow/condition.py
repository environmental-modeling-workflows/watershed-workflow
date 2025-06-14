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

@attr.s
class _Point:
    """POD struct of coordinate and set of neighbors"""
    coords = attr.ib()
    neighbors: Set[int] = attr.ib(factory=set, converter=set)


def fillPits(m2 : Mesh2D,
             outletID : Optional[int] = None,
             algorithm : int = 3) -> None:
    """Conditions a mesh, IN PLACE, by removing pits.
    
    Starts at outlet and works through all coordinates in the mesh,
    ensuring that there is a path to all vertices of the mesh from the
    outlet that monotonically increases in elevation.

    Available algorithms (likely all should be equivalent):
     1: original, 2-pass algorithm
     2: refactored single-pass algorithm based on sorted lists
     3: boundary marching method.  Should be the fastest.

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to condition.
    outletID : int, optional
      If provided, the ID of the point to start conditioning from.  If
      not provided, will use the boundary vertex with minimal elevation.
    algorithm : int
      See above, defaults to 3.

    """
    # generate a dictionary of ID,Point for all points of the mesh
    points_dict = dict((i, _Point(c)) for (i, c) in enumerate(m2.coords))
    for conn in m2.conn:
        for c in conn:
            points_dict[c].neighbors.update(conn)
    for i, p in points_dict.items():
        p.neighbors.remove(i)

    # set the outlet as minimal boundary elevation
    if outletID is None:
        boundary_vertices = m2.boundary_vertices
        outletID = boundary_vertices[np.argmin(m2.coords[boundary_vertices, 2])]

    if algorithm == 1:
        fillPits1(points_dict, outletID)
    elif algorithm == 2:
        fillPits2(points_dict, outletID)
    elif algorithm == 3:
        fillPits3(points_dict, outletID)
    else:
        raise RuntimeError('Unknown algorithm "%r"' % (algorithm))

    m2.coords = np.array([p.coords for p in points_dict.values()])


def fillPits1(points : Dict[int, _Point],
              outletID : int) -> None:
    """This is the origional, 2-pass algorithm, and is likely inefficient."""

    # create a sorted list of elevations, from largest to smallest
    elev = sorted(list(points.items()), key=lambda id_p: -id_p[1].coords[2])

    visited = set([outletID, ])
    pits = set()
    waterway = set([outletID, ])

    # loop over elevation list from small to large
    while len(elev) != 0:
        current, current_p = elev.pop()
        if current in visited:
            # still in the waterway
            waterway.add(current)
            visited.update(current_p.neighbors)
        else:
            # not in the waterway, add to pits
            pits.add(current)

    # post-conditions
    assert (len(pits.union(waterway)) == len(points))

    # loop over waterway and raise up pits as they touch the waterway
    waterway_l = sorted([(ID, points[ID]) for ID in waterway], key=lambda id_p: -id_p[1].coords[2])
    while len(waterway_l) != 0:
        current, current_p = waterway_l.pop()
        for n in current_p.neighbors:
            if n in pits:
                points[n].coords[2] = max(current_p.coords[2], points[n].coords[2])
                pits.remove(n)
                waterway_l.append((n, points[n]))

    # post-conditions
    assert (len(pits) == 0)
    return


def fillPits2(points : Dict[int, _Point],
              outletID : int) -> None:
    """This is a refactored, single pass algorithm that leverages a sorted list."""

    # create a sorted list of elevations, from largest to smallest
    elev = sortedcontainers.SortedList(list(points.items()), key=lambda id_p: id_p[1].coords[2])
    waterway = set([outletID, ])

    # loop over elevation list from small to large
    while len(elev) != 0:
        current, current_p = elev.pop(0)
        if current in waterway:
            # still in the waterway
            waterway.update(current_p.neighbors)
        else:
            # not in the waterway, fill
            ww_neighbors = [n for n in current_p.neighbors if n in waterway]
            if len(ww_neighbors) != 0:
                current_p.coords[2] = min(points[n].coords[2] for n in ww_neighbors)
            else:
                current_p.coords[2] = min(points[n].coords[2] for n in current_p.neighbors)

            # push back into elev list with new, higher elevation
            elev.add((current, current_p))
    return


def fillPits3(points : Dict[int, _Point],
              outletID : int) -> None:
    """This algorithm is based on a boundary marching method"""
    # Waterway is the list of things that are already conditioned and
    # can be reached.
    waterway = set()

    # Boundary is an elevation-sorted list of (ID, point) tuples which
    # are NOT yet in the waterway, but have a neighbor in the
    # waterway.  Additionally, points in the boundary have been
    # conditioned such that all boundary points must be at least as
    # high as the highest elevation in the waterway.
    boundary = sortedcontainers.SortedList([(outletID, points[outletID]), ],
                                           key=lambda id_p: id_p[1].coords[2])
    waterway_max = -1e10

    while len(boundary) > 0:
        # pop the lowest boundary point and stick it in the waterway
        next_p = boundary.pop(0)
        waterway.add(next_p[0])

        # increment the waterway elevation
        assert (next_p[1].coords[2] >= waterway_max)
        waterway_max = next_p[1].coords[2]

        # Insert all neighbors (that aren't in the waterway already)
        # into the boundary, first checking that their elevation is at
        # least as high as the new waterway elevation.  Note that
        # there can be no "alternate" pathway to the waterway, as this
        # pathway would already be in the boundary (somewhere along
        # that pathway) and therefore would have been popped before
        # this path due to our sorted elevation boundary list.
        for n in next_p[1].neighbors:
            if n not in waterway and (n, points[n]) not in boundary:
                points[n].coords[2] = max(points[n].coords[2], waterway_max)
                boundary.add((n, points[n]))

    assert (len(waterway) == len(points))
    return


def fillPitsDual(m2 : Mesh2D,
                 is_waterbody : Optional[np.ndarray] = None,
                 outlet_edge : Optional[Tuple[int,int]] = None,
                 eps : float = 1e-3) -> None:
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
    outlet_edge : (int,int), optional
      If provided, the point to start conditioning from.  If not
      provided, will use the edge on the boundary of m2 with the
      lowest elevation.
    eps : float, optional=1e-3
      A small vertical displacement for soft tolerances.

    """
    if outlet_edge is None:
        # determine the outlet edge -- the lowest edge point
        boundary_edges = m2.boundary_edges
        outlet_edge = boundary_edges[0]
        be_elev = (m2.coords[outlet_edge[0], 2] + m2.coords[outlet_edge[1], 2]) / 2.

        for e in boundary_edges[1:]:
            eh = (m2.coords[e[0], 2] + m2.coords[e[1], 2]) / 2.
            if eh < be_elev:
                be_elev = eh
                outlet_edge = e
    outlet_edge = m2.edge_hash(*outlet_edge)

    outlet_cell = m2.edges_to_cells[outlet_edge]
    assert (len(outlet_cell) == 1)
    outlet_cell = outlet_cell[0]

    class Waterway:
        """Waterway is the set of cells that are already conditioned and can be reached."""
        def __init__(self):
            self.cells = set()

            # Waterway edges is the set of edges whose cells are all in waterway
            self.edges = set()
            self.max_z = -1e10

        def add(self, be):
            """Add BoundaryEntry object to the waterway"""
            logging.debug(f"adding cell {be.cell} (z = {be.z})")
            self.cells.add(be.cell)
            for e in be.edges:
                self.edges.add(e)
            assert (be.z >= self.max_z)
            self.max_z = be.z

    waterway = Waterway()

    class BoundaryEntry:
        """A cell that is not yet in the waterway, but has at least one edge whose other cell is in the waterway."""
        def __init__(self, cell, edges):
            assert (type(cell) is int)
            assert (0 <= cell < m2.num_cells)
            assert (type(edges) is list)
            for e in edges:
                assert (type(e) is tuple)

            self.cell = cell
            self.edges = edges
            self.z = m2.computeCentroid(self.cell)[2]

    boundary = sortedcontainers.SortedList([BoundaryEntry(outlet_cell, [outlet_edge, ]), ],
                                           key=lambda be: be.z)

    # masked cells are always in the boundary, allowing them to be picked up as we reach that elevation.
    if is_waterbody is not None:
        assert (len(is_waterbody) == m2.num_cells)
        masked_cells = [BoundaryEntry(c, list()) for (c, mask) in enumerate(is_waterbody) if mask]
        boundary.update(masked_cells)

    while len(boundary) > 0:
        # pop the lowest boundary cell and stick its edge and cell
        next_be = boundary.pop(0)
        waterway.add(next_be)

        # find all other edges of the cell just added
        for other_e in m2.cell_edges(m2.conn[next_be.cell]):
            if other_e in waterway.edges: continue

            # find the cell on the other side of other_e
            other_e_cells = m2.edges_to_cells[other_e]
            if len(other_e_cells) == 1:
                # boundary edge, add it to the waterway
                assert (next_be.cell == other_e_cells[0])
                waterway.edges.add(other_e)
                continue

            assert (len(other_e_cells) == 2)
            assert (next_be.cell in other_e_cells)
            if next_be.cell == other_e_cells[0]:
                other_c = other_e_cells[1]
            else:
                assert (next_be.cell == other_e_cells[1])
                other_c = other_e_cells[0]

            # this would break assumption of what it means to be
            # in boundary.
            assert (other_c not in waterway.cells)

            # now we have an other_e, other_c pair to add into
            # boundary.  But first we may need to condition.
            other_c_centroid = m2.computeCentroid(other_c)
            if other_c_centroid[2] < waterway.max_z:
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

                for e in m2.cell_edges(other_c_vertices):
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
                assert (len(free_vertices) > 0)

                # calculate the z of the free vertex required to
                # make the triangle's centroid == waterway_max
                #
                # this formula is likely triangle-only?
                z_free = (waterway.max_z * len(other_c_vertices)
                          - sum(fixed_vertex_elevs.values())) / len(free_vertices) + eps

                # for now, we'll assume triangular.  I'm not sure
                # what to do if this is bigger than length
                # 1... something like evenly raise up all free
                # vertices?  But with triangles, there can only be
                # one free vertex so it is easy.
                assert (len(free_vertices) == 1)
                logging.debug(
                    f'  moving z vertex {free_vertices[0]} from {m2.coords[free_vertices[0],2]} to {z_free}'
                )
                m2.coords[free_vertices[0], 2] = z_free

            # now it is conditioned, add it to the boundary
            try:
                # is it already in the boundary?
                other_be = next(be for be in boundary if be.cell == other_c)
            except StopIteration:
                # no, add it
                logging.debug(f'  adding to boundary: edge: {other_e}  cell: {other_c}')
                boundary.add(BoundaryEntry(other_c, [other_e, ]))
            else:
                # yes, just add this edge to that entry
                if other_e not in other_be.edges:
                    other_be.edges.append(other_e)

    # when this is done, all cells should be in waterway
    assert (len(waterway.cells) == m2.num_cells)
    assert (len(waterway.edges) == m2.num_edges)

    # delete the centroid info to force recalculation
    m2.clearGeometryCache()
    return


def identifyLocalMinima(m2 : Mesh2D) -> np.ndarray:
    """For all cells, identify if their centroid elevation is lower than
    the elevation of all neighbors.

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to check.

    Returns
    -------
    np.array
      Array of 0s and 1s, where 1 indicates a local minima.

    """
    # this is effectively used as a diagnostic
    res = np.zeros((m2.num_cells, ), )
    for cell, conn in enumerate(m2.conn):
        higher = []
        for e in m2.cell_edges(conn):
            # find the other cell
            e_cells = m2.edges_to_cells[e]
            if len(e_cells) > 1:
                if e_cells[0] == cell:
                    other_cell = e_cells[1]
                elif e_cells[1] == cell:
                    other_cell = e_cells[0]
                else:
                    raise RuntimeError("Mismatch, cell not in edges_to_cells?")
            else:
                continue
            if m2.centroids[other_cell][-1] > m2.centroids[cell][-1]:
                higher.append(True)
            else:
                higher.append(False)
                break
        if all(higher):
            res[cell] = 1

    return res


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


def createDepressionFunction(arg):
    if isinstance(arg, dict):
        def func(reach):
            return arg[reach[names.ORDER]]
    elif isinstance(arg, Callable):
        func = arg
    else:
        def func(reach):
            return arg
    return func

        
def conditionRiverMesh(m2 : Mesh2D,
                       river : River,
                       smooth : bool = False,
                       use_parent : bool = False,
                       lower : bool = False,
                       bank_integrity_elevation : float = 0.0,
                       depress_headwaters_by : Optional[float] = None,
                       network_burn_in_depth : Optional[float | Dict[int,float] | Callable[[River,], float]] = None,
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
    network_burn_in_depth: float, dict, or function
        Like depress_headwaters_by, this also lowers river-mesh elements
        by this value, but this variant lowers all reaches.  The depth
        may be provided as a float (uniform lowering), dictionary
        {stream order : depth to depress by}, or as a function of
        drainage area.
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
                assert depress_headwaters_by > 0.
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
                network_burn_in_depth : float | Dict[int,float] | Callable[[River,], float]) -> None:
    """Reduce reach elevations by a float or function."""
    depressionFunction = createDepressionFunction(network_burn_in_depth)
    for reach in river:
        coords = np.array(reach.linestring.coords)
        coords[:,2] = coords[:,2] - depressionFunction(reach)
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
    cells_to_edge.remove(elem)
    bank_tri = cells_to_edge[0]
    vertex_id = (set(bank_tri) - set(edge)).pop()
    return vertex_id


