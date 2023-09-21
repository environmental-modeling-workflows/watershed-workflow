import numpy as np
import attr
import sortedcontainers
import logging
import copy
import math
import scipy.ndimage

import watershed_workflow


@attr.s
class _Point:
    """POD struct of coordinate and set of neighbors"""
    coords = attr.ib()
    neighbors = attr.ib(factory=set)


def fill_pits(m2, outlet=None, algorithm=3):
    """Conditions a mesh, IN PLACE, by removing pits.
    
    Starts at outlet and works through all coordinates in the mesh,
    ensuring that there is a path to all nodes of the mesh from the
    outlet that monotonically increases in elevation.

    Available algorithms (likely all should be equivalent):
     1: original, 2-pass algorithm
     2: refactored single-pass algorithm based on sorted lists
     3: boundary marching method.  Should be the fastest.

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to condition.
    outlet : int, optional
      If provided, the ID of the point to start conditioning from.  If
      not provided, will use the boundary node with minimal elevation.
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
    if outlet is None:
        boundary_nodes = m2.boundary_nodes
        outlet = boundary_nodes[np.argmin(m2.coords[boundary_nodes, 2])]

    if algorithm == 1:
        fill_pits1(points_dict, outlet)
    elif algorithm == 2:
        fill_pits2(points_dict, outlet)
    elif algorithm == 3:
        fill_pits3(points_dict, outlet)
    else:
        raise RuntimeError('Unknown algorithm "%r"' % (algorithm))

    m2.points = np.array([p.coords for p in points_dict.values()])


def fill_pits1(points, outletID=None):
    """This is the origional, 2-pass algorithm, and is likely inefficient."""

    if outletID is None:
        outletID = np.argmin(np.array([points[i].coords[2] for i in range(len(points))]))

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
    waterway = sorted([(ID, points[ID]) for ID in waterway], key=lambda id_p: -id_p[1].coords[2])
    while len(waterway) != 0:
        current, current_p = waterway.pop()
        for n in current_p.neighbors:
            if n in pits:
                points[n].coords[2] = max(current_p.coords[2], points[n].coords[2])
                pits.remove(n)
                waterway.append((n, points[n]))

    # post-conditions
    assert (len(pits) == 0)
    return


def fill_pits2(points, outletID):
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


def fill_pits3(points, outletID):
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


def fill_pits_dual(m2, is_waterbody=None, outlet_edge=None, eps=1e-3):
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
            self.z = m2.compute_centroid(self.cell)[2]

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
            other_c_centroid = m2.compute_centroid(other_c)
            if other_c_centroid[2] < waterway.max_z:
                other_c_nodes = m2.conn[other_c]

                # for this to be possible, there must be at least
                # one free node in the nodes of next_c.  By free,
                # we mean that its elevation can be changed
                # without breaking everything.  This means that
                # neither of that node's edges can be in
                # waterway.edges or boundary.
                #
                # we also need the fixed (non-free) node elevations
                fixed_node_elevs = dict()

                for e in m2.cell_edges(other_c_nodes):
                    if (e == other_e) or (e in waterway.edges) or any(
                        (e == i) for be in boundary for i in be.edges):
                        if e[0] not in fixed_node_elevs:
                            fixed_node_elevs[e[0]] = m2.coords[e[0], 2]
                        if e[1] not in fixed_node_elevs:
                            fixed_node_elevs[e[1]] = m2.coords[e[1], 2]
                free_nodes = [n for n in other_c_nodes if n not in fixed_node_elevs]

                # should not be possible to be both lower
                # elevation and not have a free node, or it would
                # already be in boundary, and therefore have no
                # free nodes
                assert (len(free_nodes) > 0)

                # calculate the z of the free node required to
                # make the triangle's centroid == waterway_max
                #
                # this formula is likely triangle-only?
                z_free = (waterway.max_z * len(other_c_nodes)
                          - sum(fixed_node_elevs.values())) / len(free_nodes) + eps

                # for now, we'll assume triangular.  I'm not sure
                # what to do if this is bigger than length
                # 1... something like evenly raise up all free
                # nodes?  But with triangles, there can only be
                # one free node so it is easy.
                assert (len(free_nodes) == 1)
                logging.debug(
                    f'  moving z node {free_nodes[0]} from {m2.coords[free_nodes[0],2]} to {z_free}'
                )
                m2.coords[free_nodes[0], 2] = z_free

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
    m2.clear_geometry_cache()
    return


def identify_local_minima(m2):
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


def smooth(img_in, algorithm='gaussian', **kwargs):
    """Smooths an image according to an algorithm, passing kwargs on to that algorithm."""
    if algorithm == 'gaussian':
        if 'method' not in kwargs:
            kwargs['method'] = 'nearest'
        if 'sigma' not in kwargs:
            sigma = 5
        else:
            sigma = kwargs.pop('sigma')
        return scipy.ndimage.gaussian_filter(img_in, sigma, **kwargs)
    else:
        raise ValueError(f'Unknown smoothing algorithm: "{algorithm}"')


def fill_gaps(img_in, nodata=np.nan):
    import scipy.interpolate

    if nodata is np.nan:
        mask = ~np.isnan(img_in)
    else:
        mask = ~(img_in == nodata)

    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(img_in.shape[1]), np.arange(img_in.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values, as 1D arrays (in the same order as their coordinates in xym)
    img_in0 = np.ravel(img_in[:, :][mask])

    # interpolator
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, img_in0)

    # interpolate the whole image
    return interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)


def condition_river_mesh(m2,
                         river,
                         smooth=False,
                         use_parent=False,
                         lower=False,
                         use_nhd_elev=False,
                         treat_banks=False,
                         depress_upstream_by=None,
                         network_burn_in_depth=None,
                         ignore_in_sweep=None):
    """Condition, IN PLACE, the elevations of stream-corridor elements
   to ensure connectivity throgh culverts, skips ponds, maintain
   monotonicity, or otherwise enforce depths of constructed channels.

    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
        2D mesh with 3D coordinates.
    river: watershed_workflow.river_tree.River object
        River tree with node.elements added for quads
    smooth: boolean, optional
        If true, smooth the profile of each reach using a gaussian
        filter (mainly to pass through railroads and avoid reservoirs).
    use_parent: boolean, optional
        If true, use the segment from the original parent reach while
        smoothing (seems to be not making a huge difference).
    lower: boolean, optional
        If true, lower the smoothed bed profile to match the lower
        points on the raw bed profile. This is useful particularly for
        narrow ag. ditches where NHDPLus flowlines often do not
        coincide with the DEM depressions and so stream-elements
        intermitently fall into them.
    use_nhd_elev: boolean, optional
        If true, enforce maximum and minimum elevation for each
        reach provided in NHDPlus.
    treat_banks: boolean, optional
        Where the river is passing right next to the reservoir or
        NHDline is misplaced into the reservoir, banks may fall into
        the reservoir. If true, this will enforce that the bank node
        is at a higher elevation than the stream bed elevation.
    depress_upstream_by: float, optional
        If the depression is not captured well in the DEM, the
        river-mesh elements (streambed) headwater reaches may be
        lowered by this number.  The effect of propogated downstream
        only upto where it is needed to maintain topographic gradients
        on the network scale in the network sweep step.
    network_burn_in_depth: float, dict, or function
        Like depress_upstream_by, this also lowers river-mesh elements
        by this value, but this variant lowers all reaches.  The depth
        may be provided as a float (uniform lowering), dictionary
        {stream order : depth to depress by}, or as a function of
        drainage area.
    ignore_in_sweep: list, optional
        If provided, a list of IDs to not be burned in via the network
        sweep.

    """
    river_corr_ids = []  # collecting IDs of all nodes in the river/stream
    for node in river.preOrder():
        for elem in node.elements:
            for id in elem:
                if id not in river_corr_ids:
                    river_corr_ids.append(id)

    # conditioning of stream-bed profiles to enforce typical channel
    # depths, large-scale topographic gradients in the streambeds, and
    # connectivity through culverts that pass under road and railway
    # embankments
    if smooth:
        for node in river.preOrder():  # reachwise smoothing
            # adds smooth profile in node properties
            _smooth_profile(node, use_parent=use_parent, lower=lower)

        _network_sweep(river,
                       depress_upstream_by=depress_upstream_by,
                       use_nhd_elev=use_nhd_elev,
                       ignore_in_sweep=ignore_in_sweep)  # network-wide conditioning

    # transferring network-scale-conditioned stream-bed elevations
    # onto the mesh
    for node in river.preOrder():
        if smooth:
            profile = node.properties["SmoothProfile"]
        else:
            # only centerline elevation is to be used, without any
            # conditioning
            profile = _get_profile(node)

        if network_burn_in_depth is not None:
            if isinstance(network_burn_in_depth, dict):
                order = node.properties["StreamOrder"]
                if order > max(network_burn_in_depth.keys()):
                    burn_in_depth = network_burn_in_depth[max(network_burn_in_depth.keys())]
                elif order < min(network_burn_in_depth.keys()):
                    burn_in_depth = network_burn_in_depth[min(network_burn_in_depth.keys())]
                else:
                    burn_in_depth = network_burn_in_depth[order]
            elif callable(network_burn_in_depth):
                # can we pass such user-defined functions for channel width
                # and depth as a function of drainage area W or D =
                # f(Drainage_area)
                DA_sqm = node.properties['TotalDrainageAreaSqKm'] * 1e6
                burn_in_depth = network_burn_in_depth(DA_sqm)
            else:
                burn_in_depth = network_burn_in_depth

            profile[:, 1] = profile[:, 1] - burn_in_depth

        for i, elem in enumerate(node.elements):
            if i == 0:  # for the first point
                m2.coords[elem[0]][2] = m2.coords[elem[-1]][2] = profile[i, 1]

            # assign elevations to the upstream points of each
            # river-corridor element (quads)
            for coord_id in elem[1:-1]:
                m2.coords[coord_id][2] = profile[i + 1, 1]

            # this to ensure that a diked channel passing over/around
            # a pond or reservoirs does not have bank-nodes fall into
            # the depression
            if treat_banks:
                bank_node_ids = _bank_nodes_from_elem(elem, m2)
                for node_id in bank_node_ids:
                    if node_id not in river_corr_ids:
                        if m2.coords[node_id][2] < min(profile[i + 1, 1], profile[i, 1]):
                            logging.info(f"raised node {node_id} for bank integrity")
                            m2.coords[node_id][2] = 0.5 * (profile[i, 1] + profile[i + 1, 1]) + 0.55
    m2.clear_geometry_cache()


def _get_profile(node):
    """For a given node, generate a bedprofile using elevations on the node.segment."""
    # Note that node_elems are downstream to upstream, while segment coords are upstream to downstream.
    stream_bed_coords = list(reversed(node.segment.coords))
    dists = [math.dist(stream_bed_coords[0], point) for point in stream_bed_coords]
    elevs = node.properties['elev_profile'][::-1]  # reversed
    profile = np.array([dists, elevs]).T
    return profile


def _smooth_profile(node, use_parent=False, lower=False):
    """Applies gaussian filter smoothing to the bed-profile obtained from DEM.

    This option becomes important in ag. watersheds when NHDPLus is
    inconsistent with the depression in the DEM.  One can also include
    elevation profile of the parent node for better continuity,
    although, subsequent network sweep option makes using parent
    profile redundant.

    """
    profile = _get_profile(node)
    profile_new = copy.deepcopy(profile)

    if use_parent:
        if node.parent is None:
            profile_new[:, 1] = scipy.ndimage.gaussian_filter(profile[:, 1], 5, mode='nearest')
        else:
            parent_profile = node.parent.properties['SmoothProfile']
            profile_to_smooth = np.vstack((parent_profile, profile_new[1:, :]))
            profile_to_smooth[len(parent_profile):,
                              0] = profile_to_smooth[len(parent_profile):, 0] + profile_to_smooth[
                                  len(parent_profile) - 1, 0]  # shift the distances
            profile_to_smooth[:, 1] = scipy.ndimage.gaussian_filter(profile_to_smooth[:, 1],
                                                                    5,
                                                                    mode='nearest')  # smooth filter
            profile_new[:, 1] = profile_to_smooth[-len(profile):, 1]

    else:
        profile_new[:, 1] = scipy.ndimage.gaussian_filter(profile[:, 1], 5, mode='nearest')

    if lower:
        # NHDPlus flowlines may not fall on the DEM depression of the
        # narrow ditch, hence the smoothed bed profile will
        # underestimate the depression In this step, the smoothed bed
        # propfile is depressed by a median of one-sided difference
        # between the raw and smoothed profile
        diffs = profile_new[:, 1] - profile[:, 1]
        if any(diffs > 0):
            profile_new[:, 1] = profile_new[:, 1] - np.median(diffs[diffs > 0])

    node.properties["SmoothProfile"] = profile_new
    return profile_new


def _enforce_monotonicity(profile, moving='upstream'):
    """Ensures that the streambed-profile elevations are monotonically
    increasing as we move upstream, or decreasing as we move
    downstream.
    """
    profile_new = copy.deepcopy(profile)
    if moving == 'upstream':
        for i in range(len(profile_new) - 1):
            if profile_new[i + 1, 1] < profile_new[i, 1]:
                profile_new[i + 1, 1] = profile_new[i, 1]

    elif moving == 'downstream':
        for i in range(len(profile_new) - 1, 0, -1):
            if profile_new[i - 1, 1] > profile_new[i, 1]:
                profile_new[i - 1, 1] = profile_new[i, 1]

    else:
        raise ValueError(f"Invalid value '{moving}' for enforce_monotonicity()")

    return profile_new


def _network_sweep(river, depress_upstream_by=None, use_nhd_elev=False, ignore_in_sweep=None):
    """Sweep the river network from each headwater reach (leaf node)
    to the watershed outlet (root node), removing aritificial
    obstructions in the river mesh and enforcing depths of constructed
    channels.

    """
    if ignore_in_sweep is None:
        ignore_in_sweep = []

    for leaf in river.leaf_nodes():  #starting from one of the leaf nodes
        # providing extra depression at the upstream end
        if depress_upstream_by is not None:
            leaf.properties['SmoothProfile'][-1, 1] -= depress_upstream_by

        for node in leaf.pathToRoot():
            # traversing from leaf node (headwater) catchment to the root node
            node.properties['SmoothProfile'] = _enforce_monotonicity(
                node.properties['SmoothProfile'], 'downstream')  # making monotonous
            junction_elevs = [sib.properties['SmoothProfile'][0, 1] for sib in node.siblings()]

            if use_nhd_elev:
                junction_elevs.append(node.properties['MinimumElevationSmoothed'] / 100)
            if node.parent != None and node.properties['ID'] not in ignore_in_sweep:
                junction_elevs.append(node.parent.properties['SmoothProfile'][-1, 1])
                node.parent.properties['SmoothProfile'][-1, 1] = min(
                    junction_elevs)  # giving min junction elevation to both the siblings
            for sib in node.siblings():
                sib.properties['SmoothProfile'][0, 1] = min(junction_elevs)


def _bank_nodes_from_elem(elem, m2):
    """For a given m2 mesh and id of river-corridor element, returns
    longitudinal edges of the river-corridor element.

    """
    # edge on the right as we look from the downstream direction
    edge_r = list(m2.cell_edges(elem))[0]

    # edge on the left as we look from the downstream direction
    edge_l = list(m2.cell_edges(elem))[2]

    return [_bank_nodes_from_edge(edge_r, elem, m2), _bank_nodes_from_edge(edge_l, elem, m2)]


def _bank_nodes_from_edge(edge, elem, m2):
    """For a given m2 mesh, id of river-corridor element, and edge,
    returns the bank-node id, i.e., for the triangle attached to the
    river-corridor, node that does not form the river corridor.
    """

    cell_ids = m2.edges_to_cells[edge]
    cells_to_edge = [m2.conn[cell_id] for cell_id in cell_ids]
    cells_to_edge.remove(elem)
    bank_tri = cells_to_edge[0]
    node_id = (set(bank_tri) - set(edge)).pop()
    return node_id


def elevate_rivers(rivers, crs, dem, dem_profile):
    """Elevate the river using dem and store reach-bed-profile as node properties."""
    for river in rivers:
        for i, node in enumerate(river.preOrder()):
            node_points = (np.array(node.segment.xy).T)
            node_elevs = watershed_workflow.elevate(node_points, crs, dem, dem_profile)[:, 2]
            assert (len(node_elevs) == len(node.segment.coords))
            node.properties['elev_profile'] = node_elevs
