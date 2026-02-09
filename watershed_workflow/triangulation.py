"""Triangulates polygons"""
from typing import Tuple, List, Optional, Set, Iterator

import logging
import collections
import itertools
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import scipy.spatial

import shapely

import watershed_workflow.river_tree
import watershed_workflow.split_hucs


class Nodes:
    """A collection of nodes that are indexed in the order they are added.

    Note this uses round() for an efficient solution, which is
    potentially fragile if numbers were not generated consistently.
    In this use case, however, it should be safe -- numbers were
    originally rounded (in watershed_workflow.config), and then any function which
    inserted new points or moved points were always careful to ensure
    that all duplicates were always assigned the identical values --
    i.e. all math was done BEFORE assignment.  So duplicates have
    identical representations in floating point.  This suggests we
    shouldn't have to round here at all, but we do anyway to keep
    things cleaner.
    """
    def __init__(self, decimals: int = 3) -> None:
        """Initialize a Nodes collection.
        
        Parameters
        ----------
        decimals : int, optional
            Number of decimal places for rounding coordinates. Default is 3.
        """
        self.decimals = decimals
        self._i = 0
        self._store: collections.OrderedDict[Tuple[float, ...], int] = collections.OrderedDict()

    def __len__(self) -> int:
        """Length is tracked by the counter"""
        return self._i

    def __getitem__(self, key: Tuple[float, ...]) -> int:
        """Get item index, setting if default."""
        # Note this implementation could be optimized, and should be
        # done so assuming keys are NOT there, which is the most
        # likely case.
        key = tuple(round(p, self.decimals) for p in key)
        if key not in self._store:
            self._store[key] = self._i
            self._i += 1
        return self._store[key]

    def __iter__(self) -> Iterator[Tuple[float, ...]]:
        """Iterable collection"""
        for k in self._store.keys():
            yield k


def connectOnewayTrip(inds: List[int]) -> List[Tuple[int, int]]:
    """Connect indices in edges in a oneway fashion"""
    return [(inds[i], inds[i + 1]) for i in range(len(inds) - 1)]


def connectRoundTrip(inds: List[int]) -> List[Tuple[int, int]]:
    """Connect indices in edges in a round-trip fashion"""
    return connectOnewayTrip(inds) + [(inds[-1], inds[0]), ]


def orient(e: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Orient an edge consistently.
    
    Parameters
    ----------
    e : Tuple[int, int]
        Edge as tuple of two vertex indices.
        
    Returns
    -------
    Optional[Tuple[int, int]]
        Oriented edge or None if vertices are identical.
    """
    if e[0] > e[1]:
        return e[1], e[0]
    elif e[0] < e[1]:
        return e[0], e[1]
    else:
        return None


class NodesEdges:
    """A collection of nodes and edges."""
    def __init__(
        self,
        objlist: Optional[List[shapely.geometry.LineString | shapely.geometry.Polygon]] = None
    ) -> None:
        """Initialize NodesEdges collection.
        
        Parameters
        ----------
        objlist : Optional[List[shapely.geometry.LineString | shapely.geometry.Polygon]], optional
            List of geometry objects to add. Default is None.
        """
        self.nodes = Nodes()
        self.edges: Set[Tuple[int, int]] = set()

        if objlist != None:
            for obj in objlist:
                self.add(obj)

    def add(self, obj: shapely.geometry.LineString | shapely.geometry.Polygon) -> None:
        """Adds nodes and edges from obj into collection."""
        if isinstance(obj, shapely.geometry.LineString):
            inds = [self.nodes[c] for c in obj.coords]

            for e in connectOnewayTrip(inds):
                oe = orient(e)
                if oe is not None:
                    self.edges.add(oe)

        elif isinstance(obj, shapely.geometry.Polygon):
            inds = [self.nodes[c] for c in obj.boundary.coords]

            for e in connectRoundTrip(inds):
                oe = orient(e)
                if oe is not None:
                    self.edges.add(oe)

        else:
            raise TypeError("Invalid type for add, %r" % type(obj))

    def check(self, tol: float) -> None:
        """Checks consistency of the internal representation."""
        logging.info(" checking graph consistency")
        logging.info(" tolerance is set to {}".format(tol))
        min_dist = 1.e10
        coords = np.array(list(self.nodes))
        kdtree = scipy.spatial.cKDTree(coords)
        bad_pairs = kdtree.query_pairs(tol)

        # Retrieve the coordinates for each bad pair
        bad_pair_coords = [(coords[i], coords[j]) for i, j in bad_pairs]

        if len(bad_pairs) != 0:
            raise ValueError('tol= {} is too large, try decrease tolerance!'.format(tol)
                             + 'or check bad pairs={}'.format(bad_pair_coords))

        min_node = min(self.nodes[n] for n in self.nodes)
        max_node = max(self.nodes[n] for n in self.nodes)
        assert (min_node == 0)
        assert (max_node == len(self.nodes) - 1)

        min_edge_node = min(n for e in self.edges for n in e)
        max_edge_node = max(n for e in self.edges for n in e)
        assert (min_edge_node == 0)
        assert (max_edge_node == len(self.nodes) - 1)


def triangulate(hucs: watershed_workflow.split_hucs.SplitHUCs,
                internal_boundaries: Optional[List[shapely.geometry.LineString]] = None,
                hole_points: Optional[List[shapely.geometry.Point]] = None,
                additional_vertices: Optional[List[Tuple[float, float]]] = None,
                tol: float = 1.0,
                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulates HUCs and rivers.

    Note, refinement of a given triangle is done if any of the provided
    criteria is met.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs()
    internal_boundaries : list, optional
        List of shapely objects or RiverTrees or other iterable
        collections of coordinates used as internal boundaries that
        must be included in the mesh.
    hole_points : list(shapely.Point), optional
        List of points inside the polygons to be left as holes/voids (excluded from mesh).
    additional_vertices : list(Tuple[float, float]), optional
        List of points to be included in the triangulation.
    tol : float, optional
        Set tolerance for minimum distance between two nodes. The unit
        is the same as that of the watershed's CRS. The default is 1.

    Additional keyword arguments include all options for meshpy.triangle.build()

    """
    import meshpy.triangle

    logging.info("Triangulating...")

    # get a list of required boundaries, both interior and exterior
    linestrings: List[shapely.geometry.LineStrings] = []

    # -- internal boundaries may include a river mesh and must go first
    if internal_boundaries is not None:
        linestrings.extend(internal_boundaries)
    linestrings = linestrings + list(hucs.linestrings)

    # convert to nodes and edges
    nodes_edges = NodesEdges(linestrings)
    logging.info("   %i points and %i facets" % (len(nodes_edges.nodes), len(nodes_edges.edges)))
    nodes_edges.check(tol=tol)

    logging.info(" building graph data structures")
    info = meshpy.triangle.MeshInfo()
    nodes = np.array(list(nodes_edges.nodes))
    
    # add additional vertices if provided
    if additional_vertices is not None:
        nodes = np.vstack((nodes, additional_vertices))

    pdata = [tuple([float(c) for c in p]) for p in nodes]
    info.set_points(pdata)
    fdata = [[int(i) for i in f] for f in nodes_edges.edges]
    info.set_facets(fdata)

    # add hole points, which should include the river mesh interior
    if hole_points is not None:
        info.set_holes(hole_points)

    logging.info(" triangle.build...")

    # pop this option if false, which silences the warning if it does
    # not exist but we didn't ask for it anyway.
    if 'enforce_delaunay' in kwargs.keys() and not kwargs['enforce_delaunay']:
        kwargs.pop('enforce_delaunay')

    try:
        mesh = meshpy.triangle.build(info, **kwargs)
    except TypeError as err:
        try:
            # our modification to meshpy.triangle is not present, try without it
            kwargs.pop('enforce_delaunay')
        except KeyError:
            raise err
        else:
            logging.warning(
                "Triangulate: '--enforce-delaunay' option requires a hacked `meshpy.triangle`.  Proceeding without this option because it is not recognized."
            )
            mesh = meshpy.triangle.build(info, **kwargs)

    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    logging.info("  ...built: %i mesh points and %i triangles" % (len(mesh_points), len(mesh_tris)))
    return mesh_points, mesh_tris


def refineByMaxArea(max_area):
    """Returns a refinement function based on max area, for use with Triangle."""
    def refine(vertices, area):
        """A function for use with watershed_workflow.triangulate.triangulate's refinement_func argument based on a global max area."""
        res = bool(area > max_area)
        # if area < 1.e-5:
        #     raise RuntimeError("TinyTriangle Error")
        return res

    return refine


def refineByRiverDistance(near_distance, near_area, away_distance, away_area, rivers):
    """Returns a graded refinement function based upon a distance function from rivers, for use with Triangle.

    Triangle area must be smaller than near_area when the triangle
    centroid is within near_distance from the river network.
    Area must be smaller than away_area when the triangle
    centroid is at least away_distance from the river network.
    Area must be smaller than a linear interpolant between
    near_area and away_area when between
    near_distance and away_distance from the river
    network.
    """
    def max_area_valid(distance):
        """A function to make sure max area scales with distance from river network
        
        Units in [m]
        """
        if distance > away_distance:
            area = away_area
        elif distance < near_distance:
            area = near_area
        else:
            area = near_area + (distance-near_distance) / (away_distance
                                                           -near_distance) * (away_area-near_area)
        return area

    if type(rivers[0]) == shapely.geometry.Polygon:
        river_multiline = shapely.geometry.MultiPolygon(rivers)
    else:
        river_multiline = shapely.geometry.MultiLineString(
            [r.linestring for river in rivers for r in river])

    def refine(vertices, area):
        """A function for use with watershed_workflow.triangulate.triangulate's refinement_func argument based on size gradation from a river."""
        bary = np.sum(np.array(vertices), axis=0) / 3
        bary_p = shapely.geometry.Point(bary[0], bary[1])
        distance = bary_p.distance(river_multiline)
        max_area = max_area_valid(distance)
        res = bool(area > max_area_valid(distance))
        #logging.debug("refine? area = %g, distance = %g, max_area = %g: refine = %r"%(area,distance,max_area,res))
        return res

    return refine


def refineByPolygons(polygons, areas):
    """Returns a graded refinement function based upon polygon area limits, for use with Triangle.

    Triangle area must be smaller than the area limit for the polygon when the triangle
    centroid is within the polygon.
    """
    def refine(vertices, area):
        """A function for use with watershed_workflow.triangulate.triangulate's refinement_func argument based on polygon area limits."""
        bary = np.sum(np.array(vertices), axis=0) / 3
        bary_p = shapely.geometry.Point(bary[0], bary[1])

        # Check if a single area value is provided
        if isinstance(areas, (int, float)):
            max_area = areas
            for polygon in polygons:
                if polygon.contains(bary_p) and area > max_area:
                    return True
        else:
            # Assume areas is a list of area limits
            for polygon, max_area in zip(polygons, areas):
                if polygon.contains(bary_p) and area > max_area:
                    return True

        return False

    return refine

def refineByStreamTriangles(river_corrs):
    """Returns a refinement function for triangles that have all three vertices on stream mesh."""
    riv_corr = shapely.ops.unary_union(river_corrs).buffer(1)

    def refine(vertices, area):
        return all(riv_corr.intersects(shapely.geometry.Point(p)) for p in vertices)

    return refine


def refineByMaxEdgeLength(edge_length):
    """Returns a refinement function based on max edge length, for use with Triangle."""
    def refine(vertices, area):
        verts4 = np.array([vertices[0], vertices[1], vertices[2], vertices[0]])
        edge_lengths = la.norm(verts4[1:] - verts4[:-1], 2, 1)
        return bool(edge_lengths.max() > edge_length)

    return refine


def splitStreamTriangles(vertices, triangles, river_corrs):
    """Split triangles that have all vertices on stream corridors.
    
    Parameters
    ----------
    vertices : np.ndarray
        Array of mesh vertices/coords
    triangles : list
        List of triangle connectivity/conns
    river_corrs : list
        List of river corridor shapely polygons
        
    Returns
    -------
    vertices : np.ndarray
        Updated vertices array with new split points
    triangles : list
        Updated triangles list with split triangles
    """
    river_buffer = shapely.ops.unary_union(river_corrs).buffer(1)
    
    stream_queue = list(
        _findStreamTriangles(vertices, triangles, river_buffer)
    )
    
    split_triangles = set()

    while stream_queue:
        ti = stream_queue.pop()

        # skip if already split (explicitly or implicitly)
        if ti in split_triangles:
            continue

        vertices, triangles, new_tris, implicitly_split = _splitSingleStreamTriangle(
            vertices, triangles, ti, river_buffer
        )

        # mark all affected triangles as split
        split_triangles.add(ti)
        split_triangles.update(implicitly_split)

        for new_tri in new_tris:
            assert not _isStreamTriangle(vertices, new_tri, river_buffer)

    return vertices, triangles


def _findStreamTriangles(vertices, triangles, river_buffer):
    """Find triangles where all vertices lie on river corridors."""
    stream_tri_indices = []
    
    for ind, tri in enumerate(triangles):
        if _isStreamTriangle(vertices, tri, river_buffer):
            stream_tri_indices.append(ind)
            
    return stream_tri_indices


def _isStreamTriangle(vertices, triangle_conn, river_buffer):
    """Check if a triangle has all vertices on river corridors."""
    triangle_verts = vertices[triangle_conn]
    return all(river_buffer.intersects(shapely.geometry.Point(float(p[0]), float(p[1]))) for p in triangle_verts)


def _splitSingleStreamTriangle(vertices, triangles, triangle_index, river_buffer):
    """Split a single triangle by adding a vertex at the midpoint of an edge."""
    triangle_conn = triangles[triangle_index]
    triangle_verts = vertices[triangle_conn]
    
    # Find the split point and edge
    split_point, edge_index = _findTriangleSplitPoint(triangle_verts, river_buffer)
    
    # Get the edge vertices and find adjacent triangle
    edge_vertices = (triangle_conn[edge_index], triangle_conn[(edge_index + 1) % 3])
    edge_hash = tuple(sorted(edge_vertices))
    adjacent_triangles = _findEdgeSharingTriangleIndices(triangles, edge_hash)
    
    if len(adjacent_triangles) != 2:
        adjacent_triangle_exists = False
    else:
        adjacent_triangle_exists = True
    
    # Add the split point to vertices
    vertices = np.vstack([vertices, np.array([split_point])])
    split_vertex_index = len(vertices) - 1
    
    # Create new triangles for the current triangle
    new_triangles_current = _createSplitTriangles(triangle_conn, edge_index, split_vertex_index)
    
    if adjacent_triangle_exists:
        opposite_triangle_index = next(t for t in adjacent_triangles if t != triangle_index)    
        opposite_triangle_conn = triangles[opposite_triangle_index]
        opposite_edge_index = _findEdgeIndexInTriangle(opposite_triangle_conn, edge_hash)
        
        # Create new triangles for the adjacent triangle
        new_triangles_opposite = _createSplitTriangles(opposite_triangle_conn, opposite_edge_index, split_vertex_index)
        
        # Replace original triangles and add new ones
        triangles[triangle_index] = new_triangles_current[0]
        triangles[opposite_triangle_index] = new_triangles_opposite[0]
        triangles = np.vstack([triangles, [new_triangles_current[1]], [new_triangles_opposite[1]]])
        
        return vertices, triangles, new_triangles_current + new_triangles_opposite, [opposite_triangle_index]
    else:
        # Only split the current triangle (boundary edge case)
        triangles[triangle_index] = new_triangles_current[0]
        triangles = np.vstack([triangles, [new_triangles_current[1]]])
        
        return vertices, triangles, new_triangles_current, []


def _findTriangleSplitPoint(triangle_vertices, river_buffer):
    """Find the point to split a triangle and the edge it lies on."""
    if len(triangle_vertices) != 3:
        raise ValueError("Triangle must have exactly 3 vertices")
    
    # Calculate midpoints for each edge and check if they're off the corridor
    for i in range(3):
        midpoint = watershed_workflow.utils.computeMidpoint(
            triangle_vertices[i], 
            triangle_vertices[(i + 1) % 3]
        )
        midpoint_geom = shapely.geometry.Point(midpoint[0], midpoint[1])
        
        if not river_buffer.intersects(midpoint_geom):
            return midpoint, i
    
    # Fallback: if all midpoints are on corridor, use the first edge
    midpoint = watershed_workflow.utils.computeMidpoint(triangle_vertices[0], triangle_vertices[1])
    return midpoint, 0


def _findEdgeSharingTriangleIndices(triangles, edge_hash):
    """Find triangles that share the same edge."""
    sharing_triangles = []
    
    for i, triangle in enumerate(triangles):
        triangle_edges = [
            tuple(sorted([triangle[j], triangle[(j + 1) % len(triangle)]]))
            for j in range(len(triangle))
        ]
        
        if edge_hash in triangle_edges:
            sharing_triangles.append(i)
    
    return sharing_triangles


def _findEdgeIndexInTriangle(triangle_conn, edge_hash):
    """Find the index of an edge within a triangle's connectivity."""
    for i in range(len(triangle_conn)):
        edge = tuple(sorted([triangle_conn[i], triangle_conn[(i + 1) % len(triangle_conn)]]))
        if edge == edge_hash:
            return i
    raise ValueError("Edge not found in triangle")


def _createSplitTriangles(original_connectivity, edge_index, split_vertex_index):
    """Create two new triangles from splitting along an edge."""
    vertex1_idx = original_connectivity[edge_index]
    vertex2_idx = original_connectivity[(edge_index + 1) % 3]
    opposite_vertex_idx = original_connectivity[(edge_index + 2) % 3]
    
    triangle1 = [opposite_vertex_idx, vertex1_idx, split_vertex_index]
    triangle2 = [split_vertex_index, vertex2_idx, opposite_vertex_idx]
    
    return [triangle1, triangle2]