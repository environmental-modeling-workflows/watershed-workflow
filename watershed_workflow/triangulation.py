"""Triangulates polygons"""
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
    def __init__(self, decimals=3):
        self.decimals = decimals
        self._i = 0
        self._store = collections.OrderedDict()

    def __len__(self):
        """Length is tracked by the counter"""
        return self._i

    def __getitem__(self, key):
        """Get item index, setting if default."""
        # Note this implementation could be optimized, and should be
        # done so assuming keys are NOT there, which is the most
        # likely case.
        key = tuple(round(p, self.decimals) for p in key)
        if key not in self._store:
            self._store[key] = self._i
            self._i += 1
        return self._store[key]

    def __iter__(self):
        """Iterable collection"""
        for k in self._store.keys():
            yield k


def oneway_trip_connect(inds):
    """Connect indices in edges in a oneway fashion"""
    return [(inds[i], inds[i + 1]) for i in range(len(inds) - 1)]


def round_trip_connect(inds):
    """Connect indices in edges in a round-trip fashion"""
    return oneway_trip_connect(inds) + [(inds[-1], inds[0]), ]


def orient(e):
    if e[0] > e[1]:
        return e[1], e[0]
    elif e[0] < e[1]:
        return e[0], e[1]
    else:
        return None


class NodesEdges:
    """A collection of nodes and edges."""
    def __init__(self, objlist=None):
        self.nodes = Nodes()
        self.edges = set()

        if objlist != None:
            [self.add(obj) for obj in objlist]

    def add(self, obj):
        """Adds nodes and edges from obj into collection."""
        if type(obj) is shapely.geometry.LineString:
            inds = [self.nodes[c] for c in obj.coords]
            [self.edges.add(orient(e)) for e in oneway_trip_connect(inds) if orient(e)]
        elif type(obj) is shapely.geometry.Polygon:
            inds = [self.nodes[c] for c in obj.boundary.coords]
            [self.edges.add(orient(e)) for e in round_trip_connect(inds) if orient(e)]
        else:
            raise TypeError("Invalid type for add, %r" % type(obj))

    def check(self, tol):
        """Checks consistency of the interal representation."""
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


def triangulate(hucs,
                river_corrs=None,
                internal_boundaries=None,
                hole_points=None,
                additional_points=None,
                tol=1,
                **kwargs):
    """Triangulates HUCs and rivers.

    Note, refinement of a given triangle is done if any of the provided
    criteria is met.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs()
    river_corrs : list(shapely.geometry.Polygons), optional
        A list of river corridor polygons for each river
    refinement_polygon : list, optional
        List of RiverTrees or other iterable collections of
        coordinates used to refine the mesh given the distance
        function refinement.
    internal_boundaries : list, optional
        List of shapely objects or RiverTrees or other iterable
        collections of coordinates used as internal boundaries that
        must be included in the mesh.
    hole_points : list(shapely.Point), optional
        List of points inside the polygons to be left as holes/voids (excluded from mesh).
    tol : float, optional
        Set tolerance for minimum distance between two nodes. The unit
        is the same as that of the watershed's CRS. The default is 1.

    Additional keyword arguments include all options for meshpy.triangle.build()

    """
    import meshpy.triangle

    logging.info("Triangulating...")
    segments = list(hucs.segments)

    if internal_boundaries != None:
        if type(internal_boundaries) is list:
            for item in internal_boundaries:
                if isinstance(item, shapely.geometry.LineString) or isinstance(
                        item, shapely.geometry.Polygon):
                    segments = [item, ] + segments
                elif isinstance(item, watershed_workflow.river_tree.River):
                    segments = list(item) + segments
        elif isinstance(internal_boundaries, shapely.geometry.Polygon):
            segments = [internal_boundaries, ] + segments

    if river_corrs != None:
        if type(river_corrs) is list:
            segments = river_corrs + segments
        elif type(river_corrs) is shapely.geometry.Polygon:
            segments = [river_corrs, ] + segments
        else:
            raise RuntimeError("Triangulate not implemented for container of type '%r'"
                               % type(hucs))

    nodes_edges = NodesEdges(segments)

    logging.info("   %i points and %i facets" % (len(nodes_edges.nodes), len(nodes_edges.edges)))
    nodes_edges.check(tol=tol)

    logging.info(" building graph data structures")
    info = meshpy.triangle.MeshInfo()
    nodes = np.array(list(nodes_edges.nodes), dtype=np.float64)

    # Add additional points if provided
    if additional_points is not None:
        nodes = np.vstack((nodes, additional_points))

    pdata = [tuple([float(c) for c in p]) for p in nodes]
    info.set_points(pdata)
    fdata = [[int(i) for i in f] for f in nodes_edges.edges]
    info.set_facets(fdata)

    if river_corrs is not None:
        # adding hole in the river corridor for quad elements
        logging.info("defining hole..")
        rc_hole_points = []
        for river_corr in river_corrs:
            rc_hole_point = pick_hole_point(river_corr)
            rc_hole_points.append(rc_hole_point.coords[0])  # a point inside the river corridor
            assert (river_corr.contains(rc_hole_point))
        if hole_points != None:
            hole_points = hole_points + rc_hole_points
        else:
            hole_points = rc_hole_points
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


def refine_from_max_area(max_area):
    """Returns a refinement function based on max area, for use with Triangle."""
    def refine(vertices, area):
        """A function for use with watershed_workflow.triangulate.triangulate's refinement_func argument based on a global max area."""
        res = bool(area > max_area)
        # if area < 1.e-5:
        #     raise RuntimeError("TinyTriangle Error")
        return res

    return refine


def refine_from_river_distance(near_distance, near_area, away_distance, away_area, rivers):
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
        river_multiline = shapely.geometry.MultiLineString([r for river in rivers for r in river])

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


def refine_from_polygons(polygons, areas):
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


def refine_from_max_edge_length(edge_length):
    """Returns a refinement function based on max edge length, for use with Triangle."""
    def refine(vertices, area):
        verts4 = np.array([vertices[0], vertices[1], vertices[2], vertices[0]])
        edge_lengths = la.norm(verts4[1:] - verts4[:-1], 2, 1)
        return bool(edge_lengths.max() > edge_length)

    return refine


def refine_stream_triangles(river_corrs):
    """Returns a refinement function for triangles that have all three vertices on stream mesh."""
    riv_corr = shapely.ops.unary_union(river_corrs).buffer(1)

    def refine(vertices, area):
        return all(riv_corr.intersects(shapely.geometry.Point(p)) for p in vertices)

    return refine


def pick_hole_point(poly):
    """A function to pick a point inside a polygon"""
    nodes_edges_rc = NodesEdges([poly])
    p1 = list(nodes_edges_rc.nodes)[0]
    p2 = list(nodes_edges_rc.nodes)[-1]
    p3 = list(nodes_edges_rc.nodes)[1]
    p4 = list(nodes_edges_rc.nodes)[-2]

    return shapely.geometry.MultiPoint([p1, p2, p3, p4]).centroid
