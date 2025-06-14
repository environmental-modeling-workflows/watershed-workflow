"""Triangulates polygons"""
from typing import Tuple, List, Optional, Set

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
    def __init__(self, decimals : int = 3):
        self.decimals = decimals
        self._i = 0
        self._store : collections.OrderedDict[Tuple[float, ...], int] = collections.OrderedDict()

    def __len__(self) -> int:
        """Length is tracked by the counter"""
        return self._i

    def __getitem__(self, key : Tuple[float, ...]) -> int:
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


def connectOnewayTrip(inds : List[int]) -> List[Tuple[int, int]]:
    """Connect indices in edges in a oneway fashion"""
    return [(inds[i], inds[i + 1]) for i in range(len(inds) - 1)]


def connectRoundTrip(inds : List[int]) -> List[Tuple[int, int]]:
    """Connect indices in edges in a round-trip fashion"""
    return connectOnewayTrip(inds) + [(inds[-1], inds[0]), ]


def orient(e : Tuple[int, int]) -> Tuple[int, int] | None:
    if e[0] > e[1]:
        return e[1], e[0]
    elif e[0] < e[1]:
        return e[0], e[1]
    else:
        return None


class NodesEdges:
    """A collection of nodes and edges."""
    def __init__(self,
                 objlist : Optional[shapely.geometry.LineString | shapely.geometry.Polygon] = None):
        self.nodes = Nodes()
        self.edges : Set[Tuple[int,int]] = set()

        if objlist != None:
            for obj in objlist:
                self.add(obj)

    def add(self, obj : shapely.geometry.LineString | shapely.geometry.Polygon) -> None:
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

    def check(self, tol : float) -> None:
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


def triangulate(hucs : watershed_workflow.split_hucs.SplitHUCs,
                internal_boundaries : Optional[List[shapely.geometry.LineString]] = None,
                hole_points : Optional[List[shapely.geometry.Point]] = None,
                tol : float = 1.0,
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
    tol : float, optional
        Set tolerance for minimum distance between two nodes. The unit
        is the same as that of the watershed's CRS. The default is 1.

    Additional keyword arguments include all options for meshpy.triangle.build()

    """
    import meshpy.triangle

    logging.info("Triangulating...")

    # get a list of required boundaries, both interior and exterior
    linestrings : List[shapely.geometry.LineStrings] = []

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
        river_multiline = shapely.geometry.MultiLineString([r.linestring for river in rivers for r in river])

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


def refineByMaxEdgeLength(edge_length):
    """Returns a refinement function based on max edge length, for use with Triangle."""
    def refine(vertices, area):
        verts4 = np.array([vertices[0], vertices[1], vertices[2], vertices[0]])
        edge_lengths = la.norm(verts4[1:] - verts4[:-1], 2, 1)
        return bool(edge_lengths.max() > edge_length)

    return refine


