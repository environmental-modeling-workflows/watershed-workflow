"""Triangulates polygons"""
import logging
import collections
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

import shapely
import meshpy.triangle

import workflow.tree


class Nodes:
    """A collection of nodes that are indexed in the order they are added.

    Note this uses round() for an efficient solution, which is
    potentially fragile if numbers were not generated consistently.
    In this use case, however, it should be safe -- numbers were
    originally rounded (in workflow.conf), and then any function which
    inserted new points or moved points were always careful to ensure
    that all duplicates were always assigned the identical values --
    i.e. all math was done BEFORE assignment.  So duplicates have
    identical representations in floating point.  This suggests we
    shouldn't have to round here at all, but we do anyway to keep
    things cleaner.
    """
    def __init__(self, decimals=7):
        self.decimals = decimals
        self._i = 0
        self._store = collections.OrderedDict()

    def __len__(self):
        """Length is tracked by the counter"""
        return self._i

    def __getitem__(self,key):
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
    return [(inds[i], inds[i+1]) for i in range(len(inds)-1)]
def round_trip_connect(inds):
    """Connect indices in edges in a round-trip fashion"""
    return oneway_trip_connect(inds) + [(inds[-1],inds[0]),]

class NodesEdges:
    """A collection of nodes and edges."""
    def __init__(self, objlist=None):
        self.nodes = Nodes()
        self.edges = list()

        if objlist is not None:
            [self.add(obj) for obj in objlist]

    def add(self, obj):
        """Adds nodes and edges from obj into collection."""
        if type(obj) is shapely.geometry.LineString:
            inds = [self.nodes[c] for c in obj.coords]
            self.edges.extend(oneway_trip_connect(inds))
        elif type(obj) is shapely.geometry.Polygon:
            inds = [self.nodes[c] for c in obj.boundary.coords]
            self.edges.extend(round_trip_connect(inds))
        else:
            raise TypeError("Invalid type for add, %r"%type(obj))


def triangulate(hucs, rivers, **kwargs):
    """Triangulates HUCs and rivers.

    Arguments:
      hucs              | a workflow.hucs.HUCs instance
      rivers            | a list of workflow.tree.Tree instances

    Additional keyword arguments include all options for meshpy.triangle.build()
    """
    logging.info("Triangulating...")
    segments = list(hucs.segments) + list(workflow.tree.forest_to_list(rivers))
    nodes_edges = NodesEdges(segments)

    logging.info("   %i points and %i facets"%(len(nodes_edges.nodes), len(nodes_edges.edges)))

    info = meshpy.triangle.MeshInfo()
    info.set_points(list(nodes_edges.nodes))
    info.set_facets(nodes_edges.edges)
    logging.info("   ...building")
    mesh = meshpy.triangle.build(info, **kwargs)
    logging.info("   ...built")
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)

    logging.info("   %i mesh points and %i triangles"%(len(mesh_points),len(mesh_tris)))
    return mesh_points, mesh_tris


def refine_from_max_area(max_area):
    """Returns a refinement function used with triangulate's refinement_func argument."""
    def refine(vertices, area):
        """A function for use with workflow.triangulate.triangulate's refinement_func argument based on a global max area."""
        print("refine check: area=%g, max_area=%g"%(area, max_area))
        return bool(area > max_area)
    return refine

def refine_from_river_distance(near_distance, near_size, away_distance, away_size, rivers):
    """Returns a graded refinement function based upon a distance function from rivers.

    Typical triangle diameter (size) must be smaller than near_size when the triangle
    centroid is within near_distance from the river network.
    Size must be smaller than away_size when the triangle
    centroid is at least away_distance from the river network.
    Size must be smaller than a linear interpolant between
    near_size and away_size when between
    near_distance and away_distance from the river
    network.
    """
    def max_size_valid(distance):
        """A function to make sure max size scales with distance from river network
        
        Units in [m]
        """
        if distance > away_distance:
            size = away_size
        elif distance < near_distance:
            size = near_size
        else:
            size = near_size + (distance - near_distance) / (away_distance - near_distance) * (away_size - near_size)
        return size**2 / 2

    river_multiline = workflow.tree.forest_to_list(rivers)
    def refine(vertices, area):
        """A function for use with workflow.triangulate.triangulate's refinement_func argument based on size gradation from a river."""
        bary = np.sum(np.array(vertices), axis=0)/3
        bary_p = shapely.geometry.Point(bary[0], bary[1])
        distance = bary_p.distance(river_multiline)
        print("refine check: centroid=%r, distance=%g, maxsize=%g"%(bary, distance, max_size_valid(distance)))
        return bool(area > max_size_valid(distance))

    return refine
