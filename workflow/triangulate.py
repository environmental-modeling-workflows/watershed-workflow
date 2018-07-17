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
    def __init__(self, decimals=3):
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
def orient(e):
    if e[0] > e[1]:
        return e[1],e[0]
    elif e[0] < e[1]:
        return e[0],e[1]
    else:
        raise RuntimeError("self-edge!")


class NodesEdges:
    """A collection of nodes and edges."""
    def __init__(self, objlist=None):
        self.nodes = Nodes()
        self.edges = set()

        if objlist is not None:
            [self.add(obj) for obj in objlist]

    def add(self, obj):
        """Adds nodes and edges from obj into collection."""
        if type(obj) is shapely.geometry.LineString:
            inds = [self.nodes[c] for c in obj.coords]
            [self.edges.add(orient(e)) for e in oneway_trip_connect(inds)]
        elif type(obj) is shapely.geometry.Polygon:
            inds = [self.nodes[c] for c in obj.boundary.coords]
            [self.edges.add(orient(e)) for e in round_trip_connect(inds)]
        else:
            raise TypeError("Invalid type for add, %r"%type(obj))

    def check(self, tol=0.1):
        """Checks consistency of the interal representation."""
        logging.info("checking graph consistency")
        min_dist = 1.e10
        nodes = list(self.nodes)
        for i in range(len(nodes)):
            for j in range(i):
                dist = workflow.utils.distance(nodes[i], nodes[j])
                min_dist = min(min_dist, dist)
                assert(dist > tol)
        logging.info("  min internal nodal distance = %g"%min_dist)
        
        min_node = min(self.nodes[n] for n in self.nodes)
        max_node = max(self.nodes[n] for n in self.nodes)
        logging.info("  min/max nodal index = %i, %i out of %i"%(min_node, max_node, len(self.nodes)))
        
        min_edge_node = min(n for e in self.edges for n in e)
        max_edge_node = max(n for e in self.edges for n in e)
        logging.info("  min/max edge-nodal index = %i, %i out of %i"%(min_edge_node, max_edge_node, len(self.nodes)))
        assert(min_edge_node == 0)
        assert(max_edge_node == len(self.nodes)-1)

        
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
    nodes_edges.check(tol=1)
    
    info = meshpy.triangle.MeshInfo()
    nodes = np.array(list(nodes_edges.nodes), dtype=np.float64)
    centroid = np.round(np.mean(nodes, axis=0),3)
    shifted_nodes = nodes - np.expand_dims(centroid,0)
    #np.savetxt("points.txt", shifted_nodes)
    #np.savetxt("facets.txt", np.array(list(nodes_edges.edges),dtype=np.int32))
    
    pdata = [tuple([float(c) for c in p]) for p in shifted_nodes]
    info.set_points(pdata)
    fdata = [[int(i) for i in f] for f in nodes_edges.edges]
    info.set_facets(fdata)

    # plt.figure()
    # for e in fdata:
    #     plt.plot([pdata[e[0]][0], pdata[e[1]][0]],
    #              [pdata[e[0]][1], pdata[e[1]][1]], '-', color='gray')
    # plt.scatter([p[0] for p in pdata], [p[1] for p in pdata],marker='+')
    # plt.show()
    
    logging.info("   ...building")
    mesh = meshpy.triangle.build(info, **kwargs)
    logging.info("   ...built")
    mesh_points = np.array(mesh.points) + np.expand_dims(centroid,0)
    mesh_tris = np.array(mesh.elements)
    logging.info("   %i mesh points and %i triangles"%(len(mesh_points),len(mesh_tris)))
    return mesh_points, mesh_tris


def refine_from_max_area(max_area):
    """Returns a refinement function used with triangulate's refinement_func argument."""
    def refine(vertices, area):
        """A function for use with workflow.triangulate.triangulate's refinement_func argument based on a global max area."""
        res = bool(area > max_area)
        if area < 1.e-5:
            raise RuntimeError("bah")
        return res
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
        return bool(area > max_size_valid(distance))

    return refine
