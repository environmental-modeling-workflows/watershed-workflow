"""Module for working with tree data structures, built on watershed_workflow.tinytree"""
import logging
import collections
import numpy as np
import itertools

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree


_tol = 1.e-7

class RiverTree(watershed_workflow.tinytree.Tree):
    """A tree node data structure"""
    def __init__(self, segment=None, properties=None, children=None):
        super(RiverTree, self).__init__(children)
        self.segment = segment

        assert(properties is None)
        # assert(hasattr(self.segment, 'properties'))
        if properties is not None:
            self.properties = properties
        elif hasattr(self.segment, 'properties'):
            self.properties = self.segment.properties
        else:
            self.properties = dict()

    def addChild(self, segment):
        if type(segment) is RiverTree:
            super(RiverTree,self).addChild(segment)
        else:
            super(RiverTree,self).addChild(type(self)(segment))
        return self.children[-1]

    def dfs(self):
        for node in self.preOrder():
            if node.segment is not None:
                yield node.segment

    def leaf_nodes(self):
        """Generator for all leaves of the tree."""
        for it in self.preOrder():
            if len(it.children) == 0 and it.segment is not None:
                yield it

    def leaves(self):
        """Generator for all leaves of the tree."""
        for n in self.leaf_nodes():
            yield n.segment

    def __len__(self):
        # kinda hacky way of getting the count
        return sum(1 for i in self.dfs())

    def __iter__(self):
        return self.dfs()

    def check_child_consistency(self, tol=1.e-8):
        for child in self.children:
            if not watershed_workflow.utils.close(child.segment.coords[-1], self.segment.coords[0]):
                return False
        return True

    def get_inconsistent(self, tol=1.e-8):
        inconsistent = []
        for child in self.children:
            if not watershed_workflow.utils.close(child.segment.coords[-1], self.segment.coords[0]):
                logging.warning("  INCONSISTENT:")
                logging.warning("    child: %r"%(child.segment.coords[:]))
                logging.warning("    parent: %r"%(self.segment.coords[:]))
                inconsistent.append(child)
        return inconsistent
    
            
def _get_matches(seg, segments, segment_found):
    """Find segments attached to seg amongst those not already found"""
    matches = [i for i in range(len(segments)) if not segment_found[i]
               and watershed_workflow.utils.close(segments[i].coords[-1], seg.coords[0])]
    segment_found[matches] = True
    return matches

def _go(i_seg, tree, segments, segments_found):
    """Recursive helper function for generating a tree based on a matching function."""
    count = 0
    tree.addChild(segments[i_seg])
    for m in _get_matches(segments[i_seg], segments, segments_found):
        count += _go(m, tree, segments, segments_found)
    return count + 1

def make_trees(segments):
    """Forms tree(s) from a list of segments."""
    logging.debug("Generating trees")
    endpoint_indices = find_endpoints(segments)
    logging.debug("  found: %i outlets"%len(endpoint_indices))
    for endp in endpoint_indices:
        logging.debug("    at: %r"%list(segments[endp].coords[-1]))

    # check if any endpoint lives on another segment
    segs_to_remove = []
    segs_to_add = []
    for endpoint_index in endpoint_indices:
        endpoint_seg = shapely.geometry.LineString(segments[endpoint_index].coords[-2:])
        try:
            inter = next(i for i,seg in enumerate(segments)
                         if endpoint_seg.intersects(seg)
                         and i != endpoint_index
                         and watershed_workflow.utils.close(endpoint_seg.intersection(seg).coords[0], endpoint_seg.coords[-1], 1.e-5))
        except StopIteration:
            logging.debug("   outlet %i is not faux"%endpoint_index)
        else:
            logging.debug("   faux outlet: %i segment: %i"%(endpoint_index, inter))
            segs_to_remove.append(inter)
            
            print("splitting segment: %r"%list(segments[inter].coords))
            print("   at: %r"%list(segments[endpoint_index].coords[-1]))
            segs_to_add.extend(watershed_workflow.utils.cut(segments[inter], endpoint_seg))
            
    if len(segs_to_remove) != 0:
        segments = list(segments)
        for i in sorted(segs_to_remove, reverse=True):
            segments.pop(i)
        segments.extend(segs_to_add)
        segments = shapely.geometry.MultiLineString(segments)

        # regenerate endpoints (indicies may have changed)
        endpoint_indices = find_endpoints(segments)
        logging.debug("  found: %i outlets"%len(endpoint_indices))

    # generate all trees
    segment_found = np.zeros((len(segments),), bool)
    gcount = 0
    trees = []
    for endpoint_index in endpoint_indices:
        tree = RiverTree()
        gcount += _go(endpoint_index, tree, segments, segment_found)
        trees.append(tree)
    assert(gcount == len(segments))
    return trees

def find_endpoints(segments):
    """Finds a list of indices of all segments whose endpoint is not a beginpoint.

    Note these may truely be the tree root, or they may end at a
    midpoint on another segment (mistake in input data).
    """
    endpoints = []
    for i,s in enumerate(segments):
        c = s.coords[-1]
        try:
            next(s2 for s2 in segments if watershed_workflow.utils.close(s2.coords[0], c))
        except StopIteration:
            endpoints.append(i)
    return endpoints

def tree_to_list(tree):
    return shapely.geometry.MultiLineString(list(tree.dfs()))

def forest_to_list(forest):
    """A forest is a list of trees.  Returns a flattened list of trees."""
    return shapely.geometry.MultiLineString([r for tree in forest for r in tree.dfs()])

def is_consistent(tree, tol=1.e-8):
    """Checks the geometric consistency of the tree."""
    return not any(not node.check_child_consistency(tol) for node in tree.preOrder())

def get_inconsistent(tree, tol=1.e-8):
    """Gets a list of inconsistent nodes of the tree."""
    return [n for node in tree.preOrder() for n in node.get_inconsistent(tol)]
