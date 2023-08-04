"""Module for working with tree data structures, built on watershed_workflow.tinytree"""
import logging
import collections
import numpy as np
import copy
import fiona
import pandas as pd
import itertools
from scipy.spatial import cKDTree

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree

_tol = 1.e-7


class River(watershed_workflow.tinytree.Tree):
    """A tree node data structure"""
    def __init__(self, segment=None, children=None):
        """Do not call me.  Instead use the class factory methods, one of:

        - construct_rivers_by_geometry() # generic data
        - construct_rivers_by_hydroseq() # NHDPlus data

        This method initializes a single node in the River,
        representing one reach and its upstream children.

        """
        super(River, self).__init__(children)
        self.segment = segment

        if hasattr(self.segment, 'properties'):
            self.properties = self.segment.properties
        else:
            self.properties = dict()

    def addChild(self, segment):
        """Append a child (upstream) reach to this reach."""
        if type(segment) is River:
            super(River, self).addChild(segment)
        else:
            super(River, self).addChild(type(self)(segment))
        return self.children[-1]

    def removePreserveProperties(self):
        """Removes a reach, preserving catchments and area properties."""
        if self.parent is not None:
            parent = self.parent
            if 'catchment' in self.properties and self.properties['catchment'] is not None:
                if parent.properties['catchment'] is None:
                    parent.properties['catchment'] = self.properties['catchment']
                else:
                    parent.properties['catchment'] = shapely.ops.unary_union(
                        [self.properties['catchment'], parent.properties['catchment']])

            if 'area' in self.properties:
                parent.properties['area'] += self.properties['area']

        self.remove()

    def prune(self):
        """Removes a reach all the way to the leaf node."""
        for node in self.postOrder():
            node.removePreserveProperties()

    def leaf_nodes(self):
        """Generator for all leaves of the tree."""
        for it in self.preOrder():
            if len(it.children) == 0 and it.segment is not None:
                yield it

    def leaves(self):
        """Generator for all leaf reaches of the tree."""
        for n in self.leaf_nodes():
            yield n.segment

    def iter_streamlevel(self):
        """Generator to iterate over all reachs of the same streamlevel.

        streamlevel is a concept defined by NHDPlus attributes, so
        this is only valid if the VAA was loaded.
        """
        yield self
        for c in self.children:
            if c.properties['StreamLevel'] == self.properties['StreamLevel']:
                for r in c.iter_streamlevel():
                    yield r

    def iter_stream_children(self):
        """Find all roots of next-level streams that flow into this reach."""
        for r in self.iter_streamlevel():
            for c in r.children:
                if c.properties['StreamLevel'] > self.properties['StreamLevel']:
                    yield c

    def iter_stream_roots(self):
        """Generator to iterate over all roots of streamlevels."""
        yield self
        for root in self.iter_stream_children():
            for r in root.iter_stream_roots():
                yield r

    def accumulate(self, to_accumulate, to_save=None, op=sum):
        """Accumulates a property across the river tree."""
        val = op(child.accumulate(to_accumulate, to_save, op) for child in self.children)
        val = op([val, self.properties[to_accumulate]])
        if to_save is not None:
            self.properties[to_save] = val
        return val

    def getNode(self, id):
        """return node for a given ID"""
        try:
            node = next(node for node in self.preOrder() if node.properties['ID'] == id)
        except StopIteration:
            node = None
        return node

    def accumulateContributingArea(self, name=None):
        """Form a polygon of all contributing areas based on the 'catchment' property."""
        if name is None:
            name = self.properties['ID']
        
        catch = shapely.ops.unary_union([node.properties['catchment'].buffer(1e-6)
                                         for node in self.preOrder() if node.properties['catchment'] is not None])
        catch.properties = dict()
        catch.properties['outlet_ID'] = self.properties['ID']
        catch.properties['name'] = name
        catch.properties['outlet_point'] = self.segment.coords[-1]
        self.properties['catchment name'] = name
        return catch

    def depthFirst(self):
        """Iterates of reaches in the river in an "upstream-first" or "depth-first" ordering."""
        for node in self.preOrder():
            if node.segment is not None:
                yield node.segment

    def breadthFirst(self):
        """Iterates of reaches in the river in an "breadth-first" ordering."""
        for node in self.breadthFirstOrder():
            if node.segment is not None:
                yield node.segment

    def __iter__(self):
        return self.depthFirst()

    def _is_continuous(self, child, tol=_tol):
        """Is a given child continuous with self?"""
        return watershed_workflow.utils.close(child.segment.coords[-1], self.segment.coords[0], tol)

    def is_locally_continuous(self, tol=_tol):
        """Is this node continuous with its parent and children?"""
        res = all(self._is_continuous(child) for child in self.children)
        if self.parent is not None:
            res = res and self.parent._is_continuous(self)
        return res

    def is_continuous(self, tol=_tol):
        """Checks geometric continuity of the river.

        Confirms that all upstream children's downstream coordinate
        coincides with self's upstream coordinate.
        """
        return all(self._is_continuous(child, tol) for child in self.children) and \
            all(child.is_continuous(tol) for child in self.children)

    def _make_continuous(self, child):
        child_coords = list(child.segment.coords)
        child_coords[-1] = list(self.segment.coords)[0]
        child.segment = shapely.geometry.LineString(child_coords)

    def make_continuous(self, tol=_tol):
        """Sometimes there can be small gaps between segments of river tree if river is constructed using
        HydrologicSequence and Snap option is not used. Here we make them consistent"""
        for node in self.preOrder():
            for child in node.children:
                if not node._is_continuous(child, tol):
                    node._make_continuous(child)
        assert (self.is_continuous())

    def is_hydroseq_consistent(self):
        """Confirms that hydrosequence is valid."""
        if len(self.children) == 0:
            return True

        self.children = sorted(self.children, key=lambda c: c.properties['HydrologicSequence'])
        return self.properties['HydrologicSequence'] < self.children[0].properties['HydrologicSequence'] and \
            all(child.is_hydroseq_consistent() for child in self.children)

    def is_consistent(self, tol=_tol):
        """Validity checking of the tree."""
        good = self.is_continuous(tol)
        if 'HydrologicSequence' in self.properties:
            good |= self.is_hydroseq_consistent()
        return good

    #
    # Factory functions
    #
    @classmethod
    def construct_rivers_by_geometry(cls, reaches, tol=_tol):
        """Forms a list of River trees from a list of reaches by looking for
        close endpoints of those reaches.

        Note that this expects that endpoints of a reach coincide with
        beginpoints of their downstream reach, and does not work for
        cases where the junction is at a midpoint of a reach.
        """
        logging.debug("Generating Rivers")

        if len(reaches) == 0:
            return list()

        # make a kdtree of beginpoints
        coords = np.array([r.coords[0] for r in reaches])
        kdtree = cKDTree(coords)

        # make a node for each segment
        nodes = [cls(r) for r in reaches]

        # match nodes to their parent through the kdtree
        rivers = []
        divergence = []
        divergence_matches = []
        for j, n in enumerate(nodes):
            # find the closest beginpoint the this node's endpoint
            closest = kdtree.query_ball_point(n.segment.coords[-1], tol)
            if len(closest) > 1:
                logging.debug("Bad multi segment:")
                logging.debug(" connected to %d: %r" % (j, list(n.segment.coords[-1])))
                divergence.append(j)
                divergence_matches.append(closest)

                # end at the same point, pick the min angle deviation
                my_tan = np.array(n.segment.coords[-1]) - np.array(n.segment.coords[-2])
                my_tan = my_tan / np.linalg.norm(my_tan)

                other_tans = [
                    np.array(reaches[c].coords[1]) - np.array(reaches[c].coords[0]) for c in closest
                ]
                other_tans = [ot / np.linalg.norm(ot) for ot in other_tans]
                dots = [np.inner(ot, my_tan) for ot in other_tans]
                for i, c in enumerate(closest):
                    logging.debug("  %d: %r --> %r with dot product = %g" %
                                  (c, coords[c], reaches[c].coords[-1], dots[i]))
                c = closest[np.argmax(dots)]
                nodes[c].addChild(n)

            elif len(closest) == 0:
                rivers.append(n)
            else:
                assert (len(closest) == 1)
                nodes[closest[0]].addChild(n)

        assert (len(rivers) > 0)
        return rivers

    @classmethod
    def construct_rivers_by_hydroseq(cls, segments):
        """Given a list of segments, create a list of rivers using the
        HydroSeq maps provided in NHDPlus datasets.
        """
        # create a map of all segments from HydroSeqID to segment
        hydro_seq_ids = dict((seg.properties['HydrologicSequence'], cls(seg)) for seg in segments)

        roots = []
        for hs_id, node in hydro_seq_ids.items():
            down_hs_id = node.properties['DownstreamMainPathHydroSeq']
            try:
                hydro_seq_ids[down_hs_id].addChild(node)
            except KeyError:
                roots.append(node)
        return roots

    def deep_copy(self):
        """Creates a copy of self  including properties"""
        cp = copy.deepcopy(self)
        for node1, node2 in zip(cp.preOrder(), self.preOrder()):
            node1.properties = copy.deepcopy(node2.properties)
        return cp


def accumulateContributingArea(rivers, outlet_IDs, names=None):
    """Given a list of outlet_IDs, find the reach in rivers and form its contributing area.
    
    Parameters:
    -----------
    rivers: list(watershed_workflow.river_tree.RiverTree)
      Rivers from which outlet reaches are potentially from 
    outlet_IDs: list(str)
      List of IDs of the outlet reaches
    names: list(str), optional
      Names for the catchments

    Returns
    -------
    list(river_tree.RiverTree)
      The trunks of the rivers at outlet_IDs
    list(shapely.geometry.Polygon)
      The contributing areas to those trunks.

    """
    if names is None:
        names = outlet_IDs

    roots = []
    catchments = []
    for id, name in zip(outlet_IDs, names):
        found = False
        for river in rivers:
            root = river.getNode(id)
            if root is not None:
                if found:
                    raise RuntimeError(f'accumulateContributingArea: outlet_ID {outlet_ID} appears in more than one river')
                roots.append(root)
                catchments.append(root.accumulateContributingArea(name))
                found = True
    return roots, catchments
