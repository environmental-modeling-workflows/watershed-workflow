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

    def getSegment(self):
        """Used if one needs segments with properties."""
        self.segment.properties = self.properties
        return self.segment

    def prune(self, preserve_catchments=False):
        """Removes a reach all the way to the leaf node."""
        
        if self.parent is None:
            raise ValueError("Cannot prune a branch with no parent.")

        if preserve_catchments:
            parent = self.parent

            # accumulate the sub-catchments
            catches = [n.properties['catchment'] for n in self.preOrder()
                       if 'catchment' in n.properties and n.properties['catchment'] is not None]
            if len(catches) > 0:
                ca = shapely.ops.unary_union(catches)
            else:
                ca = None
                
            if ca is not None:
                if 'catchment' not in parent.properties or parent.properties['catchment'] is None:
                    parent.properties['catchment'] = ca
                else:
                    parent.properties['catchment'] = shapely.ops.unary_union(
                        [ca, parent.properties['catchment']])

            # accumulate the area
            if 'area' in self.properties:
                total_area = sum(r.properties['area'] for r in self.preOrder())
                parent.properties['area'] += total_area

        self.remove()

    def split(self, i):
        """Split the reach at the ith coordinate of the segment.

        Note that this could, but does not, split the catchment!

        Returns upstream_node, downstream_node
        """
        if i < 0:
            i = len(self.segment.coords) + i
        assert(i > 0 and i < len(self.segment.coords)-1)

        upstream_segment = shapely.geometry.LineString(list(self.segment.coords)[0:i+1])
        downstream_segment = shapely.geometry.LineString(list(self.segment.coords)[i:])
        downstream_area_frac = downstream_segment.length / self.segment.length

        # fix properties
        downstream_props = copy.deepcopy(self.properties)
        if 'AreaSqKm' in downstream_props:
            downstream_props['AreaSqKm'] = self.properties['AreaSqKm'] * downstream_area_frac
            self.properties['AreaSqKm'] = self.properties['AreaSqKm'] * (1 - downstream_area_frac)
        if 'HydrologicSequence' in downstream_props:
            downstream_props['HydrologicSequence'] -= 0.5
            self.properties['DownstreamMainPathHydroSeq'] = downstream_props['HydrologicSequence']
            
        ID = self.properties['ID']
        downstream_props['ID'] = ID+'a'
        self.properties['ID'] = ID+'b'

        if 'DivergenceCode' in downstream_props:
            downstream_props['DivergenceCode'] = 0

        # fix self
        self.segment = upstream_segment
        parent = self.parent
        self.remove()

        # new node
        downstream_segment.properties = downstream_props
        new_node = self.__class__(downstream_segment, [self,])
        parent.addChild(new_node)
        return self, new_node

    def merge(self):
        """Merges this with its parent."""
        assert(len(list(self.siblings())) == 0)

        # fix properties
        if 'areasqkm' in self.properties:
            self.parent.properties['areasqkm'] += self.properties['areasqkm']
        if 'AreaSqKm' in self.properties:
            self.parent.properties['AreaSqKm'] += self.properties['AreaSqKm']
        if 'catchment' in self.properties and self.properties['catchment'] is not None:
            if self.parent.properties['catchment'] is None:
                self.parent.properties['catchment'] = self.properties['catchment']
            else:
                self.parent.properties['catchment'] = shapely.ops.unary_union([self.properties['catchment'],
                                                                               self.parent.properties['catchment']])
        if 'HydrologicSequence' in self.parent.properties:
            self.parent.properties['HydrologicSequence'] = self.properties['HydrologicSequence']

        if 'DivergenceCode' in self.parent.properties:
            self.parent.properties['DivergenceCode'] = self.properties['DivergenceCode']

        parent = self.parent
   
        self.remove()
        for child in self.children:
            parent.addChild(child)

    def moveCoordinate(self, i, xy):
        """Moves the ith coordinate of self.segment to a new location."""
        if i < 0:
            i = len(self.segment.coords) + i
        coords = list(self.segment.coords)
        coords[i] = xy
        self.segment = shapely.geometry.LineString(coords)

    def insertCoordinate(self, i, xy):
        """Inserts a new coordinate before the ith coordinate."""
        if i < 0:
            i = len(self.segment.coords) + i
        coords = list(self.segment.coords)
        coords.insert(i, xy)
        self.segment = shapely.geometry.LineString(coords)

    def appendCoordinate(self, xy):
        """Appends a coordinate at the end (downstream) of the segment."""
        coords = list(self.segment.coords) + [xy,]
        self.segment = shapely.geometry.LineString(coords)

    def extendCoordinates(self, xys):
        """Appends multiple coordinates at the end (downstream) of the segment."""
        coords = list(self.segment.coords) + xys
        self.segment = shapely.geometry.LineString(coords)

    def prependCoordinates(self, xys):
        """Prepends multiple coordinates at the beginning (upstream) of the segment."""
        coords = xys + list(self.segment.coords)
        self.segment = shapely.geometry.LineString(coords)

    def popCoordinate(self, i):
        """Removes the ith coordinate and returns its value."""
        coords = list(self.segment.coords)
        coords.pop(i)
        self.segment = shapely.geometry.LineString(coords)
        
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

    def accumulateCatchments(self, name=None):
        """Form a polygon of all contributing areas based on the 'catchment' property."""
        if name is None:
            name = self.properties['ID']
        
        catch = shapely.ops.unary_union([node.properties['catchment']
                                         for node in self.preOrder() if node.properties['catchment'] is not None])
        catch.properties = dict()
        catch.properties['outlet_ID'] = self.properties['ID']
        catch.properties['ID'] = 'CA_'+self.properties['ID']
        catch.properties['name'] = name
        catch.properties['outlet_point'] = self.segment.coords[-1]
        self.properties['contributing area name'] = name
        return catch

    def depthFirst(self):
        """Iterates of reaches in the river in an "depth-first" ordering."""
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
        res = all(self._is_continuous(child, tol=_tol) for child in self.children)
        if self.parent is not None:
            res = res and self.parent._is_continuous(self, tol=_tol)
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
            good &= self.is_hydroseq_consistent()
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

    def deepcopy(self):
        """Creates a copy of self  including properties"""
        cp = copy.deepcopy(self)
        for node1, node2 in zip(cp.preOrder(), self.preOrder()):
            node1.properties = copy.deepcopy(node2.properties)
        return cp


def getNode(rivers, nid):
    """Finds the node, by ID, in a list of rivers"""
    for river in rivers:
        n = river.getNode(nid)
        if n is not None:
            return n
    return None


def combineSiblings(n1, n2):
    """Combines two sibling nodes, merging catchments and metadata."""
    assert(n1.isSiblingOf(n2))
    beginpoint = (np.array(n1.segment.coords[0]) + np.array(n2.segment.coords[0]))/2
    endpoint = (np.array(n1.segment.coords[-1]) + np.array(n2.segment.coords[-1]))/2
    new_seg = shapely.geometry.LineString([beginpoint, endpoint])
    n1.segment = new_seg

    if 'AreaSqKm' in n1.properties:
        n1.properties['AreaSqKm'] += n2.properties['AreaSqKm']

    if 'catchment' in n2.properties and n2.properties['catchment'] is not None:
        if n1.properties['catchment'] is None:
            n1.properties['catchment'] = n2.properties['catchment']
        else:
            n1.properties['catchment'] = shapely.ops.unary_union([n1.properties['catchment'], n2.properties['catchment']])

    for child in n2.children:
        if 'DownstreamMainPathHydroSeq' in child.properties:
            child.properties['DownstreamMainPathHydroSeq'] = n1.properties['HydrologicSequence']
        n1.addChild(child)
    n2.children = []
    n2.remove()

    for child in n1.children:
        child.moveCoordinate(-1, n1.segment.coords[0])
    
    return n1


def accumulateCatchments(rivers, outlet_IDs, names=None):
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
                    raise RuntimeError(f'accumulateCatchments: outlet_ID {outlet_ID} appears in more than one river')
                roots.append(root)
                catchments.append(root.accumulateCatchments(name))
                found = True
    return roots, catchments


def accumulateIncrementalCatchments(rivers, outlet_IDs, names=None):
    """Given a list of outlet_IDs, form the incremental contributing areas.
    
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

    roots = [getNode(rivers, out_id) for out_id in outlet_IDs]
    assert(all(root is not None for root in roots))

    sorted_ids = sorted(outlet_IDs)
    def truncated_tree_iter(n):
        yield n
        for c in n.children:
            if c.properties['ID'] not in sorted_ids:
                for nn in truncated_tree_iter(c):
                    yield nn

    incremental_cas = [shapely.ops.unary_union([n.properties['catchment'] for n in truncated_tree_iter(root) if n.properties['catchment'] is not None]) for root in roots]

    for catch, root, name in zip(incremental_cas, roots, names):
        catch.properties = dict()
        catch.properties['outlet_ID'] = root.properties['ID']
        catch.properties['ID'] = 'CA_'+root.properties['ID']
        catch.properties['name'] = name
        catch.properties['outlet_point'] = root.segment.coords[-1]
        root.properties['incremental contributing area name'] = name

    return roots, incremental_cas
