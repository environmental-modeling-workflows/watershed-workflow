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
            super(River,self).addChild(segment)
        else:
            super(River,self).addChild(type(self)(segment))
        return self.children[-1]

    def dfs(self):
        """Iterates of reaches in the river in an "upstream-first" or "depth-first" ordering."""
        for node in self.preOrder():
            if node.segment is not None:
                yield node.segment

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
            
    def __len__(self):
        """Number of total reaches in the river."""
        return sum(1 for i in self.dfs())

    def __iter__(self):
        return self.dfs()

    def _is_continuous(self, child, tol=_tol):
        """Is a given child continuous with self.."""
        return watershed_workflow.utils.close(child.segment.coords[-1], self.segment.coords[0], tol)
    
    def is_continuous(self, tol=_tol):
        """Checks geometric continuity of the river.

        Confirms that all upstream children's downstream coordinate
        coincides with self's upstream coordinate.
        """
        return all(self._is_continuous(child, tol) for child in self.children) and \
            all(child.is_continuous(tol) for child in self.children)
    
    def _make_continuous(self, child):
        child_coords=list(child.segment.coords)
        child_coords[-1]=list(self.segment.coords)[0]
        child.segment=shapely.geometry.LineString(child_coords)

    def make_continuous(self, tol=_tol):
        """Sometimes there can be small gaps between segments of river tree if river is constructed using
        HydrologicSequence and Snap option is not used. Here we make them consistent"""
        for node in self.preOrder():
            for child in node.children:
                if not node._is_continuous(child, tol):
                    node._make_continuous(child)
        assert(self.is_continuous())

    def is_hydroseq_consistent(self):
        """Confirms that hydrosequence is valid."""
        if len(self.children) == 0:
            return True
        
        self.children = sorted(self.children, key=lambda c : c.properties['HydrologicSequence'])
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
        for j,n in enumerate(nodes):
            # find the closest beginpoint the this node's endpoint
            closest = kdtree.query_ball_point(n.segment.coords[-1], tol)
            if len(closest) > 1:
                logging.debug("Bad multi segment:")
                logging.debug(" connected to %d: %r"%(j,list(n.segment.coords[-1])))
                divergence.append(j)
                divergence_matches.append(closest)

                # end at the same point, pick the min angle deviation
                my_tan = np.array(n.segment.coords[-1]) - np.array(n.segment.coords[-2])
                my_tan = my_tan / np.linalg.norm(my_tan)

                other_tans = [np.array(reaches[c].coords[1]) - np.array(reaches[c].coords[0]) for c in closest]
                other_tans = [ot/np.linalg.norm(ot) for ot in other_tans]
                dots = [np.inner(ot,my_tan) for ot in other_tans]
                for i,c in enumerate(closest):
                    logging.debug("  %d: %r --> %r with dot product = %g"%(c,coords[c],reaches[c].coords[-1], dots[i]))
                c = closest[np.argmax(dots)]
                nodes[c].addChild(n)

            elif len(closest) == 0:
                rivers.append(n)
            else:
                assert(len(closest) == 1)
                nodes[closest[0]].addChild(n)

        assert(len(rivers) > 0)
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
        cp = copy.deepcopy(self)
        for node1,node2 in zip(cp.preOrder(),self.preOrder()):
            node1.properties = copy.deepcopy(node2.properties)
        return cp


def sort_children_by_angle(tree, reverse=False):
    """Sorts the children of a given segment by their angle with respect to that segment."""
    for node in tree.preOrder():
        if len(node.children) > 1:
            # compute tangents
            my_seg_tan = np.array(node.segment.coords[0]) - np.array(node.segment.coords[1])

            if reverse: sign = -1
            else: sign = 1

            def angle(c):
                tan = np.array(c.segment.coords[-2]) - np.array(c.segment.coords[-1])
                return sign*watershed_workflow.utils.angle(my_seg_tan, tan)
           
            node.children.sort(key=angle)


def create_rivers_meshes(rivers, widths=8, enforce_convexity=True):
    """Returns list of elems and river corridor polygons for a given list of river trees

    Parameters:
    -----------
    rivers: List(watershed_workflow.river_tree.RiverTree object)
        List of river tree along which river meshes are to be created
    widths: Float or a dictionary {stream-order: width}
    junction_treatment: boolean 
        flag for enforcing convexity of the pentagons at the junctions
    
    Returns
    -------
    elems: List(List)
        List of river elements
    corrs: List(shapely.geometry.Polygon)
        List of river corridor polygons
    """

    elems=[]
    corrs=[]
    gid_shift=0
    for river in rivers:
        if len(elems)!=0:
            gid_shift=np.max([max(map(int, elem)) for elem in elems])+1
        elems_river, corr = create_river_mesh(river, widths=widths, enforce_convexity=enforce_convexity, gid_shift=gid_shift)
        elems=elems + elems_river
        corrs=corrs+[corr,]

    return elems, corrs


def create_river_mesh(river, widths=8, enforce_convexity=True, gid_shift=0):
    """Returns list of elems and river corridor polygons for a given river tree

    Parameters:
    -----------
    river: watershed_workflow.river_tree.RiverTree object)
        river tree along which mesh is to be created
    widths: Float or a dictionary {stream-order: width}
    junction_treatment: boolean 
        flag for enforcing convexity of the pentagons at the junctions
    gid_shift: Integer
        all the node-ids used in the element defination are shifted by
        this number to make it consistant with the global ids in the 
        m2 mesh, important in case of multiple rivers
        
    Returns
    -------
    elems: List(List)
        List of river elements
    corr: List(shapely.geometry.Polygon)
        a river corridor polygon
    """ 
    if type(widths)== dict:
        dilation_width=np.min(list(widths.values())) 
    else:
        dilation_width=widths

    # creating a polygon for river corridor by dilating the river tree
    corr = create_river_corridor(river, dilation_width)
    # defining special elements in the mesh
    elems = to_quads(river, corr, dilation_width, gid_shift=gid_shift)
    # setting river_widths in the river corridor polygon
    if type(widths)==dict:
        corr = set_width_by_order(river, corr, widths=widths)
    # treating non-convexity at junctions
    if enforce_convexity:
        corr =convexity_enforcement(river, corr)

    return elems, corr


def create_river_corridor(river, river_width):
    """Returns a polygon representing the river corridor.
    
    Parameters
    ----------
    river: watershed_workflow.river_tree.RiverTree object
        river tree along which river corrid polygon is to be created
    river_width: Float
        width of the river corridor for initial dilation

    Returns
    -------
    corr3: shapely.geometry.Polygon
        river corridor polygon for the given river     
    """

    # first sort the river so that in a search we always take paddlers right...
    sort_children_by_angle(river, True)
    delta = river_width / 2.
    length_scale=3*delta 

    # check river consistency
    if not river.is_continuous():
        river.make_continuous()

    # buffer by the width
    mls = shapely.geometry.MultiLineString([r for r in river.dfs()])
    corr = mls.buffer(delta, cap_style=shapely.geometry.CAP_STYLE.flat,
                      join_style=shapely.geometry.JOIN_STYLE.mitre)
    
    # cycle the corridor points to start and end with the 1st point...
    corr_p = list(corr.exterior.coords[:-1])
    outlet_p = river.segment.coords[-1]
    index_min = min(range(len(corr_p)), key=lambda i : watershed_workflow.utils.distance(corr_p[i], outlet_p))
    plus_one = (index_min+1)%len(corr_p)
    minus_one = (index_min-1)%len(corr_p)
    if (watershed_workflow.utils.distance(corr_p[plus_one], outlet_p) < watershed_workflow.utils.distance(corr_p[minus_one], outlet_p)):
        corr2_p = corr_p[plus_one:]+corr_p[0:plus_one]
    else:
        corr2_p = corr_p[index_min:]+corr_p[0:index_min]
    corr2 = shapely.geometry.Polygon(corr2_p)

    # remove endpoint-doubles that we want to be a single point and
    # weird artifact triples at junctions
    corr3_p = []
    i = 0
    while i < len(corr2_p):
        logging.debug(f'considering {i}')
        if i == 0 or i == len(corr2_p)-1:
            # keep first and last always -- first two points make the outlet segment
            logging.debug(f' always keeping')
            corr3_p.append(corr2_p[i])
        else:
            if watershed_workflow.utils.distance(corr2_p[i-1], corr2_p[i]) < length_scale:
                # is this a triple point?
                if watershed_workflow.utils.distance(corr2_p[i+1], corr2_p[i]) < length_scale:
                    logging.debug(' triple point!')
                    # triple point, average neighbors and skip the next point
                    corr3_p.append(watershed_workflow.utils.midpoint(corr2_p[i+1], corr2_p[i-1]))
                    i += 1
                else:
                    # double point -- an end of a first order stream
                    logging.debug(' double point')
                    corr3_p.append(watershed_workflow.utils.midpoint(corr2_p[i-1], corr2_p[i]))
            else:
                # will the next point deal with this?
                if watershed_workflow.utils.distance(corr2_p[i], corr2_p[i+1]) < length_scale:
                    logging.debug(' not my problem')
                    pass
                else:
                    logging.debug(' keeping')
                    corr3_p.append(corr2_p[i])
        i += 1

    # create the polgyon
    corr3 = shapely.geometry.Polygon(corr3_p)
    return corr3


def to_quads(river, corr, delta, gid_shift=0 , ax=None):
    """Iterate over the rivers, creating quads and pentagons forming the corridor.
    The global_id_adjustment is to keep track of node_id in elements w.r.t to global id in mesh
    mainly relevant for multiple river

    Parameters
    ----------
    rivers : watershed_workflow.river_tree.RiverTree object
        river tree 
    corr : shapely.geometry.Polygons
        a river corridor polygon for the river
    delta : Float
        river width used for creating corr from river in function "create_river_corridor"
    gid_shift: Integer
        all the node-ids used in the element defination are shifted by
        this number to make it consistant with the global ids in the 
        m2 mesh, important in case of multiple rivers

    Returns
    -------
    elems: List(List)
        List of river elements
    """
    
    coords=corr.exterior.coords[:-1]
    # number the nodes in a dfs pattern, creating empty space for elements
    for i, node in enumerate(river.preOrder()):
        node.id = i
        node.elements = [list() for l in range(len(node.segment.coords)-1)]
        assert(len(node.elements) >= 1)
        node.touched = 0
        
    # iterate over the tree in an out-and-back-and-in-between
    # traversal, where every node appears num_children + 1 times,
    # before and after and between each child.    
    ic = 0
    total_touches = 0
    for node in river.prePostInBetweenOrder():
        logging.debug(f'touching {node.id} (previously touched {node.touched} times with {len(node.children)} children)')
        if node.touched == 0:
            logging.debug(f'  first time around! {node.touched+1}')
            # not yet touched -- add the first coordinates
            seg_coords = [coords[ic],]
            for j in range(len(node.elements)):
                node.elements[j].append(ic)
                ic += 1
                node.elements[j].append(ic)
                seg_coords.append(coords[ic])

            node.touched += 1
            total_touches += 1

            seg_coords = np.array(seg_coords)

        elif node.touched == 1 and len(node.children) == 0:
            # leaf node, last time
            logging.debug(f' last time around a leaf! {node.touched+1}')
            # increment to avoid double-counting the point in the triangle on the ends
            seg_coords = [coords[ic],]
            ic += 1
            node.elements[-1].append(ic)
            seg_coords.append(coords[ic])
            for j in reversed(range(len(node.elements)-1)):
                node.elements[j].append(ic)
                ic += 1
                node.elements[j].append(ic)
                seg_coords.append(coords[ic])
            node.touched += 1
            total_touches += 1

            seg_coords = np.array(seg_coords)
        
            for i, elem in enumerate(node.elements):
                looped_conn = elem[:]
                looped_conn.append(elem[0])
                if i == len(node.elements)-1:
                    assert(len(looped_conn) == 4)
                else:
                    assert(len(looped_conn) == 5)
                cc = np.array([coords[n] for n in looped_conn])

        elif node.touched == len(node.children):
            logging.debug(f'  last time around! {node.touched+1}')
            seg_coords = [coords[ic],]
            # touched enough times that this is the last appearance
            # add the last coordinates
            for j in reversed(range(len(node.elements))):
                node.elements[j].append(ic)
                ic += 1
                node.elements[j].append(ic)
                seg_coords.append(coords[ic])
            node.touched += 1
            total_touches += 1

            seg_coords = np.array(seg_coords)

            for i,elem in enumerate(node.elements):
                looped_conn = elem[:]
                looped_conn.append(elem[0])
                if i == len(node.elements)-1:
                    assert(len(looped_conn) == (node.touched+3))
                else:
                    assert(len(looped_conn) == 5)
                cc = np.array([coords[n] for n in looped_conn])

                for c in cc:
                    # note, the more acute an angle, the bigger this distance can get...
                    # so it is a bit hard to pin this multiple down -- using 5 seems ok?
                    assert(watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+1)], 10*delta) or \
                           watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+2)], 10*delta))
                 
        else:
            logging.debug(f'  middle time around! {node.touched+1}')
            assert(node.touched < len(node.children))
            # touched in between children
            # therefore this is at least a pentagon
            # add the middle node on the last element
            node.elements[-1].append(ic)
            node.touched += 1

    assert(len(coords) == (ic+1))
    assert(len(river)*2 == total_touches)

    # this nodeid-shift is needed in case of multiple rivers, to make this id consistent with global nodeids in a m2 mesh
    for node in river.preOrder():
        for i, elems in enumerate(node.elements):
            elems_new = [node_id + gid_shift for node_id in elems]
            node.elements[i]=elems_new

    elems=[el for node in river.preOrder() for el in node.elements]
    return elems

def set_width_by_order(river, corr, widths=None):
    """this functions takes the river-corridor polygon and sets the width of the corridor based on the order 
       dependent width dictionary

    Parameters
    ----------
    river: watershed_workflow.river_tree.RiverTree object)
        river tree along which mesh is to be created
    corr : shapely.geometry.Polygons
        a river corridor polygon for the river    
    widths: Float or a dictionary {stream-order: width}

    Returns
    -------
    shapely.geometry.Polygon(corr_coords_new): 
        river corridor polygon with adjusted width
    """

    corr_coords=corr.exterior.coords[:-1]
    for j,node in enumerate(river.preOrder()):
    
        order=node.properties["StreamOrder"]
        target_width=width_cal(widths,order)

        for i, elem in enumerate(node.elements): # treating the upstream edge of the element
            if len(elem)==4:
                p1=np.array(corr_coords[elem[1]][:2]) # points of the upstream edge of the quad
                p2=np.array(corr_coords[elem[2]][:2])
                [p1_, p2_]= move_to_target_separation(p1,p2,target_width)
                corr_coords[elem[1]]=tuple(p1_)
                corr_coords[elem[2]]=tuple(p2_)

            if len(elem)==5:
                p1=np.array(corr_coords[elem[1]][:2]) # neck of the pent
                p2=np.array(corr_coords[elem[3]][:2])
                [p1_, p2_]= move_to_target_separation(p1,p2,target_width)
                corr_coords[elem[1]]=tuple(p1_)
                corr_coords[elem[3]]=tuple(p2_)
                
            if i==0: # this is to treat the most downstream edge which is left out so far
                p1=np.array(corr_coords[elem[0]][:2]) # points of the upstream edge of the quad/pent
                p2=np.array(corr_coords[elem[-1]][:2])
                [p1_, p2_]= move_to_target_separation(p1,p2,target_width)
                corr_coords[elem[0]]=tuple(p1_)
                corr_coords[elem[-1]]=tuple(p2_)

    corr_coords_new=corr_coords+[corr_coords[0]]
    return shapely.geometry.Polygon(corr_coords_new)


def move_to_target_separation(p1, p2, target):
    """Returns the points after moving them to a target separation from each other"""
    import math
    d_vec=p1-p2 # separation vector
    d=np.sqrt(d_vec.dot(d_vec)) # distance
    delta= target-d
    p1_=p1+0.5*delta*(d_vec)/d
    p2_=p2-0.5*delta*(d_vec)/d
    d_=watershed_workflow.utils.distance(p1_,p2_)
    assert(math.isclose(d_, target, rel_tol=1e-5))
    return [p1_, p2_]

def width_cal(width_dict, order):
    """Returns the reach width based using the {order:width dictionary}"""
    if order > max(width_dict.keys()):
        width=width_dict[max(width_dict.keys())]
    elif order < min(width_dict.keys()):
        width=width_dict[min(width_dict.keys())]
    else: 
         width=width_dict[order]
    return width


def convexity_enforcement(river, corr):
    """this functions check the river-corridor polygon for convexity, if non-convex, moves the node onto the convex hull of the polygon

    Parameters
    ----------
    river: watershed_workflow.river_tree.RiverTree object)
        river tree along which mesh is to be created
    corr : shapely.geometry.Polygons
        a river corridor polygon for the river 
  
    Returns
    -------
    shapely.geometry.Polygon(corr_coords_new): 
        river corridor polygon with adjusted width
    """  
    coords=corr.exterior.coords[:-1]

    for j, node in enumerate(river.preOrder()):
        for elem in node.elements:
                if len(elem)==5 or len(elem)==6: # checking and treating this pentagon/hexagon
                    points=[coords[id] for id in elem]
                    if not watershed_workflow.utils.is_convex(points):
                        from shapely.ops import nearest_points
                        convex_ring = shapely.geometry.Polygon(points).convex_hull.exterior
                        for i, point in enumerate(points): # replace point with closest point on convext hull
                            p = shapely.geometry.Point(point)                           
                            new_point = nearest_points(convex_ring, p)[0].coords[0]
                            points[i] = new_point
                    
                    assert(watershed_workflow.utils.is_convex(points))
                        # updating coords
                    for id, point in zip(elem, points):
                        coords[id]=point    

    corr_coords_new=coords+[coords[0]]                     
    return shapely.geometry.Polygon(corr_coords_new)                       