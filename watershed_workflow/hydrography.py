import math
import copy
import logging
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial
from scipy.spatial import cKDTree
import itertools
import collections

import shapely.geometry

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.river_tree
import watershed_workflow.split_hucs
import watershed_workflow.plot


def snap(hucs, rivers, tol=0.1, tol_triples=None, cut_intersections=False):
    """Snap HUCs to rivers."""
    assert(type(hucs) is watershed_workflow.split_hucs.SplitHUCs)
    assert(type(rivers) is list)
    assert(all(watershed_workflow.river_tree.is_consistent(river) for river in rivers))
    list(hucs.polygons())

    if len(rivers) == 0:
        return True
    assert(len(rivers) > 0)
    for r in rivers:
        assert(len(r) > 0)

    if tol_triples is None:
        tol_triples = tol

    # snap boundary triple junctions to river endpoints
    logging.info("  snapping polygon segment boundaries to river endpoints")
    snap_polygon_endpoints(hucs, rivers, tol_triples)
    if not all(watershed_workflow.river_tree.is_consistent(river) for river in rivers):
        logging.info("    ...resulted in inconsistent rivers!")
        return False
    try:
        list(hucs.polygons())
    except AssertionError:
        logging.info("    ...resulted in inconsistent HUCs")
        return False

    logging.debug('snap part 1')
    logging.debug(list(rivers[0].segment.coords))
    logging.debug(list(hucs.polygon(0).boundary.coords))
    #watershed_workflow.plot.hucs(hucs, style='-x')
    
    # snap endpoints of all rivers to the boundary if close
    # note this is a null-op on cases dealt with above
    logging.info("  snapping river endpoints to the polygon")
    for tree in rivers:
        snap_endpoints(tree, hucs, 0.5*tol)
    if not all(watershed_workflow.river_tree.is_consistent(river) for river in rivers):
        logging.info("    ...resulted in inconsistent rivers!")
        return False
    try:
        list(hucs.polygons())
    except AssertionError:
        logging.info("    ...resulted in inconsistent HUCs")
        return False

    logging.debug('snap part 2')
    logging.debug(list(rivers[0].segment.coords))
    logging.debug(list(hucs.polygon(0).boundary.coords))

    # deal with intersections
    if cut_intersections:
        logging.info("  cutting at crossings")
        snap_crossings(hucs, rivers, tol)
        consistent = all(watershed_workflow.river_tree.is_consistent(river) for river in rivers)
        if not consistent:
            logging.info("  ...resulted in inconsistent rivers!")
            return False
        try:
            list(hucs.polygons())
        except AssertionError:
            logging.info("  ...resulted in inconsistent HUCs")
            return False

    logging.debug('snap part 3')
    logging.debug(list(rivers[0].segment.coords))
    logging.debug(list(hucs.polygon(0).boundary.coords))

    # dealing with crossings might have generated river segments
    # outside of my space.  remove these.  Note the use of negative tol
    logging.info("  filtering rivers to HUC")
    rivers = filter_reaches_to_shape(hucs.exterior(), rivers, -0.1*tol)
    return rivers


def _snap_and_cut(point, line, tol=0.1):
    """Determine the closest point to a line and, if it is within tol of
    the line, cut the line at that point and snapping the endpoints as
    needed.
    """
    if watershed_workflow.utils.in_neighborhood(shapely.geometry.Point(point), line, tol):
        logging.debug("  - in neighborhood")
        nearest_p = watershed_workflow.utils.nearest_point(line, point)
        dist = watershed_workflow.utils.distance(nearest_p, point)
        logging.debug("  - nearest p = {0}, dist = {1}, tol = {2}".format(nearest_p, dist, tol))
        if dist < tol:
            if dist < 1.e-7:
                # filter case where the point is already there
                if any(watershed_workflow.utils.close(point, c) for c in line.coords):
                    return None 
            return nearest_p
    return None


def _snap_crossing(hucs, river_node, tol=0.1):
    """Snap a single river node"""
    r = river_node.segment
    logging.debug("len spine, boundary = {0},{1}".format(len(hucs.intersections), len(hucs.boundaries)))
    for b,spine in itertools.chain(hucs.intersections.items(), hucs.boundaries.items()):
        logging.debug("len spine seg = {0}".format(len(spine)))
        #for b,spine in hucs.intersections.items():
        for s,seg_handle in spine.items():
            seg = hucs.segments[seg_handle]

            logging.debug("  - intersection?:")
            logging.debug(list(r.coords))
            logging.debug(list(seg.coords))

            if seg.intersects(r):
                logging.debug("  - YES")
                #try:
                new_spine = watershed_workflow.utils.cut(seg, r, tol)
                # except RuntimeError as err:
                #     plt.figure()
                #     watershed_workflow.plot.hucs(hucs,None,color='gray')
                #     plt.plot(seg.xy[0], seg.xy[1], 'b-+')
                #     plt.plot(r.xy[0], r.xy[1], 'r-x')
                #     plt.show()
                #     raise err

                #try:
                new_rivers = watershed_workflow.utils.cut(r, seg, tol)
                # except RuntimeError as err:
                #     plt.figure()
                #     watershed_workflow.plot.hucs(hucs,None,color='gray')
                #     plt.plot(seg.xy[0], seg.xy[1], 'b-+')
                #     plt.plot(r.xy[0], r.xy[1], 'r-x')
                #     plt.show()
                #     raise err
                
                river_node.segment = new_rivers[-1]
                if len(new_rivers) > 1:
                    assert(len(new_rivers) == 2)
                    river_node.inject(watershed_workflow.river_tree.RiverTree(new_rivers[0]))

                hucs.segments[seg_handle] = new_spine[0]
                if len(new_spine) > 1:
                    assert(len(new_spine) == 2)
                    new_handle = hucs.segments.add(new_spine[1])
                    spine.add(new_handle)
                break

            
def snap_crossings(hucs, rivers, tol=0.1):
    """Snaps HUC boundaries and rivers to crossings."""
    for tree in rivers:
        for river_node in tree.preOrder():
            _snap_crossing(hucs, river_node, tol)

            
def snap_polygon_endpoints(hucs, rivers, tol=0.1):
    """Snaps the endpoints of HUC segments to endpoints of rivers."""
    # make the kdTree of endpoints of all rivers
    coords1 = np.array([r.coords[-1] for tree in rivers for r in tree.dfs()])
    coords2 = np.array([r.coords[0] for tree in rivers for r in tree.leaves()])
    coords = np.concatenate([coords1, coords2], axis=0)

    # limit to x,y
    if (coords.shape[1] != 2):
        coords = coords[:, 0:2]
    # kdtree = scipy.spatial.cKDTree(coords)
    kdtree = cKDTree(coords)  
    # for each segment of the HUC spine, find the river outlet that is
    # closest.  If within tolerance, move it
    for seg_handle, seg in hucs.segments.items():
        # check point 0, -1
        endpoints = np.array([seg.coords[0], seg.coords[-1]])
        # limit to x,y
        if (endpoints.shape[1] != 2):
            endpoints = endpoints[:,0:2]
        dists,inds = kdtree.query(endpoints)
        if dists.min() < tol:
            new_seg = list(seg.coords)
            if dists[0] < tol:
                new_seg[0] = coords[inds[0]]
                logging.debug(f"  Moving HUC segment point 0,1: {list(seg.coords)[0]}, {list(seg.coords)[-1]}")
                logging.debug("        point 0 to river at %r"%list(new_seg[0]))
            if dists[1] < tol:
                new_seg[-1] = coords[inds[1]]
                logging.debug(f"  Moving HUC segment point 0,1: {list(seg.coords)[0]}, {list(seg.coords)[-1]}")
                logging.debug("        point -1 to river at %r"%list(new_seg[-1]))
            hucs.segments[seg_handle] = shapely.geometry.LineString(new_seg)

def snap_endpoints(tree, hucs, tol=0.1):
    """Snap river endpoints to huc segments and insert that point into
    the boundary.

    Note this is O(n^2), and could be made more efficient.
    """
    to_add = []
    for node in tree.preOrder():
        river = node.segment
        for b,component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):

            # note, this is done in two stages to allow it deal with both endpoints touching
            for s,seg_handle in component.items():
                seg = hucs.segments[seg_handle]
                #logging.debug("SNAP P0:")
                #logging.debug("  huc seg: %r"%seg.coords[:])
                #logging.debug("  river: %r"%river.coords[:])
                altered = False
                logging.debug("  - checking river coord: %r"%list(river.coords[0]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = _snap_and_cut(river.coords[0], seg, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord != None:
                    logging.info("    snapped river: %r to %r"%(river.coords[0], new_coord))

                    # move new_coord onto an existing segment coord
                    dist = np.linalg.norm(np.array(seg.coords) - np.expand_dims(new_coord,0), 2, axis=1)
                    assert(len(dist) == len(seg.coords))
                    assert(len(dist.shape) == 1)
                    i = int(np.argmin(dist))
                    if (dist[i] < tol):
                        new_coord = seg.coords[i]
                    
                    # remove points that are closer
                    coords = list(river.coords)
                    done = False
                    while len(coords) > 2 and watershed_workflow.utils.distance(new_coord, coords[1]) < \
                          watershed_workflow.utils.distance(new_coord, coords[0]):
                        coords.pop(0)
                    coords[0] = new_coord
                    river = shapely.geometry.LineString(coords)
                    node.segment = river
                    to_add.append((seg_handle, component, 0, node))
                    break

            # second stage
            for s,seg_handle in component.items():
                seg = hucs.segments[seg_handle]
                # logging.debug("SNAP P1:")
                # logging.debug("  huc seg: %r"%seg.coords[:])
                # logging.debug("  river: %r"%river.coords[:])
                altered = False
                logging.debug("  - checking river coord: %r"%list(river.coords[-1]))
                logging.debug("  - seg coords: {0}".format(list(seg.coords)))
                new_coord = _snap_and_cut(river.coords[-1], seg, tol)
                logging.debug("  - new coord: {0}".format(new_coord))
                if new_coord != None:
                    logging.info("  - snapped river: %r to %r"%(river.coords[-1], new_coord))

                    # move new_coord onto an existing segment coord
                    dist = np.linalg.norm(np.array(seg.coords) - np.expand_dims(new_coord,0), 2, axis=1)
                    assert(len(dist) == len(seg.coords))
                    assert(len(dist.shape) == 1)
                    i = int(np.argmin(dist))
                    if (dist[i] < tol):
                        new_coord = seg.coords[i]
                    
                    # remove points that are closer
                    coords = list(river.coords)
                    done = False
                    while len(coords) > 2 and \
                       watershed_workflow.utils.distance(new_coord, coords[-2]) < watershed_workflow.utils.distance(new_coord, coords[-1]):
                        coords.pop(-1)
                    coords[-1] = new_coord
                    river = shapely.geometry.LineString(coords)
                    node.segment = river
                    to_add.append((seg_handle, component, -1, node))
                    break

    # find the list of points to add to a given segment
    to_add_dict = dict()
    for seg_handle, component, endpoint, node in to_add:
        if seg_handle not in to_add_dict.keys():
            to_add_dict[seg_handle] = list()
        to_add_dict[seg_handle].append((component, endpoint, node))

    # find the set of points to add to each given segment
    def equal(p1,p2):
        if watershed_workflow.utils.close(p1[2].segment.coords[p1[1]], p2[2].segment.coords[p2[1]], 1.e-5):
            assert(p1[0] == p2[0])
            return True
        else:
            return False
    to_add_dict2 = dict()
    for seg_handle, insert_list in to_add_dict.items():
        new_list = []
        for p1 in insert_list:
            if (all(not equal(p1, p2) for p2 in new_list)):
                new_list.append(p1)
        to_add_dict2[seg_handle] = new_list

    # add these points to the segment
    for seg_handle, insert_list in to_add_dict2.items():
        seg = hucs.segments[seg_handle]
        # make a list of the coords and a flag to indicate a new
        # coord, then sort it by arclength along the segment.
        #
        # Note this needs special care if the seg is a loop, or else the endpoint gets sorted twice        
        if not watershed_workflow.utils.close(seg.coords[0], seg.coords[-1]):
            new_coords = [[p[2].segment.coords[p[1]],1] for p in insert_list]
            old_coords = [[c,0] for c in seg.coords if not any(watershed_workflow.utils.close(c, nc, tol) for nc in new_coords)]
            new_seg_coords = sorted(new_coords+old_coords,
                                    key = lambda a:seg.project(shapely.geometry.Point(a)))

            # determine the new coordinate indices
            breakpoint_inds = [i for i,(c,f) in enumerate(new_seg_coords) if f == 1]

        else:
            new_coords = [[p[2].segment.coords[p[1]],1] for p in insert_list]
            old_coords = [[c,0] for c in seg.coords[:-1] if not any(watershed_workflow.utils.close(c, nc, tol) for nc in new_coords)]
            new_seg_coords = sorted(new_coords+old_coords,
                                    key = lambda a:seg.project(shapely.geometry.Point(a)))
            breakpoint_inds = [i for i,(c,f) in enumerate(new_seg_coords) if f == 1]
            assert(len(breakpoint_inds) > 0)
            new_seg_coords = new_seg_coords[breakpoint_inds[0]:] + new_seg_coords[0:breakpoint_inds[0]+1]
            new_seg_coords[0][1] = 0
            new_seg_coords[-1][1] = 0
            breakpoint_inds = [i for i,(c,f) in enumerate(new_seg_coords) if f == 1]

        # now break into new segments
        new_segs = []
        ind_start = 0
        for ind_end in breakpoint_inds:
            assert(ind_end != 0)
            new_segs.append(shapely.geometry.LineString([c for (c,f) in new_seg_coords[ind_start:ind_end+1]]))
            ind_start = ind_end

        assert(ind_start < len(new_seg_coords)-1)
        new_segs.append(shapely.geometry.LineString([tuple(c) for (c,f) in new_seg_coords[ind_start:]]))

        # put all new_segs into the huc list.  Note insert_list[0][0] is the component
        hucs.segments[seg_handle] = new_segs.pop(0)
        new_handles = hucs.segments.add_many(new_segs)
        insert_list[0][0].add_many(new_handles)

    return river


def make_global_tree(rivers, tol=0.1):
    """Sorts shapely river objects into a list of tree structures."""
    if len(rivers) == 0:
        return list()

    # make a kdtree of beginpoints
    coords = np.array([r.coords[0] for r in rivers])
    # kdtree = scipy.spatial.cKDTree(coords)
    kdtree = cKDTree(coords)
    # make a node for each segment
    nodes = [watershed_workflow.river_tree.RiverTree(r) for r in rivers]
    assert(len(nodes) > 0)
    
    # match nodes to their parent through the kdtree
    trees = []
    doublesegs = []
    doublesegs_matches = []
    doublesegs_winner = []
    for j,n in enumerate(nodes):
        # find the closest beginpoint the this node's endpoint
        closest = kdtree.query_ball_point(n.segment.coords[-1], tol)
        if len(closest) > 1:
            logging.debug("Bad multi segment:")
            logging.debug(" connected to %d: %r"%(j,list(n.segment.coords[-1])))
            doublesegs.append(j)
            doublesegs_matches.append(closest)

            # end at the same point, pick the min angle deviation
            my_tan = np.array(n.segment.coords[-1]) - np.array(n.segment.coords[-2])
            my_tan = my_tan / np.linalg.norm(my_tan)
            
            other_tans = [np.array(rivers[c].coords[1]) - np.array(rivers[c].coords[0]) for c in closest]
            other_tans = [ot/np.linalg.norm(ot) for ot in other_tans]
            dots = [np.inner(ot,my_tan) for ot in other_tans]
            for i,c in enumerate(closest):
                logging.debug("  %d: %r --> %r with dot product = %g"%(c,coords[c],rivers[c].coords[-1], dots[i]))
            c = closest[np.argmax(dots)]
            doublesegs_winner.append(c)
            nodes[c].addChild(n)

        elif len(closest) == 0:
            trees.append(n)
        else:
            assert(len(closest) == 1)
            nodes[closest[0]].addChild(n)

    assert(len(trees) > 0)
    return trees


def filter_reaches_to_shape(shape, reaches, tol):
    """Filters out reaches (or reaches in rivers) not inside the HUCs provided."""
    shape = shape.buffer(2*tol)
    
    # removes any reaches that are not at least partial contained in the hucs
    if type(reaches) is list and len(reaches) == 0:
        return list()

    logging.info("  ...filtering")
    if type(reaches) is shapely.geometry.MultiLineString or \
       (type(reaches) is list and type(reaches[0]) is shapely.geometry.LineString):
        reaches2 = [r for r in reaches if watershed_workflow.utils.non_point_intersection(shape,r)]
    elif type(reaches) is list and type(reaches[0]) is watershed_workflow.river_tree.RiverTree:
        reaches2 = [r for river in reaches for r in river.preOrder() if watershed_workflow.utils.non_point_intersection(shape,r.segment)]
        for r in reaches2:
            r.segment.properties = r.properties
        reaches2 = make_global_tree([r.segment for r in reaches2])
    else:
        raise RuntimeError("Unrecognized river shape type?")
    return reaches2
    
def quick_cleanup(rivers, tol=0.1):
    """First pass to clean up hydro data"""
    logging.info("  quick cleaning rivers")
    assert(type(rivers) is shapely.geometry.MultiLineString)
    rivers = shapely.ops.linemerge(rivers).simplify(tol)
    return rivers

def cleanup(rivers, simp_tol=0.1, prune_tol=10, merge_tol=10):
    """Some hydrography data seems to get some random branches, typically
    quite short, that are nearly perfectly parallel to other, longer
    branches.  Surely this is a data error -- remove them.
    
    This returns rivers in a forest, not in a list.
    """
    merge_tol=10
    # simplify
    if simp_tol is not None:
        for tree in rivers:
            simplify(tree, simp_tol)

    # prune short leaf branches and merge short interior reaches
    for tree in rivers:
        if merge_tol is not None:
            merge(tree, merge_tol)
        if merge_tol != prune_tol and prune_tol is not None:
            prune_by_segment_length(tree, prune_tol)

def prune_by_segment_length(tree, prune_tol=10):
    """Removes any leaf segments that are shorter than prune_tol"""
    for leaf in tree.leaf_nodes():
        if leaf.segment.length < prune_tol:
            logging.info("  ...cleaned leaf segment of length: %g at centroid %r"%(leaf.segment.length, leaf.segment.centroid.coords[0]))
            if 'area' in leaf.properties and 'area' in leaf.parent.properties:
                leaf.parent.properties['area'] += leaf.properties['area']
            leaf.remove()

def accumulate(tree):
    """Accumulates areas up the tree."""
    try:
        for node in tree.postOrder():
            total_area = sum(c.properties['total contributing area'] for c in node.children)
            node.properties['total contributing area'] = total_area + node.properties['area']
    except KeyError:
        raise ValueError("accumulate() cannot be called on rivers whose reaches do not include the 'area' property.")

def prune_by_area(tree, tol):
    """Removes segments whose contributing area is less than tolerance.  

    Units of tol are that of the CRS, squared"""
    if 'total contributing area' not in tree.properties:
        accumulate(tree)

    count = 0
    for node in tree.preOrder():
        if node.properties['total contributing area'] < tol:
            count += 1
            node.remove()
            node.clear() # this ensures we can stop removing anything upstream of this
    return count

def prune_by_area_fraction(tree, tol, total_area=None):
    """Removes segements whose contributing area, divided by the total area, is < tol."""
    if 'total contributing area' not in tree.properties:
        accumulate(tree)
    if total_area is None:
        total_area = tree.properties['total contributing area']
    logging.info(f'... total contributing area = {total_area}')
        
    count = 0
    for node in tree.preOrder():
        if node.properties['total contributing area'] / total_area < tol:
            logging.info(f'... removing: {node.properties["total contributing area"]} of {total_area}')
            count += 1
            node.remove()
            node.clear() # this ensures we can stop removing anything upstream of this
    return count
    
def merge(tree, tol=0.1):
    """Remove inner branches that are short, combining branchpoints as needed."""
    for node in list(tree.preOrder()):
        if node.segment.length < tol and node.parent is not None:
            logging.info("  ...cleaned inner segment of length %g at centroid %r"%(node.segment.length, node.segment.centroid.coords[0]))
            num_children = len(node.children)
            for child in node.children:
                child.segment = shapely.geometry.LineString(child.segment.coords[:-1]+[node.parent.segment.coords[0],])
                if 'area' in child.properties and 'area' in node.properties:
                    child.properties['area'] += node.properties['area']/num_children
                node.parent.addChild(child)
            node.remove()
            
def simplify(tree, tol=0.1):
    """Simplify, IN PLACE, all tree segments."""
    for node in tree.preOrder():
        if node.segment is not None:
            new_seg = node.segment.simplify(tol)
            assert(watershed_workflow.utils.close(new_seg.coords[0], node.segment.coords[0]))
            assert(watershed_workflow.utils.close(new_seg.coords[-1], node.segment.coords[-1]))
            node.segment = new_seg
            
