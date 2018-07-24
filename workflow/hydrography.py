import math
import copy
import logging
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial
import itertools

import shapely.geometry

import workflow.conf
import workflow.utils
import workflow.tree
import workflow.hucs
import workflow.plot


def snap(hucs, rivers, tol=0.1, tol_triples=None):
    """Snap HUCs to rivers."""
    assert(type(hucs) is workflow.hucs.HUCs)
    assert(type(rivers) is list)
    assert(all(workflow.tree.is_consistent(river) for river in rivers))

    if len(rivers) is 0:
        return True

    if tol_triples is None:
        tol_triples = tol

    # snap boundary triple junctions to river endpoints
    logging.info(" Snapping polygon segment boundaries to river endpoints")
    snap_polygon_endpoints(hucs, rivers, tol_triples)
    if not all(workflow.tree.is_consistent(river) for river in rivers):
        logging.info("  ...resulted in inconsistent rivers!")
        return False
    
    # snap endpoints of all rivers to the boundary if close
    # note this is a null-op on cases dealt with above
    logging.info(" Snapping river endpoints to the polygon")
    for tree in rivers:
        for node in tree.preOrder():
            node.segment = snap_endpoints(node.segment, hucs, tol)
    if not all(workflow.tree.is_consistent(river) for river in rivers):
        logging.info("  ...resulted in inconsistent rivers!")
        return False

    # deal with intersections
    logging.info(" Cutting at crossings")
    snap_crossings(hucs, rivers, tol)
    consistent = all(workflow.tree.is_consistent(river) for river in rivers)
    if not consistent:
        logging.info("  ...resulted in inconsistent rivers!")
    return consistent

def _snap_and_cut(point, line, tol=0.1):
    """Determine the closest point to a line and, if it is within tol of
    the line, cut the line at that point and snapping the endpoints as
    needed.
    """
    if workflow.utils.in_neighborhood(shapely.geometry.Point(point), line, tol):
        nearest_p = workflow.utils.nearest_point(line, point)
        dist = workflow.utils.distance(nearest_p, point)
        if dist < tol:
            #logging.debug("  snap and cut, dist = %g"%dist)
            if dist == 0.:
                point = workflow.utils.find_perp(line, point)
            
            # project out a point on the othe side of the line
            p2 = (point[0] + 2*(nearest_p[0] - point[0]),
                  point[1] + 2*(nearest_p[1] - point[1]))
            cutline = shapely.geometry.LineString([point, p2])
            if not(line.intersects(cutline)):
                logging.debug("INTERSECTING ISSUE:")
                logging.debug("  cutline:  %r, %r"%(point, p2))
                logging.debug("  line: %r"%line.coords[:])
            return nearest_p, workflow.utils.cut(line, cutline, tol)
    return None, line

def _snap_crossing(hucs, river_node, tol=0.1):
    """Snap a single river node"""
    r = river_node.segment
    for b,spine in hucs.intersections.items():
        for s,seg_handle in spine.items():
            seg = hucs.segments[seg_handle]

            if seg.intersects(r):
                new_spine = workflow.utils.cut(seg, r, tol)
                new_rivers = workflow.utils.cut(r, seg, tol)
                
                river_node.segment = new_rivers[-1]
                if len(new_rivers) > 1:
                    assert(len(new_rivers) == 2)
                    river_node.inject(workflow.tree.Tree(new_rivers[0]))

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
    logging.debug("Snap coord options: %r"%coords)
    kdtree = scipy.spatial.cKDTree(coords)

    # for each segment of the HUC spine, find the river outlet that is
    # closest.  If within tolerance, move it, deleting as need be.
    for seg_handle, seg in hucs.segments.items():
        # check point 0
        dist_ind = kdtree.query(np.array(seg.coords[0]))
        dist = dist_ind[0]
        closest_point = coords[dist_ind[1]]
        logging.debug("checking point: %r, got point: %r"%(seg.coords[0],closest_point))
        if dist < tol:
            # have a closest point, move the segment
            logging.debug("  Moving HUC segment point 0 river outlet to %r"%(closest_point))
            logging.debug("    seg coords (old): %r"%(seg.coords[:]))
            
            new_seg = seg.coords[:]
            new_seg[0] = closest_point
            hucs.segments[seg_handle] = shapely.geometry.LineString(new_seg)
        else:
            logging.debug("  Not moving: %r, %r (dist=%g)"%(closest_point, seg.coords[0],dist))

        # check point -1
        dist_ind = kdtree.query(np.array(seg.coords[-1]))
        dist = dist_ind[0]
        closest_point = coords[dist_ind[1]]
        logging.debug("checking point: %r, got point: %r"%(seg.coords[-1],closest_point))
        if dist < tol:
            # have a closest point, move the segment
            logging.debug("  Moving HUC segment point -1 river outlet to %r"%(closest_point))
            logging.debug("    seg coords (old): %r"%(seg.coords[:]))
            new_seg = seg.coords[:]
            new_seg[-1] = closest_point
            hucs.segments[seg_handle] = shapely.geometry.LineString(new_seg)
        else:
            logging.debug("  Not moving: %r, %r (dist=%g)"%(closest_point, seg.coords[0], dist))

            
def snap_endpoints(river, hucs, tol=0.1):
    """Snap river endpoints to huc segments and insert that point into
    the boundary.

    Note this is O(n^2), and could be made more efficient.
    """
    assert(type(river) is shapely.geometry.LineString)
    for b,component in itertools.chain(hucs.boundaries.items(), hucs.intersections.items()):
        
        # note, this is done in two stages to allow it deal with both endpoints touching
        for s,seg_handle in component.items():
            seg = hucs.segments[seg_handle]
            #logging.debug("SNAP P0:")
            #logging.debug("  huc seg: %r"%seg.coords[:])
            #logging.debug("  river: %r"%river.coords[:])
            altered = False
            new_coord, new_segs = _snap_and_cut(river.coords[0], seg, tol)
            if new_coord is not None:
                logging.info("  - snapped river: %r to %r"%(river.coords[0], new_coord))

                # remove points that are closer
                coords = list(river.coords)
                done = False
                while len(coords) > 2 and workflow.utils.distance(new_coord, coords[1]) < \
                      workflow.utils.distance(new_coord, coords[0]):
                    coords.pop(0)
                coords[0] = new_coord
                river = shapely.geometry.LineString(coords)

                # check for nearly parallel river and huc boundary
                river_tan = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
                for i, seg in enumerate(new_segs):
                    # find the coincident point
                    if workflow.utils.close(seg.coords[0], new_coord):
                        seg_p = 1
                        seg_tan = (seg.coords[1][0] - new_coord[0],
                                   seg.coords[1][1] - new_coord[1])
                    elif workflow.utils.close(seg.coords[-1], new_coord):
                        seg_p = -2
                        seg_tan = (seg.coords[-2][0] - new_coord[0],
                                   seg.coords[-2][1] - new_coord[1])
                    else:
                        assert(False)

                    mag_seg = math.sqrt(seg_tan[0]*seg_tan[0] + seg_tan[1]*seg_tan[1])
                    mag_river = math.sqrt(river_tan[0]*river_tan[0] + river_tan[1]*river_tan[1])
                    angle = np.arccos((seg_tan[0]*river_tan[0] + seg_tan[1]*river_tan[1]) / mag_seg / mag_river)
                    logging.info("angle measured at %g"%angle)
                    if angle < math.pi/16:
                        # dot of tans
                        logging.info("nearly parallel river with angle = %g degrees"%(angle/math.pi*180))
                        if mag_river > 2*mag_seg:
                            # insert segment point into river
                            coords = [coords[0],seg.coords[seg_p],]+coords[1:]
                            river = shapely.geometry.LineString(coords)
                        elif mag_seg > 2*mag_river:
                            # insert river point into segment
                            if seg_p == 1:
                                seg_coords = [seg.coords[0],coords[1],]+seg.coords[1:]
                            else:
                                seg_coords = seg.coords[:-1],[coords[1], seg.coords[-1],]
                            new_segs[i] = shapely.geometry.LineString(seg_coords)
                        else:
                            # move river onto segment
                            coords[1] = seg.coords[seg_p]
                            river = shapely.geometry.LineString(coords)

                # set the new segments
                if len(new_segs) > 1:
                    assert(len(new_segs) is 2)
                    hucs.segments[seg_handle] = new_segs[0]
                    new_handle = hucs.segments.add(new_segs[1])
                    component.add(new_handle)
                elif len(new_segs) is 1:
                    hucs.segments[seg_handle] = new_segs[0]
                    
                break # can only intersect one component -- triple points are already dealt with

        # second stage
        for s,seg_handle in component.items():
            seg = hucs.segments[seg_handle]
            # logging.debug("SNAP P1:")
            # logging.debug("  huc seg: %r"%seg.coords[:])
            # logging.debug("  river: %r"%river.coords[:])
            altered = False
            new_coord, new_segs = _snap_and_cut(river.coords[-1], seg, tol)
            if new_coord is not None:
                logging.info("  - snapped river: %r to %r"%(river.coords[-1], new_coord))

                # remove points that are closer
                coords = list(river.coords)
                done = False
                while len(coords) > 2 and workflow.utils.distance(new_coord, coords[-2]) < \
                      workflow.utils.distance(new_coord, coords[-1]):
                    coords.pop(-1)
                coords[-1] = new_coord
                river = shapely.geometry.LineString(coords)

                # check for nearly parallel river and huc boundary
                river_tan = (coords[-2][0] - coords[-1][0], coords[-2][1] - coords[-1][1])
                for i, seg in enumerate(new_segs):
                    # find the coincident point
                    if workflow.utils.close(seg.coords[0], new_coord):
                        seg_p = 1
                        seg_tan = (seg.coords[1][0] - new_coord[0],
                                   seg.coords[1][1] - new_coord[1])
                    elif workflow.utils.close(seg.coords[-1], new_coord):
                        seg_p = -2
                        seg_tan = (seg.coords[-2][0] - new_coord[0],
                                   seg.coords[-2][1] - new_coord[1])
                    else:
                        assert(False)

                    mag_seg = math.sqrt(seg_tan[0]*seg_tan[0] + seg_tan[1]*seg_tan[1])
                    mag_river = math.sqrt(river_tan[0]*river_tan[0] + river_tan[1]*river_tan[1])
                    angle = np.arccos((seg_tan[0]*river_tan[0] + seg_tan[1]*river_tan[1]) / mag_seg / mag_river)
                    logging.info("angle measured at %g"%angle)
                    if angle < math.pi/16:
                        # dot of tans
                        logging.info("nearly parallel river with angle = %g degrees"%(angle/math.pi*180))
                        if mag_river > 2*mag_seg:
                            # insert segment point into river
                            coords = coords[:-1]+[seg.coords[seg_p],coords[-1]]
                            river = shapely.geometry.LineString(coords)
                        elif mag_seg > 2*mag_river:
                            # insert river point into segment
                            if seg_p == 1:
                                seg_coords = [seg.coords[0],coords[-2],]+seg.coords[1:]
                            else:
                                seg_coords = seg.coords[:-1],[coords[-2], seg.coords[-1],]
                            new_segs[i] = shapely.geometry.LineString(seg_coords)
                        else:
                            # move river onto segment
                            coords[-2] = seg.coords[seg_p]
                            river = shapely.geometry.LineString(coords)


                # set the new segments
                if len(new_segs) > 1:
                    assert(len(new_segs) is 2)
                    hucs.segments[seg_handle] = new_segs[0]
                    new_handle = hucs.segments.add(new_segs[1])
                    component.add(new_handle)
                elif len(new_segs) is 1:
                    hucs.segments[seg_handle] = new_segs[0]

                break # can only intersect one component -- triple points are already dealt with
    return river

def make_global_tree(rivers, tol=0.1):
    if len(rivers) is 0:
        return list()

    # make a kdtree of beginpoints
    coords = np.array([r.coords[0] for r in rivers])
    kdtree = scipy.spatial.cKDTree(coords)

    # make a node for each segment
    nodes = [workflow.tree.Tree(r) for r in rivers]

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

        elif len(closest) is 0:
            trees.append(n)
        else:
            nodes[closest[0]].addChild(n)

    # if len(doublesegs) > 0:
    #     plt.figure()
    #     for r in rivers:
    #         plt.plot(r.xy[0], r.xy[1], 'k')
    #     for r in doublesegs:
    #         plt.plot(rivers[r].xy[0], rivers[r].xy[1], 'r')
    #     for matches in doublesegs_matches:
    #         for m in matches:
    #             plt.plot(rivers[m].xy[0], rivers[m].xy[1], 'b')
    #     for r in doublesegs_winner:
    #         plt.plot(rivers[r].xy[0], rivers[r].xy[1], 'g')
    #     plt.show()            
    return trees


def filter_rivers_to_huc(hucs, rivers, tol):
    """Filters out rivers not inside the HUCs provided."""
    # removes any rivers that are not at least partial contained in the hucs
    if type(rivers) is list and len(rivers) is 0:
        return list()

    logging.info("  ...forming union")
    union = shapely.ops.cascaded_union(list(hucs.polygons()))
    union = union.buffer(tol, 4)
    
    logging.info("  ...filtering")
    if type(rivers) is shapely.geometry.MultiLineString or \
       (type(rivers) is list and type(rivers[0]) is shapely.geometry.LineString):
        rivers2 = [r for r in rivers if union.intersects(r)]
    elif type(rivers) is list and type(rivers[0]) is workflow.tree.Tree:
        rivers2 = [r for river in rivers for r in river.dfs() if union.intersects(r)]

    logging.info("  ...making global tree")
    rivers_tree = workflow.hydrography.make_global_tree(rivers2, tol=0.1)
    logging.info("  ...done")
    return rivers_tree

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
    # simplify
    if simp_tol is not None:
        for tree in rivers:
            simplify(tree, simp_tol)

    # prune short leaf branches and merge short interior reaches
    for tree in rivers:
        if merge_tol is not None:
            merge(tree, merge_tol)
        if merge_tol != prune_tol and prune_tol is not None:
            prune(tree, prune_tol)

def prune(tree, prune_tol=10):
    """Removes any leaf segments that are shorter than prune_tol"""
    for leaf in tree.leaf_nodes():
        if leaf.segment.length < prune_tol:
            logging.info("    cleaned leaf segment of length: %g at centroid %r"%(leaf.segment.length, leaf.segment.centroid.coords[0]))
            leaf.remove()

def merge(tree, tol=0.1):
    """Remove inner branches that are short, combining branchpoints as needed."""
    for node in list(tree.preOrder()):
        if node.segment.length < tol:
            logging.info("    cleaned inner segment of length %g at centroid %r"%(node.segment.length, node.segment.centroid.coords[0]))
            for child in node.children:
                child.segment = shapely.geometry.LineString(child.segment.coords[:-1]+[node.parent.segment.coords[0],])
                node.parent.addChild(child)
            node.remove()
            
def simplify(tree, tol=0.1):
    """Simplify, IN PLACE, all tree segments."""
    for node in tree.preOrder():
        if node.segment is not None:
            node.segment = node.segment.simplify(tol)
            
