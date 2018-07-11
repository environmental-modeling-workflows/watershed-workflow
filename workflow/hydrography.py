import copy
import logging
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial

import shapely.geometry

import workflow.conf
import workflow.utils
import workflow.tree
import workflow.hucs
import workflow.plot

def snap(hucs, rivers, tol=0.1):
    """Snap HUCs to rivers."""
    assert(type(hucs) is workflow.hucs.HUCs)
    assert(type(rivers) is list)
    for r in rivers:
        assert type(r) is workflow.tree.Tree


    

        
def snap_endpoints(rivers, shps, tol=0.1):
    """Snap river endpoints to shp boundaries

    Note this is O(n^2), and could be made more efficient using a
    KDTree
    """
    if type(rivers) is shapely.geometry.MultiLineString:
        rivers = list(rivers)

    for i, river in enumerate(rivers):
        rcentroid = river.centroid
        rlength = river.length
        for j,seg in enumerate(shps.segments):
            # do the lines potentially overlap?
            if rcentroid.distance(seg.centroid) < 0.5*(rlength + seg.length):
                altered = False
                rc0 = shapely.geometry.Point(river.coords[0])
                if rc0.distance(seg) < tol:
                    # snap
                    new_coord = seg.interpolate(seg.project(rc0))
                    new_coord = (new_coord.x, new_coord.y)
                    logging.info("  - snapped river %d: %r to %r"%(i,river.coords[0], new_coord))

                    # remove points that are closer
                    coords = list(river.coords)
                    done = False
                    while len(coords) > 2 and workflow.utils.distance(new_coord, coords[1]) < \
                          workflow.utils.distance(new_coord, coords[0]):
                        coords.pop(0)
                    coords[0] = new_coord
                    altered = True

                rc1 = shapely.geometry.Point(river.coords[-1])
                if rc1.distance(seg) < tol:
                    # snap
                    new_coord = seg.interpolate(seg.project(rc1))
                    new_coord = (new_coord.x, new_coord.y)
                    logging.info("  - snapped river %d: %r to %r"%(i,river.coords[-1], new_coord))

                    # remove points that are closer
                    coords = list(river.coords)
                    done = False
                    while len(coords) > 2 and workflow.utils.distance(new_coord, coords[-2]) < \
                          workflow.utils.distance(new_coord, coords[-1]):
                        coords.pop()
                    coords[-1] = new_coord
                    altered = True

                if altered:
                    rivers[i] = shapely.geometry.LineString(coords)
                    continue
    return rivers
                    

def quick_cleanup(rivers, tol=0.1):
    """First pass to clean up hydro data"""
    logging.info("  quick cleaning rivers")
    assert(type(rivers) is shapely.geometry.MultiLineString)
    rivers = shapely.ops.linemerge(rivers).simplify(tol)
    return rivers

def bin_rivers(shps, rivers):
    """Bins rivers in shapes by their beginpoint."""
    assert(type(shps) is list)
    # bin by shape
    bins = []
    done = np.zeros((len(rivers),), bool)
    for j,s in enumerate(shps):
        inside = []
        for i,r in enumerate(rivers):
            if not done[i]:
                if s.intersects(shapely.geometry.Point(r.coords[0])):
                    inside.append(r)
                    done[i] = True
        bins.append(inside)
    return bins

def make_global_tree(rivers, tol=0.1):
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

    if len(doublesegs) > 0:
        plt.figure()
        for r in rivers:
            plt.plot(r.xy[0], r.xy[1], 'k')
        for r in doublesegs:
            plt.plot(rivers[r].xy[0], rivers[r].xy[1], 'r')
        for matches in doublesegs_matches:
            for m in matches:
                plt.plot(rivers[m].xy[0], rivers[m].xy[1], 'b')
        for r in doublesegs_winner:
            plt.plot(rivers[r].xy[0], rivers[r].xy[1], 'g')
        plt.show()            
    return trees

def cut_and_bin(hucs, rivers):
    """Two-pass cut and bin."""
    assert(type(hucs) is workflow.hucs.HUCs)
    if type(rivers) is list:
        rivers = shapely.geometry.MultiLineString(rivers)
    assert(type(rivers) is shapely.geometry.MultiLineString)
    rivers = list(rivers)

    # first do a pass to just intersect all segs with all other segs
    # and brute force add any needed things.
    logging.info("  - cut and bin pass 1: cut")
    to_remove = []
    i = 0
    while i < len(rivers):
        r = rivers[i]
        for spine in hucs.spines():
            for k_seg_handle, seg_handle in list(spine.items()):
                seg = hucs.segments[seg_handle]
                if r.intersects(seg):
                    try:
                        new_rivers = workflow.utils.cut(r, seg)
                        new_spine = workflow.utils.cut(seg, r)
                    except RuntimeError as err:
                        # workflow.plot.river(rivers, color='c')
                        # for poly in hucs.polygons():
                        #     workflow.plot.huc(poly, color='m')
                        # plt.show()
                        plt.show()
                        raise err
                        
                    if len(new_rivers) > 1:
                        rivers.extend(new_rivers)
                        to_remove.append(i)
                    if len(new_spine) > 1:
                        hucs.segments.pop(seg_handle)
                        spine.pop(k_seg_handle)
                        new_handles = hucs.segments.add_many(new_spine)
                        spine.add_many(new_handles)
        i += 1

    # now rivers and boundaries are properly cut
    logging.info("  - cut and bin pass 2: bin")
    done = [False,]*len(rivers)
    for i in to_remove:
        done[i] = True

    bins = []
    for poly in hucs.polygons():
        my_bin = []
        poly_b = poly.buffer(1.e-5, 2)
        for i,r in enumerate(rivers):
            if not done[i]:
                if poly_b.contains(r):
                    my_bin.append(r)
                    done[i] = True
        bins.append(my_bin)
    # check for unbinned
    for i,r in enumerate(rivers):
        if not done[i]:
            logging.warning("Skipping DEAD segment with length: %g"%r.length)
    return bins
    
                

def cleanup(shps, rivers, simp_tol=0.1, prune_tol=10):
    """Some hydrography data seems to get some random branches, typically
    quite short, that are nearly perfectly parallel to other, longer
    branches.  Surely this is a data error -- remove them.

    This returns rivers in a forest, not in a list.
    """
    rivers = quick_cleanup(rivers, simp_tol)
    shps = workflow.hucs.HUCs(shps)

    # prune short leaf branches
    logging.info("  cleaning rivers")
    bins = cut_and_bin(shps,rivers)
    forests = [workflow.tree.make_trees(abin) for abin in bins]
    for forest in forests:
        for rtree in forest:
            prune(rtree, prune_tol)
    return forests


def prune(tree, prune_tol=10):
    """Removes any leaf segments that are shorter than prune_tol"""
    for leaf in tree.leaf_nodes():
        if leaf.segment.length < prune_tol:
            logging.info("    cleaned segment of length: %g at centroid %r"%(leaf.segment.length, leaf.segment.centroid.coords[0]))
            leaf.remove()

def simplify(tree, tol=0.1):
    """Simplify, IN PLACE, all tree segments."""
    for node in tree.preOrder():
        if node.segment is not None:
            node.segment = node.segment.simplify(tol)
            
def sort(shps, hydro):
    """Returns a list, one for each shp in shps, of all (split) segments
    in hydro that are properly contained in the corresponding shp.
    """
    not_done = copy.deepcopy(list(hydro))
    result = []
    for j,s in enumerate(shps):
        s_result = []
        for i,r in enumerate(not_done):
            if r is not None:
                if s.intersects(r):
                    inter = s.intersection(r)
                    assert(type(inter) is shapely.geometry.LineString)
                    s_result.append(inter)

                    if s.exterior.intersects(r):
                        # insert the boundary point, make a new polygon
                        polyring = shapely.geometry.LineString(list(s.exterior.coords))
                        union = polyring.union(r)
                        polys = [g for g in shapely.ops.polygonize(union)]
                        if len(polys) is not 1:
                            import workflow.plot
                            from matplotlib import pyplot as plt
                            for p in polys:
                                workflow.plot.huc(p, style='-x')
                            plt.show()
                            assert(False)
                            
                        new_s = polys[0]
                        assert(len(new_s.exterior.coords) == len(s.exterior.coords)+1)
                        shps[j] = new_s
                    else:
                        not_done[i] = None
        result.append(shapely.geometry.MultiLineString(s_result))
    return result


def sort_precut(shps, hydro):
    """Returns a list, one for each shp in shps, of all (split) segments in hydro that are properly contained in the corresponding shp."""
    not_done = copy.deepcopy(list(hydro))
    result = []
    for j,s in enumerate(shps):
        s_result = []
        for i,r in enumerate(not_done):
            if r is not None:
                if s.intersects(r):
                    inter = s.intersection(r)
                    if type(inter) is shapely.geometry.Point:
                        pass
                    else:
                        assert(type(inter) is shapely.geometry.LineString)
                        s_result.append(inter)
                        if workflow.utils.contains(s,r):
                            not_done[i] = None
        result.append(shapely.geometry.MultiLineString(s_result))
    return result



def _split(l1, l2, merge_tolerance):
    """Cuts two lines, adding the intersection point to both."""
    segs = shapely.ops.polygonize_full(l1.union(l2))[2] # gets the cut edges

    #    inter1 = l1.intersection(segs)
    #    inter2 = l2.intersection(segs)

    # try:
    #     l1b = shapely.ops.linemerge(inter1)
    # except ValueError:
    # find the first
    first = next(seg for seg in segs if np.allclose(seg.coords[0], l1.coords[0], 1.e-7))
    last = next(seg for seg in segs if np.allclose(seg.coords[-1], l1.coords[-1], 1.e-7))

    # check for repeated points
    if np.linalg.norm(np.array(first.coords[-2]) - np.array(first.coords[-1])) < merge_tolerance:
        print("removed point")
        if len(first.coords) == 2:
            first = None
        else:
            first = shapely.geometry.LineString(first.coords[:-2]+[first.coords[-1],])
    if np.linalg.norm(np.array(last.coords[0]) - np.array(last.coords[1])) < merge_tolerance:
        print("removed point")
        if len(last.coords) == 2:
            last = None
        else:
            last = shapely.geometry.LineString([last.coords[0],]+last.coords[2:])

    assert(not(first is None and last is None))
    if first is None:
        l1b = last
    elif last is None:
        l1b = first
    else:           
        l1b = shapely.ops.linemerge([first,last])

    # try:
    #     l2b = shapely.ops.linemerge(inter2)
    # except ValueError:
    # find the first
    first = next(seg for seg in segs if np.allclose(seg.coords[0], l2.coords[0], 1.e-7))
    last = next(seg for seg in segs if np.allclose(seg.coords[-1], l2.coords[-1], 1.e-7))
    # check for repeated points
    if np.linalg.norm(np.array(first.coords[-2]) - np.array(first.coords[-1])) < merge_tolerance:
        print("removed point")
        if len(first.coords) == 2:
            first = None
        else:
            first = shapely.geometry.LineString(first.coords[:-2]+[first.coords[-1],])
    if np.linalg.norm(np.array(last.coords[0]) - np.array(last.coords[1])) < merge_tolerance:
        print("removed point")
        if len(last.coords) == 2:
            last = None
        else:
            last = shapely.geometry.LineString([last.coords[0],]+last.coords[2:])

    assert(not(first is None and last is None))
    if first is None:
        l2b = last
    elif last is None:
        l2b = first
    else:           
        l2b = shapely.ops.linemerge([first,last])

    assert(type(l1b) is shapely.geometry.LineString)
    assert(type(l2b) is shapely.geometry.LineString)
    return l1b, l2b
    

def split_spine(segs, reaches, merge_tolerance):
    """Splits both spine segments and rivers at their intersections."""
    result = []
    for i,s in enumerate(segs):
        for k,r in enumerate(reaches):
            if r.intersects(s):
                assert(type(s) is shapely.geometry.LineString)
                assert(type(r) is shapely.geometry.LineString)
                s2,r2 = _split(s,r, merge_tolerance)
                assert(type(s2) is shapely.geometry.LineString)
                assert(type(r2) is shapely.geometry.LineString)
                segs[i], reaches[k] = s2,r2
    



    
    
    
                    
        


