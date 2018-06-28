import copy
import logging
import numpy as np
from matplotlib import pyplot as plt

import shapely.geometry

import workflow.conf
import workflow.utils
import workflow.tree
import workflow.hucs

def quick_cleanup(rivers, tol=0.1):
    """First pass to clean up hydro data"""
    logging.info("  quick cleaning rivers")
    assert(type(rivers) is shapely.geometry.MultiLineString)
    rivers = shapely.ops.linemerge(rivers).simplify(tol)
    return rivers

def bin_rivers(shps, rivers):
    """Bins rivers in shapes.  

    Returns a list (of len n-shps) of 2-tuples, the first being all
    interior river entries, the second being all river entries that
    touch the boundary.  Note crossing river entries may appear in
    multiple bins.
    """
    assert(type(shps) is list)
    # bin by shape
    bins = []
    not_done = np.ones((len(rivers),))
    for j,s in enumerate(shps):
        inside = []
        boundary = []
        for i,r in enumerate(rivers):
            if not_done[i]:
                if s.intersects(r):
                    if s.exterior.intersects(r):
                        boundary.append(r)
                    else:
                        inside.append(r)
                        not_done[i] = 0
        bins.append((inside,boundary))
    return bins


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
                    new_rivers = workflow.utils.cut(r, seg)
                    new_spine = workflow.utils.cut(seg, r)
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
    

def cut_and_bin_another(hucs, rivers):
    """Cuts all hucs and rivers at crossings, and bins the rivers into their containing huc."""
    assert(type(hucs) is workflow.hucs.HUCs)
    assert(type(rivers) is shapely.geometry.MultiLineString)

    bins = [list() for h in range(len(hucs))]
    polys = list(hucs.polygons())

    for lcv_river, r in enumerate(rivers):
        try:
            # is it contained by a poly?
            containing = next(lcv_poly for (lcv_poly, poly) in enumerate(polys) if workflow.utils.contains(poly,r))
        except StopIteration:
            # not contained in one polygon -- either split across multiple or partially out of the domain
            intersections = [lcv_poly for (lcv_poly, poly) in enumerate(polys) if poly.intersects(r)]
            if len(intersections) is 0:
                raise RuntimeError("Reach out of the huc domain?")
            elif len(intersections) is 1:
                lcv_poly = intersections[0]
                # partially outside the domain
                boundary, inter = hucs.gons[lcv_poly]
                for h_boundary in boundary:
                    bspine = hucs.boundaries[h_boundary]
                    bspine_intersections = [(k_seg_handle, seg_handle) for (k_seg_handle, seg_handle) in bspine.items() if hucs.segments[seg_handle].intersects(r)]
                    if len(bspine_intersections) is 0:
                        continue # no intersection here!
                    elif len(bspine_intersections) is 1:
                        # if so, ensure the intersection point is in the boundary
                        k_seg_handle, seg_handle = bspine_intersections[0]
                        seg = hucs.segments[seg_handle]
                        new_segs = workflow.utils.cut(seg, r)
                        if len(new_segs) > 1:
                            hucs.segments.pop(seg_handle)
                            bspine.pop(k_seg_handle)
                            new_handles = hucs.segments.add_many(new_segs)
                            bspine.add_many(new_handles)
                        intersecting_boundaryseg = seg
                    elif len(bspine_intersections) is 2:
                        # boundary intersection point already there, just need to cut
                        intersecting_boundaryseg = shapely.ops.linemerge([hucs.segments[bspine_intersections[0][1]], hucs.segments[bspine_intersections[1][1]]])
                        assert(type(intersecting_boundaryseg) is shapely.geometry.LineString)
                    else:
                        raise RuntimeError("Ruh roh... should be no more than 2 intersections with the boundary!")
                        
                    # now cut the river itself
                    new_rivers = workflow.utils.cut(r, intersecting_boundaryseg)
                    assert(len(new_rivers) is 2)
                    break

                for new_river in new_rivers:
                    if workflow.utils.contains(polys[lcv_poly],new_river):
                        bins[lcv_poly].append(new_river)

            elif len(intersections) is 2:
                # find the union of spines
                inters1 = set(hucs.gons[intersections[0]][1])
                inters2 = set(hucs.gons[intersections[1]][1])
                inters = inters1.intersection(inters2)
                new_rivers = []
                for h_inter in inters:
                    # find the intersection
                    ispine = hucs.intersections[h_spine]
                    ispine_intersections = [(k_seg_handle, seg_handle) for (k_seg_handle, seg_handle) in ispine.items() if hucs.segments[seg_handle].intersects(r)]
                    if len(ispine_intersections) is 0:
                        continue # no intersection here!
                    elif len(ispine_intersections) is 1:
                        # put the intersection point in the spine
                        k_seg_handle, seg_handle = ispine_intersections[0]
                        seg = hucs.segments[seg_handle]
                        new_segs = workflow.utils.cut(seg, r)
                        if len(new_segs) > 1:
                            hucs.segments.pop(seg_handle)
                            ispine.pop(k_seg_handle)
                            new_handles = hucs.segments.add_many(new_segs)
                            ispine.add_many(new_handles)
                        intersecting_interseg = seg
                    elif len(bspine_intersections) is 2:
                        # boundary intersection point already there, just need to cut
                        intersecting_interseg = shapely.ops.linemerge([hucs.segments[ispine_intersections[0][1]], hucs.segments[ispine_intersections[1][1]]])
                        assert(type(intersecting_interseg) is shapely.geometry.LineString)
                    else:
                        raise RuntimeError("Ruh roh... should be no more than 2 intersections with the interior spine!")

                    # now cut the river itself
                    new_rivers.extend(workflow.utils.cut(r, intersecting_interseg))
                    assert(len(new_rivers) is 2)
                    break

                for new_river in new_rivers:
                    for lcv_poly in intersections:
                        if workflow.utils.contains(polys[lcv_poly], new_river):
                            bins[lcv_poly].append(new_river)

            elif len(intersections) > 2:
                raise RuntimeError("Ruh roh... reaches that touch 3 or more HUCs?")
            
        else:
            # contained in one polygon
            bins[containing].append(r)

            # check if it intersects the boundary and a boundary point must be added
            boundary,inter = hucs.gons[containing]
            for h_boundary in boundary:
                bspine = hucs.boundaries[h_boundary]
                bspine_intersections = [(k_seg_handle, seg_handle) for (k_seg_handle, seg_handle) in bspine.items() if hucs.segments[seg_handle].intersects(r)]
                if len(bspine_intersections) is 2:
                    pass # already added
                elif len(bspine_intersections) is 1:
                    # if so, ensure the intersection point is in the boundary
                    k_seg_handle, seg_handle = bspine_intersections[0]
                    seg = hucs.segments[seg_handle]
                    new_segs = workflow.utils.cut(seg, r)
                    if len(new_segs) > 1:
                        hucs.segments.pop(seg_handle)
                        bspine.pop(k_seg_handle)
                        new_handles = hucs.segments.add_many(new_segs)
                        bspine.add_many(new_handles)
                else:
                    assert(len(bspine_intersections) is 0)

    return bins
    
                

                    
                
                
        
        
    
    
           
    

def _split_and_find(segment, crossing, shp):
    # handle the case of the crossing is an endpoint of the segment
    if (workflow.utils.close(crossing.coords[0], segment.coords[0]) or
        workflow.utils.close(crossing.coords[0], segment.coords[-1])):
        inter = segment.intersection(shp)
        if type(inter) is shapely.geometry.Point:
            return None
        else:
            assert(type(inter) is shapely.geometry.LineString)
            return segment

    # handle the proper crossing case
    segs = workflow.utils.cut(segment, shp.boundary)
    inter0 = segs[0].intersection(shp)
    inter1 = segs[1].intersection(shp)
    if type(inter0) is shapely.geometry.LineString and type(inter1) is shapely.geometry.LineString:
        if inter0.length < 1.e-4 * inter1.length:
            return segs[1]
        elif inter1.length < 1.e-4 * inter0.length:
            return segs[0]
        else:
            raise RuntimeError("crossing messed up, both Lines")
    elif (type(inter0) is shapely.geometry.Point or (type(inter0) is shapely.geometry.GeometryCollection and len(inter0) is 0)):
        assert(type(inter1) is shapely.geometry.LineString)
        return segs[1]
    elif (type(inter1) is shapely.geometry.Point or (type(inter1) is shapely.geometry.GeometryCollection and len(inter1) is 0)):
        assert(type(inter0) is shapely.geometry.LineString)
        return segs[0]
    else:
        raise RuntimeError("crossing messed up, types: %r,%r"%(type(segs[0].intersection(shp)), type(segs[1].intersection(shp))))
    
def split_and_bin(shps, rivers):
    """Bins rivers in shapes, first splitting any rivers that cross shape
    boundaries so that the partial segment can be included in the
    correct bin.
    """
    bins = bin_rivers(shps, rivers)
    for shp, abin in zip(shps, bins):
        for boundary in abin[1]:
            crossing = shp.boundary.intersection(boundary)
            if type(crossing) is shapely.geometry.Point:
                mine = _split_and_find(boundary, crossing, shp)
                if mine is not None:
                    abin[0].append(mine)
            elif type(crossing) is shapely.geometry.MultiPoint:
                #if len(crossing) > 2:
                raise NotImplementedError("2 or more crossings not handled")
                # HAVE TO DO SOMETHING SPECIAL HERE?
                # seg = boundary
                # for cross in crossing:
                #     seg = _split_and_find(seg, cross, shp)
                #     if seg is None:
                #         break
                # if seg is not None:
                #     abin[0].append(seg)
    return [abin[0] for abin in bins]
                

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
    



    
    
    
                    
        


