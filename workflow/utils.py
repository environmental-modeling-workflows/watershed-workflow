"""Shapely-only utilities not provided by shapely"""
import logging
import shapely.geometry
import shapely.ops

_tol = 1.e-7
def close(s1, s2, tol=_tol):
    """Are two shapes topologically equivalent and gemoetrically close"""
    if type(s1) is shapely.geometry.Point:
        return close(s1.coords[0], s2, tol)
    if type(s2) is shapely.geometry.Point:
        return close(s1, s2.coords[0], tol)
    if type(s1) != type(s2):
        return False
    if type(s1) is tuple:
        if len(s1) != len(s2):
            return False
        return sum((p1 - p2)**2 for p1,p2 in zip(s1,s2)) < tol**2
    elif type(s1) is shapely.geometry.LineString:
        if len(s1.coords) != len(s2.coords):
            return False
        for c1, c2 in zip(s1.coords, s2.coords):
            if (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 > tol**2:
                return False
        return True
    elif type(s1) is shapely.geometry.MultiLineString:
        if len(s1) != len(s2):
            return False
        good2 = [False,]*len(s2)
        for l1 in s1:
            found = False
            for i,l2 in enumerate(s2):
                if not good2[i]:
                    if close(l1,l2,tol):
                        good2[i] = True
                        found = True
                        break
            if not found:
                return False
        return True
    raise NotImplementedError("Not implemented for type '%r'"%type(s1))

def contains(s1, s2, tol=_tol):
    """A contains algorithm that deals with close/roundoff issues"""
    return s1.buffer(tol,2).contains(s2)

# def _cut_seg_seg(seg1, seg2):
#     """Cuts seg1 and seg2, returning two lists of segments.
    
#     Note seg1 and seg2 must be length-2 coordinate LineString objects.
#     """
#     assert(type(seg1) is shapely.geometry.LineString)
#     assert(type(seg2) is shapely.geometry.LineString)
#     assert(len(seg1.coords) == 2)
#     assert(len(seg2.coords) == 2)
#     assert(seg1.intersects(seg2))

#     inter = seg1.intersection(seg2)
#     assert(type(inter) is shapely.geometry.Point)

#     if close(inter.coords[0], seg1.coords[0]):
#         seg1_split = [None, seg1]
#     elif close(inter.coords[0], seg1.coords[-1]):
#         seg1_split = [seg1, None]
#     else:
#         seg1_split = [shapely.geometry.LineString([seg1.coords[0], inter]),
#                       shapely.geometry.LineString([inter, seg1.coords[-1]])]

#     if close(inter.coords[0], seg2.coords[0]):
#         seg2_split = [None, seg2]
#     if close(inter.coords[0], seg2.coords[-1]):
#         seg2_split = [seg2, None]
#     else:
#         seg2_split = [shapely.geometry.LineString([seg2.coords[0], inter]),
#                       shapely.geometry.LineString([inter, seg2.coords[-1]])]
#     return seg1_split, seg2_split


def cut(line, cutline):
    """Cuts line at its intersection with cutline."""
    assert(type(line) is shapely.geometry.LineString)
    assert(type(cutline) is shapely.geometry.LineString)
    assert(line.intersects(cutline))

    segs = []
    coords = line.coords

    segcoords = [coords[0],]
    i = 0
    while i < len(coords)-1:
        seg = shapely.geometry.LineString(coords[i:i + 2])
        logging.debug("Intersecting seg %d"%i)
        point = seg.intersection(cutline)
        if type(point) is shapely.geometry.GeometryCollection and len(point) is 0:
            logging.debug("Cut seg no intersection")
            segcoords.append(seg.coords[-1])
            i += 1
        elif type(point) is shapely.geometry.Point:
            logging.debug("Cut intersected at point")
            logging.debug("  inter point: %r"%list(point.coords[0]))
            logging.debug("  seg final point: %r"%list(seg.coords[-1]))
            logging.debug("  close? = %r"%(close(point, seg.coords[-1])))
            if close(point, seg.coords[-1], 1.e-5):
                # intersects at the far point
                segs.append(shapely.geometry.LineString(segcoords+[seg.coords[-1],]))
                logging.debug("  appended full segment: %r"%(list(segs[-1].coords)))

                if (i < len(coords)-2):
                    logging.debug("    (not the end)")
                    segcoords = [seg.coords[-1],coords[i+2]]
                else:
                    logging.debug("    (the end)")
                    segcoords = [seg.coords[-1],]
                i += 2 # also skip the next seg, which would also
                       # intersect at that seg's start point
            elif close(point, seg.coords[0]):
                # intersects at the near point
                assert(i == 0)
                segcoords.append(seg.coords[-1])
                i += 1
            else:
                # intersects in the middle
                segs.append(shapely.geometry.LineString(segcoords+[point,]))
                logging.debug("  appended partial segment: %r"%(list(segs[-1].coords)))
                segcoords = [point,seg.coords[-1]]
                i += 1
        else:
            raise RuntimeError("Dual/multiple intersection in a single seg... ugh!")

    if len(segcoords) > 1:
        segs.append(shapely.geometry.LineString(segcoords))
    return segs
        

def intersect_and_split(list_of_shapes):
    """Given a list of shapes which share boundaries (i.e. they partition
    some space), return a compilation of their segments.

    Given a list of shapes of length N, returns:

    uniques             | An N-length-list of either None, LineString,
                        |  or MultiLineString, describing the exterior 
                        |  boundary
    intersections       | A NxN list of lists of either None, LineString, 
                        |  or MultiLineString, describing the interior
                        |  boundary.
    """
    intersections = [[None for i in range(len(list_of_shapes))] for j in range(len(list_of_shapes))]
    uniques = [shapely.geometry.LineString(sh.boundary.coords) for sh in list_of_shapes]

    for i, s1 in enumerate(list_of_shapes):
        for j, s2 in enumerate(list_of_shapes):
            if i != j and s1.intersects(s2):
                inter = s1.intersection(s2)
                if type(inter) is shapely.geometry.MultiLineString:
                    inter = shapely.ops.linemerge(inter)
                elif type(inter) is shapely.geometry.LineString:
                    pass
                elif type(inter) is shapely.geometry.Point:
                    continue
                else:
                    raise RuntimeError("Invalid type of intersection: %r"%type(inter))
                
                uniques[i] = uniques[i].difference(inter)

                # only save once!
                if i > j:
                    intersections[i][j] = inter

    # merge uniques, as we have a bunch of segments.
    for i,u in enumerate(uniques):
        if type(u) is shapely.geometry.MultiLineString:
            uniques[i] = shapely.ops.linemerge(uniques[i])

    uniques_r = [None,]*len(uniques)
    for i,u in enumerate(uniques):
        if type(u) is not shapely.geometry.GeometryCollection:
            uniques_r[i] = u
    return uniques_r, intersections
