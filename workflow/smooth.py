import copy
import numpy as np
from matplotlib import pyplot as plt

import shapely
import shapely.geometry
import shapely.ops

import workflow.conf
import sympy.geometry.point


def intersect_and_split(list_of_shapes):
    """
    uniques, intersections = intersect_and_split(list_of_shapes)

    uniques             | A list of LineSegments, one for each shape, 
                        | that do NOT intersect any other shape in the 
                        | list.
    intersections       | A list of lists, such that intersect[i,j] 
                        | contains a LineSegment representing the 
                        | intersection of shapes[i] and shapes[j].
                        | Note this is upper triangular only.
    """
    intersections = [[None for i in range(len(list_of_shapes))] for j in range(len(list_of_shapes))]
    uniques = [shapely.geometry.LineString(sh.boundary.coords) for sh in list_of_shapes]

    for i, s1 in enumerate(list_of_shapes):
        for j, s2 in enumerate(list_of_shapes):
            if i != j and s1.intersects(s2):
                inter = s1.intersection(s2)
                inter = shapely.ops.linemerge(inter)
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
    
def simplify(uniques, intersections, tolerance):
    """Smooths the resulting segments."""
    uniques_sm = [u.simplify(tolerance=tolerance) for u in uniques]

    intersections_sm = [[None for i in range(len(uniques))] for j in range(len(uniques))]
    for i,s1 in enumerate(intersections):
        for j,s2 in enumerate(s1):
            if type(s2) is not shapely.geometry.GeometryCollection():
                intersections_sm[i][j] = s2.simplify(tolerance=tolerance)
    return uniques_sm, intersections_sm

def recombine(uniques, intersections):
    """Combine the segments back into polygons"""
    polygons = []
    for i,u in enumerate(uniques):
        try:
            segs = [seg for seg in u]
        except TypeError: # single seg
            if u is None:
                segs = []
            else:
                segs = [u,]

        
        segs.extend([p for p in intersections[i] if p is not None])
        segs.extend([p[i] for p in intersections if p[i] is not None]) # transpose, get the lower triangle
        merged = shapely.ops.linemerge(segs)
        print("Merging poly %i with %s segments"%(i,len(segs)))
        if type(merged) is not shapely.geometry.LineString:
            for seg in segs:
                plt.plot(seg.xy[0], seg.xy[1])
            plt.show()
                        
        assert type(merged) is shapely.geometry.LineString
        polygons.append(shapely.geometry.Polygon(merged))
    return polygons

def _plot(uniques, intersections,glyph='-'):
    for u in uniques:
        try:
            plt.plot(u.xy[0], u.xy[1],glyph)
        except NotImplementedError: # multiline
            for seg in u:
                plt.plot(seg.xy[0], seg.xy[1],glyph)

    for i in intersections:
        for p in i:
            if p is not None:
                plt.plot(p.xy[0], p.xy[1],glyph)
        

if __name__ == "__main__":
    # load all HUC 8s in HUC 06
    # colors = ['r', 'b', 'g', 'orange']
    # for i,c in zip(range(1,5),colors):
    #     hu = '060%i'%i
    #     pr,hucs = workflow.conf.load_hucs_in(hu, 8)
    #     hucs = [shapely.geometry.shape(s['geometry']) for s in hucs]
    #     for huc in hucs:
    #         plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], color=c)

    plt.figure()
    profile, hucs = workflow.conf.load_hucs_in('06010208', 12)

    # convert to shapely
    hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]

    # intersect
    uniques, intersections = intersect_and_split(hucs_s)
    #_plot(uniques,intersections,'-x')

    # smooth
    uniques_sm, intersections_sm = simplify(uniques,intersections,1./111000.*100) # converts 100m to degrees
    #_plot(uniques_sm,intersections_sm,'-+')

    # recombine
    hucs_sm = recombine(uniques_sm, intersections_sm)

    for p in hucs_sm:
        plt.plot(p.boundary.xy[0], p.boundary.xy[1])    
    plt.show()





