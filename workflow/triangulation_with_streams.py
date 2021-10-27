"""Triangulates polygons with stream network representation"""
import logging
import collections
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import scipy.spatial
import meshpy.triangle

import shapely
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split

import workflow.river_tree
import workflow.split_hucs
from workflow.triangulation import Nodes, NodesEdges

def orient(e):
    if e[0] > e[1]:
        return e[1],e[0]
    elif e[0] < e[1]:
        return e[0],e[1]
    else:
        return None

    
def triangulate(hucs, river_corr ,mixed=True ,tol=1, **kwargs):
    """Triangulates HUCs and rivers.

    Arguments:
      hucs              | a workflow.split_hucs.SplitHUCs instance
      river_corr        | a shapely.geometry.polygon.Polygon given by river_tree.create_river_corridor
      mixed             | boolean for mixed-element mesh
       


    Additional keyword arguments include all options for meshpy.triangle.build()
    """
    logging.info("Triangulating...")
    if type(hucs) is workflow.split_hucs.SplitHUCs:
        segments = list(hucs.segments)
    elif type(hucs) is list:
        segments = hucs
    elif type(hucs) is shapely.geometry.Polygon:
        segments = [hucs,]
    else:
        raise RuntimeError("Triangulate not implemented for container of type '%r'"%type(hucs))
        
    if type(river_corr) is list:
        segments = river_corr + segments
    elif type(river_corr) is shapely.geometry.Polygon:
        segments = [river_corr,] + segments
    else:
        raise RuntimeError("Triangulate not implemented for container of type '%r'"%type(hucs))

    nodes_edges = NodesEdges(segments)

    logging.info("   %i points and %i facets"%(len(nodes_edges.nodes), len(nodes_edges.edges)))
    nodes_edges.check(tol=tol)
    
    logging.info(" building graph data structures")
    info = meshpy.triangle.MeshInfo()
    nodes = np.array(list(nodes_edges.nodes), dtype=np.float64)
    
    pdata = [tuple([float(c) for c in p]) for p in nodes]
    fdata = [[int(i) for i in f] for f in nodes_edges.edges]

    # break the huc boundary intersecting with river outlet to create outlet face
    fdata=outlet_face(fdata, pdata, river_corr, hucs)

    info.set_points(pdata)
    info.set_facets(fdata)

    # adding hole in the river corridor for quad elements
    if mixed:
        hole_point= pick_hole_point(river_corr) # a point inside the river corridor
        assert (river_corr.contains(hole_point))
        info.set_holes([hole_point.coords[0],])

    logging.info(" triangle.build...")

    # pop this option if false, which silences the warning if it does
    # not exist but we didn't ask for it anyway.
    if 'enforce_delaunay' in kwargs.keys() and not kwargs['enforce_delaunay']:
        kwargs.pop('enforce_delaunay')

    try:
        mesh = meshpy.triangle.build(info, **kwargs)
    except TypeError as err:
        try:
            # our modification to meshpy.triangle is not present, try without it
            kwargs.pop('enforce_delaunay')
        except KeyError:
            raise err
        else:
            logging.warning("Triangulate: '--enforce-delaunay' option requires a hacked `meshpy.triangle`.  Proceeding without this option because it is not recognized.")
            mesh = meshpy.triangle.build(info, **kwargs)
            
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    logging.info("  ...built: %i mesh points and %i triangles"%(len(mesh_points),len(mesh_tris)))
    return mesh_points, mesh_tris

def huc_edge_with_river_outlet(river_corr,hucs):
    """Returns huc edges (list of shapely.geometry.LineString) on which river outlet lies (can be single line or two lines)"""
    # huc as multiLineString object
    huc_perimeter=list(hucs.segments)[0]
    # spliting huc perimeter into individual lines and finding which lines has river outlet
    points_to_split = MultiPoint([Point(x,y) for x,y in huc_perimeter.coords[:-1]])
    huc_splitted = split(huc_perimeter,points_to_split)
    outlet_huc_edges=[]
    for line in huc_splitted:
        if line.intersects(river_corr):
            outlet_huc_edges.append(line)
    return outlet_huc_edges

def outlet_face(fdata, pdata, river_corr, hucs):
    """A function to update facet data such that river-corridor outlet is defined on the huc boundary"""
    # get the edge on huc where river outlet lies (as LineString)
    outlet_huc_edges= huc_edge_with_river_outlet(river_corr,hucs)
    # indices of end points of huc edges with outlet
    inds_huc_outlet=[]
    for line in outlet_huc_edges:
        inds_huc_outlet.append(list(orient([pdata.index(tuple(round(num, 3) for num in coord)) for coord in line.coords])))
    # points on the river corridor at the outlet
    nodes_edges_rc = NodesEdges([river_corr])
    inds_rc_outlet=[0,len(nodes_edges_rc.nodes)-1]
    # removing the larger facet
    for inds in inds_huc_outlet:
        fdata.remove(inds)
    
    # inds to be used in defining new fdata sets
    if len(outlet_huc_edges) == 2:
        inds_huc_inserted=[max(inds) for inds in inds_huc_outlet] # minimum index is the outlet of the huc (is it allways true??)
    else:
        inds_huc_inserted=inds_huc_outlet[0]
    # adding two smaller facets adjacent to stream outlet 
    fdata.insert(0,orient([inds_rc_outlet[0],inds_huc_inserted[0]]))
    fdata.insert(0,orient([inds_rc_outlet[1],inds_huc_inserted[1]]))
    
    return fdata

def pick_hole_point(poly):
    """A function to pick a point inside a polygon"""
    nodes_edges_rc = NodesEdges([poly])
    p1=list(nodes_edges_rc.nodes)[0] 
    p2=list(nodes_edges_rc.nodes)[-1]
    p3=list(nodes_edges_rc.nodes)[1]
    p4=list(nodes_edges_rc.nodes)[-2]
 
    return MultiPoint([p1,p2,p3,p4]).centroid
