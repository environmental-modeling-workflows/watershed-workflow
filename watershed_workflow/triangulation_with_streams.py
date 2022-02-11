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

    logging.info("Adding river outlet in huc...")

    if type(hucs) is workflow.split_hucs.SplitHUCs or list or shapely.geometry.Polygon:
        huc_segment=add_river_outlet_in_huc(river_corr,hucs) # adjusting hucs to accomodate river corridor
        segments = [huc_segment]
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

def add_river_outlet_in_huc(river_corr,hucs):
    """Returns updated huc with river outlet represented"""
    if type(hucs) is workflow.split_hucs.SplitHUCs:
        huc_segment = hucs.segments[0]
    elif type(hucs) is list:
        huc_segment = hucs[0]
    elif type(hucs) is shapely.geometry.Polygon:
        huc_segment  = hucs.exterior()

    huc_coords=list(huc_segment.coords)[:-1] # to avoid repeated points interferring in the river outlet adjustment
    ind=workflow.utils.closest_point(river_corr.exterior.coords[0],huc_coords)# this is the point to be eliminated from huc boundary
    huc_coords[ind]=river_corr.exterior.coords[-2] # point on the huc boundary closest to the river outlet is replaced by one of the two points at river outlet
    huc_coords.insert(ind+1,river_corr.exterior.coords[0]) # other point of the river outlet is inserted into the huc boundary
    huc_coords.append(huc_coords[0]) # to make the polygonal loop complete
    huc_segment_new=shapely.geometry.LineString(huc_coords)
    
    return huc_segment_new

def pick_hole_point(poly):
    """A function to pick a point inside a polygon"""
    nodes_edges_rc = NodesEdges([poly])
    p1=list(nodes_edges_rc.nodes)[0] 
    p2=list(nodes_edges_rc.nodes)[-1]
    p3=list(nodes_edges_rc.nodes)[1]
    p4=list(nodes_edges_rc.nodes)[-2]
 
    return MultiPoint([p1,p2,p3,p4]).centroid
