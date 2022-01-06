"""this code increases the resolution of river network and huc boundmary in a controlled way using original river network and huc"""

import logging
import collections
import numpy as np
import math
import copy
from scipy import interpolate

import shapely
import shapely.geometry

import workflow.utils

def DensifyTree(tree,tree_, limit=100, collinear=False):
    """This function preOrder travers through the tree and density node.segments
    
     Arguments:
      tree              | tree to be densified (workflow.river_tree.RiverTree)
      tree_             | original tree containing all the known points from NHDPlus (workflow.river_tree.RiverTree)
      limit             | limit of section length above which more points are added
      collinear         | boolean to check for collinearity 
      """
    assert (len(tree)==len(tree_))
    tree_densified=copy.deepcopy(tree)
    for node, node_ in zip(tree_densified.preOrder(), tree_.preOrder()):
        node.segment=DensifyNodeSegments(node,node_,limit=limit,collinear=collinear)
    return tree_densified


def DensifyNodeSegments(node,node_,limit=100,collinear=False):
    """This function adds equally space point in the reach-sections longer than the limit
     Arguments:
      node              | node whose segment to be densified (workflow.river_tree.RiverTree)
      node_             | original node containing all the known points from NHDPlus (workflow.river_tree.RiverTree)
      limit             | limit of section length above which more points are added
      collinear         | boolean to check for collinearity 
      """
      
    seg_coords=list(node.segment.coords) # coordinates of node.segment to be densified
    seg_coords_=list(node_.segment.coords) # coordinates of node.segment from original river network
    seg_coords_densified=seg_coords.copy() # segment coordinates densified
    j=0
    for i in range(len(seg_coords)-1):
        section_length=workflow.utils.distance(seg_coords[i],seg_coords[i+1])
        if section_length>limit:
            number_new_points=int(section_length//limit)
            end_points=[seg_coords[i],seg_coords[i+1]] # points betwen which more points will be added
            new_points=Interpolate(end_points,seg_coords_,number_new_points)
            seg_coords_densified[j+1:j+1]= new_points
            j+=number_new_points        
        j+=1 
    if collinear:
        assert (sum(CheckSegmentCollinearity(seg_coords_densified))==0)
    return shapely.geometry.LineString(seg_coords_densified)


def DensifyHucs(hucs,huc_,limit=200):

    """This function increases the resolution of huc boundary by adding equally spaced interpolated points
     Arguments:
      hucs              | hucs to be densified (workflow.split_hucs.SplitHUCs)
      node_             | original huc containing all the known points fromthe source (workflow.split_hucs.SplitHUCs)
      limit             | limit of section length above which more points are added
    """
    # first if there are multiple segments, we define outer-ring and remove close points
    huc_ring=hucs.exterior().exterior.simplify(tolerance=1)
    coords=list(huc_ring.coords) 
    coords_=list(huc_.exterior().exterior.coords)
    coords_densified=coords.copy()

    j=0
    for i in range(len(coords)-1):
        section_length=math.dist(coords[i],coords[i+1])
        if section_length>limit:
            number_new_points=int(section_length//limit)
            end_points=[coords[i],coords[i+1]]  # points betwen which more points will be added
            new_points=Interpolate(end_points,coords_,number_new_points)
            coords_densified[j+1:j+1]= new_points
            j+=number_new_points        
        j+=1 

    return workflow.split_hucs.SplitHUCs([shapely.geometry.Polygon(coords_densified)])



def Interpolate(end_points,interp_data,n):

    """this function uses original shape to interpolate points while densifying a LineString shape"""

    inds=[closest_point(point,interp_data) for point in end_points] # point-indices on original network slicing a section for interpolation 
    section_interp_data=np.array(interp_data[inds[0]:inds[1]+1]) # coordinates on section
    a=np.array(end_points); (dx,dy)=abs(a[0,:]-a[1,:])
    if dx>dy: # interpolating on x axis
        f = interpolate.interp1d(section_interp_data[:,0],section_interp_data[:,1]) # creating interpolator 
        xnew=np.linspace(end_points[0][0],end_points[1][0], n+2)[1:-1] # new xs equally space between existing points
        ynew=f(xnew) # interpolated ys
    else:  # interpolating on y axis
        f = interpolate.interp1d(section_interp_data[:,1],section_interp_data[:,0]) # creating interpolator 
        ynew=np.linspace(end_points[0][1],end_points[1][1], n+2)[1:-1] # new ys equally space between existing points
        xnew=f(ynew) # interpolated xs
    new_points=[(xnew[k],ynew[k]) for k in range(n)]
    return new_points

        
def CheckSegmentCollinearity(segment_coords):
    """this functions checks for collinearity in a node segment"""

    col_checks=[]
    for i in range(0,len(segment_coords)-2):
        p0=segment_coords[i]
        p1=segment_coords[i+1]
        p2=segment_coords[i+2]
        col_checks.append(collinearity(p0, p1, p2))
    return col_checks       

def collinearity(p0, p1, p2, tol=1e-6):
    """this fucntion checks if three points are collinear for given tolerance value"""
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < tol


def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)