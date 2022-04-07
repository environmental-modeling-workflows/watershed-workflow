"""this code increases the resolution of river network and huc boundmary in a controlled way using original river network and huc"""

from cmath import nan
import logging
import collections
from turtle import fd
import numpy as np
import math
from scipy import interpolate

import shapely
import shapely.geometry

import watershed_workflow.utils

def DensifyTree(tree,tree_=None, use_original=False,limit=100, treat_collinearity=False):
    """This function traverse in the river tree and densify node.segments
    
     Arguments:
      tree              | tree to be densified (watershed_workflow.river_tree.RiverTree)
      tree_             | original tree containing all the known points from NHDPlus (watershed_workflow.river_tree.RiverTree)
      limit             | limit of section length above which more points are added
      use_original      | boolean flag for resampling from the original to redensify
      treat_collinearity| boolean to check and treat for collinearity 
      """
       
    assert (len(tree)==len(tree_))

    tree_densified=tree.deep_copy()
    for node, node_ in zip(tree_densified.preOrder(), tree_.preOrder()):
        node.segment=DensifyNodeSegments(node,node_,limit=limit,use_original=use_original,treat_collinearity=treat_collinearity)
    return tree_densified


def DensifyNodeSegments(node,node_,limit=100,use_original=False,treat_collinearity=False):
    """This function adds equally-spaced point in the reach-sections longer than the limit at a desired resolution
     Arguments:
      node              | node whose segment to be densified (watershed_workflow.river_tree.RiverTree)
      node_             | original node containing all the known points from NHDPlus (watershed_workflow.river_tree.RiverTree)
      limit             | limit of section length above which more points are added
      collinear         | boolean to check and treat for collinearity  
      """
      
    seg_coords=list(node.segment.coords) # coordinates of node.segment to be densified
    seg_coords_=list(node_.segment.coords) # coordinates of node.segment from original river network
    seg_coords_densified=seg_coords.copy() # segment coordinates densified
    j=0
    for i in range(len(seg_coords)-1):
        section_length=watershed_workflow.utils.distance(seg_coords[i],seg_coords[i+1])
        if section_length>limit:
            number_new_points=int(section_length//limit)
            end_points=[seg_coords[i],seg_coords[i+1]] # points betwen which more points will be added
            if use_original:
                new_points=Interpolate(end_points,seg_coords_,number_new_points)
            else:
                new_points=Interpolate_simple(end_points,number_new_points)
            seg_coords_densified[j+1:j+1]= new_points
            j+=number_new_points        
        j+=1 
    if treat_collinearity:
        seg_coords_densified=TreatSegmentCollinearity(seg_coords_densified)
    node.segment.coords=seg_coords_densified
    return node.segment

def DensifyHucs(hucs,huc_,river, use_original=False, limit_scales=None):
    """This function densify huc boundaries. The densification length scale either can be a constant value or a refinement 
    function where huc segment refinedment is greater for huc segments closer to the river tree
    Arguments:
    node              | node whose segment to be densified (watershed_workflow.river_tree.RiverTree)
    node_             | original node containing all the known points from NHDPlus (watershed_workflow.river_tree.RiverTree)
    limit_scales      | limit of section length above which more points are added, either a constant value or a list for step refinement [d0, l0, d1, l1]

    """

    # first if there are multiple segments, we define outer-ring and remove close points
    huc_ring=hucs.exterior().exterior.simplify(tolerance=1)
    coords=list(huc_ring.coords) 
    coords_=list(huc_.exterior().exterior.coords)

    if use_original:
        if type(limit_scales)is list:
            # basic refine
            coords_densified_basic=DensifyHucs_(coords,coords_,river,limit_scales=limit_scales[-1])
            # adaptive refine
            coords_densified=DensifyHucs_(coords_densified_basic,coords_,river,limit_scales=limit_scales)
            
        else:
            coords_densified=DensifyHucs_(coords,coords_,river,limit_scales=limit_scales)
      
    else: # in this case original huc boundary coordinates are used for interpolation
        coords_densified=DensifyHucs_(coords,coords_,river,limit_scales=limit_scales)

        
    return watershed_workflow.split_hucs.SplitHUCs([shapely.geometry.Polygon(coords_densified)])


def DensifyHucs_(coords,coords_,river,limit_scales=None):

    """This function increases the resolution of huc boundary by adding equally spaced interpolated points
     Arguments:
      coords            | coordinates of the huc segment to be densified
      huc_              | coordinates of the huc original huc segment from which points can be resmapled
      limit_scales      | limit of section length above which more points are added, either a constant value or a list for step refinement 
                            [near_distance, near_length_scale, far_distance, far_length_scale]
    """
    adaptive=type(limit_scales) is list # setting up flag
    coords_densified=coords.copy() 
    j=0
    for i in range(len(coords)-1):

        # calculation of limit for a set of point
        if adaptive:
            limit=limit_from_river_distance([coords[i],coords[i+1]],limit_scales,river)
        else:
            limit=limit_scales
            
        section_length=math.dist(coords[i],coords[i+1])

        if section_length>limit:
            number_new_points=int(section_length//limit)
            end_points=[coords[i],coords[i+1]]  # points betwen which more points will be added

            if adaptive:
                new_points=Interpolate_simple(end_points,number_new_points)
            else:
                new_points=Interpolate(end_points,coords_,number_new_points)
                
            coords_densified[j+1:j+1]= new_points
            j+=number_new_points        
        j+=1 

    return coords_densified



def Interpolate(end_points,interp_data,n):

    """this function uses original shape to interpolate points while densifying a LineString shape"""

    inds=[watershed_workflow.utils.closest_point(point,interp_data) for point in end_points] # point-indices on original network slicing a section for interpolation 
    if inds[1]<inds[0]: # this is to deal with corner case of interpolation of the last segment
        inds[1]=-2     
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

def Interpolate_simple(end_points,n):
    xnew=np.linspace(end_points[0][0],end_points[1][0], n+2)[1:-1] # new xs equally space between existing points
    ynew=np.linspace(end_points[0][1],end_points[1][1], n+2)[1:-1] # new ys equally space between existing points
    new_points=[(xnew[k],ynew[k]) for k in range(n)]
    return new_points

        
def TreatSegmentCollinearity(segment_coords, tol=1e-6):
    """this functions removes collinearity from a node segment by making small pertubations"""
    col_checks=[]
    for i in range(0,len(segment_coords)-2): # going along segment, checking 3 consecutive points at a time
        p0=segment_coords[i]
        p1=segment_coords[i+1]
        p2=segment_coords[i+2]
        if CheckCollinearity(p0, p1, p2, tol=tol): # treating collinearity through a small pertubation
            del_ortho=10*tol # shift in the middle point
            m=(p2[1] - p0[1])/(p2[0] - p0[0])
            if abs(m)==float('inf'):
                m=1e6
            del_y=del_ortho/(1+m**2)**0.5
            del_x=-1*del_ortho*m/(1+m**2)**0.5
            p1=(p1[0]+del_x , p1[1]+del_y)
            segment_coords[i+1]=p1
        col_checks.append(CheckCollinearity(p0, p1, p2))
    assert (sum(col_checks)==0)
    return  segment_coords   

def CheckCollinearity(p0, p1, p2, tol=1e-6):
    """this fucntion checks if three points are collinear for given tolerance value"""
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < tol


def limit_from_river_distance(segment_ends,limit_scales,river):
    """Returns a graded refinement function based upon a distance function from rivers, for use with DensifyHucs function.
    HUC segment resolution must be higher in near_distance when the HUC segment midpoint is within near_distance from the river network.
    Length must be smaller than away_length when the HUC segment midpoint is at least away_distance from the river network.
    Area must be smaller than a linear interpolant between
    """
    near_distance, near_length, away_distance, away_length=limit_scales
    p0=shapely.geometry.Point(segment_ends[0])
    p1=shapely.geometry.Point(segment_ends[1])
    p_mid=shapely.geometry.Point([(segment_ends[0][0]+segment_ends[1][0])/2,(segment_ends[0][1]+segment_ends[1][1])/2])
    river_multiline = shapely.geometry.MultiLineString(list(river))
    distance=min(p0.distance(river_multiline),p_mid.distance(river_multiline),p1.distance(river_multiline))

    if distance > away_distance:
        length = away_length
    elif distance < near_distance:
        length = near_length
    else:
        length = near_length + (distance - near_distance) / (away_distance - near_distance) * (away_length - near_length)

    return length
