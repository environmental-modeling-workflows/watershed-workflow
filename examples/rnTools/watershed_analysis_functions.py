
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
import logging
import pandas
import copy
from scipy import integrate

import watershed_workflow
import watershed_workflow.source_list
import watershed_workflow.ui
import watershed_workflow.colors
import watershed_workflow.condition
import watershed_workflow.mesh
import watershed_workflow.split_hucs
import watershed_workflow.create_river_mesh
import watershed_workflow.densify_rivers_hucs

def get_RN_WB(huc, 
             crs = watershed_workflow.crs.daymet_crs(), 
             sources = watershed_workflow.source_list.get_default_sources(), 
             max_length_segment=100):

    """This function downloads the watershed boundary and river network (RN) for a given HUC.
    Then, it resamples the RN to improve the width function analysis.
     
    Parameters:
    -----------
    huc : str
        The hydrologic unit used in the analysis 
    crs : CRS object (pyproj.crs.crs.CRS)
        Coordinate Reference System.
    sources : dict
        Source list for watershed workflow
    max_length_segment : int 
        Limit on the section length above which more points are added in to densify the RN 

    Returns:
    -------
    watershed : watershed_workflow.split_hucs.SplitHUCs
        Watershed boundary.
    river : watershed_workflow.river_tree.River
        River tree. Also referred to as the river network (RN). Only the main connected river network 
        within the watershed is used. The RN is 
    """ 

    #  Read the watershed boundary ----------------------------------------

    profile_ws, ws = watershed_workflow.get_huc(sources['HUC'], huc, crs)
    watershed = watershed_workflow.split_hucs.SplitHUCs([ws])

    # Read the river reaches ------------------------------------------------

    ## download/collect the river network within that shape's bounds    

    ignore_small_rivers = False
    prune_by_area_fraction = False
    profile_reaches, reaches = watershed_workflow.get_reaches(sources['hydrography'], huc, 
                                                watershed.exterior(), crs, crs,
                                                in_network=True, properties=True, 
                                                include_catchments=True)

    rivers = watershed_workflow.construct_rivers(watershed, reaches, method='hydroseq',
                                                ignore_small_rivers=ignore_small_rivers,
                                                prune_by_area_fraction=prune_by_area_fraction,
                                                remove_diversions=True,
                                                remove_braided_divergences=True)

    watershed_workflow.get_shapes

    ## Select the largest river within the watershed

    da_km2_rivers = np.array([rr.properties['TotalDrainageAreaSqKm'] for rr in rivers]) 
    main_river = [rivers[da_km2_rivers.argmax()]]

    # Densify river network
    river = watershed_workflow.densify_rivers_hucs.densify_rivers(rivers = main_river, rivers_raw=main_river, limit=max_length_segment, 
                                                                      use_original=False, 
                                                                      treat_collinearity=True)
    return watershed, river[0]

def get_WightedWidthFunction(river, n_bins_wf = 100, weight_field= None):

    """This function estimates the weighted width function for a given river network (RN) 
    and a scalar field within the watershed.
     
    Parameters:
    -----------
    river : watershed_workflow.river_tree.River
        River tree. Also referred to as the river network (RN). Only the main connected river network 
        within the watershed is used. The RN is 
    watershed : watershed_workflow.split_hucs.SplitHUCs
        Watershed boundary.
    n_bins_wf : int
        Number of bins for the width function

    Returns:
    -------
    d_to_outlet : numpy.ndarray
        Distances to the outlet for each point within the discretized version of the RN
    weights : numpy.ndarray
        Weights used to calculate the width function 
    width_function : dict
        The width function as a pdf
    river : watershed_workflow.river_tree.River
        River tree with the new variables for each node: 
        (1)'LenthToOutlet': Distance to outlet from downstream node, and 
        (2)'SegmentLenth': length of the node segment. 
    """ 
    n_nodes = river.count() # number of nodes within the RN
    d_to_outlet = np.array([])
    nodes = list(river.preOrder())

    nodes[0].properties['LenthToOutlet'] = 0 # Distance from the downstream node of the segment
    nodes[0].properties['SegmentLenth'] = nodes[0].segment.length

    for node in nodes[1:]: # Move along the nodes 

        # Distance from the downstream node of the segment
        node.properties['LenthToOutlet'] = node.parent.properties['LenthToOutlet'] + node.parent.properties['SegmentLenth']
        node.properties['SegmentLenth'] = node.segment.length

        xy_coor = np.array(node.segment.xy) # coordinates start at the upstream node

        # xy_coor_parent = np.array(node.parent.segment.xy)
        # fig, axs = plt.subplots(1,1,figsize=[10,10])
        # axs.plot(xy_coor[0,:],xy_coor[1,:],'-or')
        # axs.plot(xy_coor_parent[0,:],xy_coor_parent[1,:],'-ob')
        # axs.plot(xy_coor[0,0],xy_coor[1,0],'-sg')
        # plt.show()

        d = np.concatenate(([0],np.cumsum(np.linalg.norm([xy_coor[0,1:]-xy_coor[0,:-1],xy_coor[1,1:]-xy_coor[1,:-1]],axis =0))))    
        d_to_outlet = np.concatenate((d_to_outlet, node.properties['LenthToOutlet'] + (d[-1] - d) )) 
        
        if weight_field == None:
            weights = np.ones_like(d_to_outlet)

        n_bins = 100
        bin_size = np.ceil(np.max(d_to_outlet)/n_bins)
        bins_edges = np.arange(n_bins+1)*bin_size
        bin_center = (bins_edges[1:]+bins_edges[:-1])/2

        histogram_data, _ = np.histogram(d_to_outlet, bins=bins_edges, weights=weights)
        x_wf = np.concatenate(([bins_edges[0]], bin_center, [bins_edges[-1]]))
        y_wf = np.concatenate(([0], histogram_data, [0]))

        area_wf = np.trapz(y=y_wf, x=x_wf)
        y_wf = y_wf/area_wf
        y_wf_cum = integrate.cumtrapz(y_wf,x_wf,initial=0)
        width_function = {'distance':x_wf, 'pdf':y_wf, 'cdf':y_wf_cum}
            
    return d_to_outlet, weights, width_function, river 













