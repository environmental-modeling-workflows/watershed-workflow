
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
import logging
import pandas
import copy

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

    _, ws = watershed_workflow.get_huc(sources['HUC'], huc, crs)
    watershed = watershed_workflow.split_hucs.SplitHUCs([ws])

    # Read the river reaches ------------------------------------------------

    ## download/collect the river network within that shape's bounds    

    ignore_small_rivers = False
    prune_by_area_fraction = False
    _, reaches = watershed_workflow.get_reaches(sources['hydrography'], huc, 
                                                watershed.exterior(), crs, crs,
                                                in_network=True, properties=True, 
                                                include_catchments=True)

    rivers = watershed_workflow.construct_rivers(watershed, reaches, method='hydroseq',
                                                ignore_small_rivers=ignore_small_rivers,
                                                prune_by_area_fraction=prune_by_area_fraction,
                                                remove_diversions=True,
                                                remove_braided_divergences=True)

    ## Select the largest river within the watershed

    da_km2_rivers = np.array([rr.properties['TotalDrainageAreaSqKm'] for rr in rivers]) 
    main_river = [rivers[da_km2_rivers.argmax()]]

    # Densify river network
    river = watershed_workflow.densify_rivers_hucs.densify_rivers(rivers = main_river, rivers_raw=main_river, limit=max_length_segment, 
                                                                      use_original=False, 
                                                                      treat_collinearity=True)
    return watershed, river[0]

def get_WightedWidthFunction(river):

    """This function estimates the weighted width function for a given river network (RN) 
    and a scalar field within the watershed.
    Then, it resamples the RN to improve the width function analysis.
     
    Parameters:
    -----------
    river : watershed_workflow.river_tree.River
        River tree. Also referred to as the river network (RN). Only the main connected river network 
        within the watershed is used. The RN is 
    watershed : watershed_workflow.split_hucs.SplitHUCs
        Watershed boundary.

    Returns:
    -------


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
            
    return d_to_outlet, river

    # n_nodes = river.count() # number of nodes within the RN
    # d_to_outlet = np.array([])

    

    # for node in river.preOrder(): # Move along the nodes   
    #     length_to_outlet = 0 
    #     for node_on_path in node.pathToRoot(): # flow from node root
    #         length_to_outlet +=  node_on_path.segment.length
        
    #     xy_coor = np.array(node.segment.xy)
    #     d = np.cumsum(np.linalg.norm([xy_coor[0,1:]-xy_coor[0,:-1],xy_coor[1,1:]-xy_coor[1,:-1]],axis =0))    
    #     node.properties['LenthToOutlet'] = length_to_outlet - d[-1] # Distance from the downstream node of the segment
    #     d_to_outlet = np.concatenate((d_to_outlet, length_to_outlet - (d[-1] - d))) 
            













