import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as pcm
import shapely
from shapely.geometry import Point
import logging
import pandas
import copy
from scipy import integrate
import rasterio
import fiona

import watershed_workflow
import watershed_workflow.source_list
import watershed_workflow.ui
import watershed_workflow.colors
import watershed_workflow.condition
import watershed_workflow.mesh
import watershed_workflow.split_hucs
import watershed_workflow.create_river_mesh
import watershed_workflow.densify_rivers_hucs


def get_Pixels_Inside_Watershed(x_daymet, y_daymet, watershed):

    """This function returns the daymet grig indexes (i,j) inside the watershed.
     
    Parameters:
    -----------
    x_daymet : numpy.ndarray
        Coordinate x of the Daymet grid 
    y_daymet : numpy.ndarray
        Coordinate y of the Daymet grid
    watershed : watershed_workflow.split_hucs.SplitHUCs
        Watershed boundary.

    Returns:
    -------
    i_grid : numpy.ndarray
        Row for the grid pixels inside the waterhsed  
    j_grid : numpy.ndarray
        Column for the grid pixels inside the waterhsed    
    idx_grid : numpy.ndarray
        Index for the grid pixels inside the waterhsed reshaped as a vector        
    x_grid : numpy.ndarray
        Coordinate x for the grid pixels reshaped as a vector 
    y_grid : numpy.ndarray
        Coordinate y for the grid pixels reshaped as a vector         
    """ 
    X_grid, Y_grid = np.meshgrid(x_daymet, y_daymet)
    n_pixels = X_grid.size
    x_grid = np.reshape(X_grid,(n_pixels,1))
    y_grid = np.reshape(Y_grid,(n_pixels,1))
    xy_grid = np.concatenate((x_grid,y_grid),axis=1)
    watershed_poly = watershed.polygon(0)
    inside = [watershed_poly.contains(Point(theCoords)) for theCoords in xy_grid]
    i_grid, j_grid = np.where(np.reshape(inside,X_grid.shape))
    idx_grid = np.where(inside)
    return i_grid, j_grid, idx_grid, x_grid, y_grid 


def get_Statistics_Daymet(daymet_data, i_grid, j_grid, idx_time, ivar):
#'tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl'

    vals = daymet_data[ivar][idx_time, i_grid[0], j_grid[0]]




