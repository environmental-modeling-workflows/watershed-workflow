#!/usr/bin/env python3
"""Plots watersheds and their context within a HUC.
"""
import os,sys
import logging
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
import shapely

import rasterio.rio.clip

import workflow.hilev
import workflow.ui
import workflow.files
import workflow.plot


def get_args():
    parser = workflow.ui.get_basic_argparse(__doc__)
    workflow.ui.huc_source_options(parser)
    workflow.ui.dem_source_options(parser)
    workflow.ui.refine_max_area_options(parser)
    workflow.ui.huc_arg(parser)
    workflow.ui.inshape_args(parser)
    return parser.parse_args()

def get_data(args):
    workflow.ui.setup_logging(args.verbosity, args.logfile)

    sources = workflow.files.get_sources(args)

    # -- huc data
    hucs, centroid = workflow.hilev.get_hucs(args.HUC, sources['HUC'])

    # -- find and center rivers
    rivers = workflow.hilev.get_rivers(args.HUC, sources['HUC'])
    rivers = [shapely.affinity.translate(r, -centroid.coords[0][0], -centroid.coords[0][1]) for r in rivers]

    # -- dem
    dem_profile, dem = workflow.hilev.get_dem_on_huc(args.HUC, sources)

    # simple triangulation for elevation data
    footprint = shapely.ops.cascaded_union(list(hucs.polygons())).simplify(10)

    if args.refine_max_area is None:
        args.refine_max_area = footprint.area / 1000.
    mesh_points2, mesh_tris = workflow.hilev.triangulate(footprint, None, args, False)

    # uncenter, elevate to 3D, and recenter
    mesh_points2_uncentered = mesh_points2 + np.expand_dims(np.array(centroid.coords[0]),0)
    mesh_points3_uncentered = workflow.hilev.elevate(mesh_points2_uncentered, dem, dem_profile)
    mesh_points3 = np.empty(mesh_points3_uncentered.shape,'d')
    mesh_points3[:,0:2] = mesh_points2
    mesh_points3[:,2] = mesh_points3_uncentered[:,2]

    # shape data
    profile, watersheds, watershed_boundary, centroid_shp = \
                workflow.hilev.get_shapes(args.infile, args.shape_index, center=False, make_hucs=False)
    watersheds = [shapely.affinity.translate(ws, -centroid.coords[0][0], -centroid.coords[0][1]) for ws in watersheds]
    return centroid, hucs, watersheds, rivers, (mesh_points3, mesh_tris)
    
def plot(centroid, hucs, watersheds, rivers, triangulation):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mappable = workflow.plot.triangulation(*triangulation, linewidth=0, color='elevation')
    #fig.colorbar(mappable, orientation="horizontal", pad=0.1)
    workflow.plot.hucs(hucs, 'k', linewidth=0.7)
    workflow.plot.rivers(rivers, color='white', linewidth=0.5)
    workflow.plot.shapes(watersheds, color='r')
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('')
    ax.set_xticklabels([round(0.001*tick) for tick in ax.get_xticks()])
    plt.ylabel('')
    ax.set_yticklabels([round(0.001*tick) for tick in ax.get_yticks()])

if __name__ == '__main__':
    args = get_args()

    centroid, hucs, watersheds, rivers, triangulation = get_data(args)
    plot(centroid, hucs, watersheds, rivers, triangulation)
    plt.savefig('huc_%s'%args.HUC)
    plt.show()
        
        

    
    
    
