#!/usr/bin/env python3
"""Plots watersheds and their context within a HUC.
"""
import os,sys
import logging
import numpy as np
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
    workflow.ui.huc_args(parser)
    return parser.parse_args()

def get_huc(args, huc):
    sources = workflow.files.get_sources(args)

    # collect data
    hucs, centroid = workflow.hilev.get_hucs(huc, sources['HUC'], center=False)
    rivers = workflow.hilev.get_rivers(huc, sources['HUC'], filter_long=10)
    dem_profile, dem = workflow.hilev.get_dem(huc, sources)

    # simple triangulation for elevation data
    footprint = shapely.ops.cascaded_union(list(hucs.polygons())).simplify(10)

    if args.refine_max_area is None:
        args.refine_max_area = footprint.area / 1000.
    mesh_points2, mesh_tris = workflow.hilev.triangulate(footprint, None, args, False)

    # elevate to 3D
    mesh_points3 = workflow.hilev.elevate(mesh_points2, dem, dem_profile)
    return centroid, hucs, rivers, (mesh_points3, mesh_tris)
    
def plot(ax, centroid, hucs, rivers, triangulation, color='k', elev_extent=None, cb=True, vmin=None, vmax=None):
    mappable = workflow.plot.triangulation(*triangulation, color='elevation', linewidth=0, vmin=vmin, vmax=vmax)
    if cb:
        fig.colorbar(mappable, orientation="horizontal", pad=0.1)
    workflow.plot.hucs(hucs, color, linewidth=0.7)
    workflow.plot.rivers(rivers, color='white', linewidth=0.5)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('')
    ax.set_xticklabels([round(0.001*tick) for tick in ax.get_xticks()])
    plt.ylabel('')
    ax.set_yticklabels([round(0.001*tick) for tick in ax.get_yticks()])

if __name__ == '__main__':
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)

    colors = ['k', 'r', 'r', 'r']
    
    fig = plt.figure(figsize=(4,5),dpi=300)
    ax = fig.add_subplot(111)
    vmin = None
    vmax = None
    
    for i, (huc, color) in enumerate(zip(args.HUCS, colors)):
        centroid, hucs, rivers, triangulation = get_huc(args, huc)
        if i is 0:
            vmin = triangulation[0][:,2].min()
            vmax = triangulation[0][:,2].max()
        plot(ax, centroid, hucs, rivers, triangulation, color=color, vmin=vmin, vmax=vmax, cb=(i == 0))
    plt.savefig('huc_%s'%args.HUCS[0])
    plt.show()
        
        

    
    
    
