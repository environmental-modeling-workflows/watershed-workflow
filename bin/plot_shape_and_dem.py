#!/usr/bin/env python3
"""Plots watersheds and their DEM at native raster scale.
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
    workflow.ui.huc_hint_options(parser)
    workflow.ui.inshape_args(parser)
    return parser.parse_args()


def get_data(args):
    workflow.ui.setup_logging(args.verbosity, args.logfile)

    args.source_hydro = args.source_huc
    args.source_huc = 'NHD WBD' # hard-coded, but need to use the full WBD dataset
    sources = workflow.files.get_sources(args)

    # collect data
    profile, watersheds, watershed_boundary, centroid = workflow.hilev.get_shapes(args.infile, args.shape_index, center=False, make_hucs=False)
    hucstr = workflow.hilev.find_huc(profile, watershed_boundary, sources['HUC'], args.hint)
    logging.info("found shapes in HUC %s"%hucstr)

    dem_profile, dem = workflow.hilev.get_dem_on_huc(hucstr, sources)
    rivers = workflow.hilev.get_rivers(hucstr, sources['Hydro'])
    rivers = [shapely.affinity.translate(r, -centroid.coords[0][0], -centroid.coords[0][1]) for r in rivers]

    return 
    
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
        
        

    
    
    
