#!/usr/bin/env python3
"""Plots watersheds and their context within a HUC."""

import os,sys
import logging
import numpy as np
from matplotlib import pyplot as plt
import shapely

import workflow
import workflow.ui
import workflow.source_list
import workflow.bin_utils


def get_args():
    parser = workflow.ui.get_basic_argparse(__doc__)
    workflow.ui.huc_arg(parser)

    workflow.ui.simplify_options(parser)
    
    data_ui = parser.add_argument_group('Data Sources')
    workflow.ui.huc_source_options(data_ui)
    workflow.ui.hydro_source_options(data_ui)
    workflow.ui.dem_source_options(data_ui)

def get_huc(args, huc):
    sources = workflow.files.get_sources(args)

    logging.info("")
    logging.info("Plotting HUC: {}".format(args.HUC))
    logging.info("="*30)
    logging.info('Target projection: "{}"'.format(args.projection['init']))
    
    # collect data
    huc, centroid = workflow.get_split_form_hucs(sources['HUC'], args.HUC, crs=args.projection, centering=args.center)
    rivers, centroid = workflow.get_rivers_by_bounds(sources['hydrography'], huc.polygon(0).bounds, args.projection, args.HUC, centering=centroid)
    rivers = workflow.simplify_and_prune(huc, rivers, args)

    # clip and mask
    dem_profile, dem = workflow.get_raster_on_shape(sources['DEM'], huc.exterior(), args.projection)
    ## -- HOW TO MASK? --
    # mask to huc.exterior()

    return centroid, huc, rivers, dem


if __name__ == '__main__':
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, (huc, color) in enumerate(zip(args.HUCS, colors)):
        centroid, hucs, rivers, triangulation = get_huc(args, huc)
        if i is 0:
            vmin = triangulation[0][:,2].min()
            vmax = triangulation[0][:,2].max()
        plot(ax, centroid, hucs, rivers, triangulation, color=color, vmin=vmin, vmax=vmax, cb=(i == 0))
    plt.savefig('huc_%s'%args.HUCS[0])
    plt.show()
        
        

    
    
    
