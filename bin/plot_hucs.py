#!/usr/bin/env python3
"""Downloads and meshes HUC based on hydrography data."""

import os,sys
import numpy as np
from matplotlib import pyplot as plt
import shapely
import logging

import workflow
import workflow.ui
import workflow.source_list
import workflow.bin_utils
import workflow.hydrography

def get_args():
    # set up parser
    parser = workflow.ui.get_basic_argparse(__doc__+'\n\n'+workflow.source_list.__doc__)
    workflow.ui.huc_arg(parser)
    workflow.ui.huc_level_arg(parser)

    workflow.ui.simplify_options(parser)
    workflow.ui.plot_options(parser)

    data_ui = parser.add_argument_group('Data Sources')
    workflow.ui.huc_source_options(data_ui)
    workflow.ui.hydro_source_options(data_ui)
    workflow.ui.dem_source_options(data_ui)

    # parse args, log
    return parser.parse_args()

def plot_hucs(args):
    sources = workflow.source_list.get_sources(args)

    if args.level == 0:
        args.level = len(args.HUC)
    
    logging.info("")
    logging.info("Plotting level {} HUCs in HUC: {}".format(args.level, args.HUC))
    logging.info("="*30)
    try:
        logging.info('Target projection: "{}"'.format(args.projection['init']))
    except TypeError:
        pass
        
    # collect data
    crs, hucs = workflow.get_split_form_hucs(sources['HUC'], args.HUC, args.level, crs=args.projection)
    args.projection = crs
    
    # hydrography
    _, rivers = workflow.get_reaches(sources['hydrography'], args.HUC, None, crs)

    # raster
    dem_profile, dem = workflow.get_masked_raster_on_shape(sources['DEM'], hucs.exterior(), crs, np.nan)
    assert(dem_profile['crs'] == crs)
    logging.info('dem crs: {}'.format(dem_profile['crs']))

    return hucs, rivers, dem, dem_profile

if __name__ == '__main__':
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)
    hucs, rivers, dem, profile = plot_hucs(args)

    if args.title is None:
        args.title = 'HUC: {}'.format(args.HUC)
            
    fig, ax = workflow.bin_utils.plot_with_dem(args, hucs, None, dem, profile, river_color='r')
        
    logging.info("SUCESS")
    if args.output_filename is not None:
        fig.savefig(args.output_filename, dpi=150)
    plt.show()
        
