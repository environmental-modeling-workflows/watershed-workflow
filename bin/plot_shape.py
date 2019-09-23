#!/usr/bin/env python3
"""Plots watersheds and their context within a HUC.
"""
import logging
from matplotlib import pyplot as plt
import numpy as np

import workflow
import workflow.ui
import workflow.source_list
import workflow.bin_utils
import workflow.plot

def get_args():
    # set up parser
    parser = workflow.ui.get_basic_argparse(__doc__+'\n\n'+workflow.source_list.__doc__)
    workflow.ui.projection(parser)
    workflow.ui.inshape_args(parser)
    workflow.ui.huc_hint_options(parser)

    workflow.ui.simplify_options(parser)
    workflow.ui.plot_options(parser)

    data_ui = parser.add_argument_group('Data Sources')
    workflow.ui.huc_source_options(data_ui)
    workflow.ui.hydro_source_options(data_ui)
    workflow.ui.dem_source_options(data_ui)

    # parse args, log
    return parser.parse_args()

def plot_shape(args):
    sources = workflow.source_list.get_sources(args)
    
    logging.info("")
    logging.info("Plotting shapes from file: {}".format(args.input_file))
    logging.info("="*30)
    try:
        logging.info('Target projection: "{}"'.format(args.projection['init']))
    except TypeError:
        pass

    # collect data
    # -- get the shapes
    crs, shapes = workflow.get_split_form_shapes(args.input_file, args.shape_index, args.projection)
    args.projection = crs

    # -- get the containing huc
    hucstr = workflow.find_huc(sources['HUC'], shapes.exterior(), crs, args.hint, shrink_factor=0.1)
    logging.info("found shapes in HUC %s"%hucstr)
    _, huc = workflow.get_huc(sources['HUC'], hucstr, crs)
    
    # -- get reaches of that huc
    _, reaches = workflow.get_reaches(sources['hydrography'], hucstr,
                                      shapes.exterior().bounds, crs)

    # -- dem
    dem_profile, dem = workflow.get_masked_raster_on_shape(sources['DEM'], shapes.exterior(), crs, np.nan)
    assert(dem_profile['crs'] == crs)
    logging.info('dem crs: {}'.format(dem_profile['crs']))

    return shapes, huc, reaches, dem, dem_profile


if __name__ == '__main__':
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)

    # get objects
    shapes, huc, reaches, dem, dem_profile = plot_shape(args)

    # plot
    fig, ax = workflow.bin_utils.plot_with_dem(args, shapes, None, dem, dem_profile, river_color='white')
    workflow.plot.shply([huc,], args.projection, color='k', ax=ax)

    logging.info("SUCESS")
    if args.output_filename is not None:
        fig.savefig(args.output_filename, dpi=150)
    plt.show()
        

    
    
    
