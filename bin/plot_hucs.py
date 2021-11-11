#!/usr/bin/env python3
"""Downloads and plots HUCs, hydrography, and DEM data."""

import logging
import numpy as np
from matplotlib import pyplot as plt

import watershed_workflow
import watershed_workflow.ui
import watershed_workflow.source_list
import watershed_workflow.bin_utils

def get_args():
    # set up parser
    parser = watershed_workflow.ui.get_basic_argparse(__doc__+'\n\n'+watershed_workflow.source_list.__doc__)
    watershed_workflow.ui.projection(parser)
    watershed_workflow.ui.huc_arg(parser)
    watershed_workflow.ui.huc_level_arg(parser)

    watershed_workflow.ui.simplify_options(parser)
    watershed_workflow.ui.plot_options(parser)

    data_ui = parser.add_argument_group('Data Sources')
    watershed_workflow.ui.huc_source_options(data_ui)
    watershed_workflow.ui.hydro_source_options(data_ui)
    watershed_workflow.ui.dem_source_options(data_ui)

    # parse args, log
    return parser.parse_args()

def plot_hucs(args):
    sources = watershed_workflow.source_list.get_sources(args)

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
    crs, hucs = watershed_workflow.get_split_form_hucs(sources['HUC'], args.HUC, args.level, out_crs=args.projection)
    args.projection = crs
    
    # hydrography
    _, reaches = watershed_workflow.get_reaches(sources['hydrography'], args.HUC, None, crs, crs)

    # raster
    dem_profile, dem = watershed_workflow.get_raster_on_shape(sources['DEM'], hucs.exterior(), crs, crs,
                                                              mask=True, nodata=np.nan)
    logging.info('dem crs: {}'.format(dem_profile['crs']))

    return hucs, reaches, dem, dem_profile

if __name__ == '__main__':
    args = get_args()
    watershed_workflow.ui.setup_logging(args.verbosity, args.logfile)
    hucs, reaches, dem, profile = plot_hucs(args)

    if args.title is None:
        args.title = 'HUC: {}'.format(args.HUC)
            
    fig, ax = watershed_workflow.bin_utils.plot_with_dem(args, hucs, reaches, dem, profile)
        
    logging.info("SUCESS")
    if args.output_filename is not None:
        fig.savefig(args.output_filename, dpi=150)
    plt.show()
        
