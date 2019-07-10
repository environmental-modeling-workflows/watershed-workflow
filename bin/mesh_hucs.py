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

def get_args():
    # set up parser
    parser = workflow.ui.get_basic_argparse(__doc__+'\n\n'+workflow.source_list.__doc__)
    workflow.ui.huc_arg(parser)
    workflow.ui.outmesh_args(parser)
    workflow.ui.center_options(parser)

    workflow.ui.simplify_options(parser)
    workflow.ui.triangulate_options(parser)

    data_ui = parser.add_argument_group('Data Sources')
    workflow.ui.huc_source_options(data_ui)
    workflow.ui.hydro_source_options(data_ui)
    workflow.ui.dem_source_options(data_ui)

    # parse args, log
    return parser.parse_args()

def mesh_hucs(args):
    sources = workflow.source_list.get_sources(args)

    logging.info("")
    logging.info("Meshing HUC: {}".format(args.HUC))
    logging.info("="*30)
    logging.info('Target projection: "{}"'.format(args.projection['init']))
    
    # collect data
    huc, centroid = workflow.get_split_form_hucs(sources['HUC'], args.HUC, crs=args.projection, centering=args.center)
    rivers, centroid = workflow.get_rivers_by_bounds(sources['hydrography'], huc.polygon(0).bounds, args.projection, args.HUC, centering=centroid)
    rivers = workflow.simplify_and_prune(huc, rivers, args)
    
    # make 2D mesh
    mesh_points2, mesh_tris = workflow.triangulate(huc, rivers, args)

    # elevate to 3D
    if args.center:
        mesh_points2_uncentered = mesh_points2 + np.expand_dims(np.array(centroid.coords[0]),0)
    else:
        mesh_points2_uncentered = mesh_points2

    dem_profile, dem = workflow.get_raster_on_shape(sources['DEM'], huc.polygon(0), args.projection)
    mesh_points3_uncentered = workflow.elevate(mesh_points2_uncentered, dem, dem_profile)

    if args.center:
        mesh_points3 = np.empty(mesh_points3_uncentered.shape,'d')
        mesh_points3[:,0:2] = mesh_points2
        mesh_points3[:,2] = mesh_points3_uncentered[:,2]
    else:
        mesh_points3 = mesh_points3_uncentered

    return centroid, huc, rivers, (mesh_points3, mesh_tris)


if __name__ == '__main__':
#    try:
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)
    centroid, hucs, rivers, triangulation = mesh_hucs(args)
    workflow.bin_utils.plot_with_triangulation(args, hucs, rivers, triangulation)
    workflow.bin_utils.save(args, centroid, triangulation)
    logging.info("SUCESS")
    plt.show()
    sys.exit(0)
#    except KeyboardInterrupt:
#        logging.error("Keyboard Interupt, stopping.")
#        sys.exit(0)
#    except Exception as err:
#        logging.error('{}'.format(str(err)))
#        sys.exit(1)
        
