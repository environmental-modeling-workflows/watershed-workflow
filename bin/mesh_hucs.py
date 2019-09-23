#!/usr/bin/env python3
"""Downloads and meshes HUCs based on hydrography data."""

from matplotlib import pyplot as plt
import logging

import workflow
import workflow.ui
import workflow.source_list
import workflow.bin_utils

def get_args():
    # set up parser
    parser = workflow.ui.get_basic_argparse(__doc__+'\n\n'+workflow.source_list.__doc__)
    workflow.ui.projection(parser)
    workflow.ui.huc_arg(parser)
    workflow.ui.outmesh_args(parser)

    workflow.ui.simplify_options(parser)
    workflow.ui.triangulate_options(parser)
    workflow.ui.plot_options(parser)

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
    try:
        logging.info('Target projection: "{}"'.format(args.projection['init']))
    except TypeError:
        pass
    
    # collect data
    crs, hucs = workflow.get_split_form_hucs(sources['HUC'], args.HUC, crs=args.projection)
    args.projection = crs

    # hydrography
    _, rivers = workflow.get_reaches(sources['hydrography'], args.HUC, None, crs)
    rivers = workflow.simplify_and_prune(hucs, rivers, args.simplify, args.prune_reach_size, args.cut_intersections)
    
    # make 2D mesh
    mesh_points2, mesh_tris = workflow.triangulate(hucs, rivers,
                                                   verbosity=args.verbosity,
                                                   refine_max_area=args.refine_max_area,
                                                   refine_distance=args.refine_distance,
                                                   refine_max_edge_length=args.refine_max_edge_length,
                                                   refine_min_angle=args.refine_min_angle,
                                                   enforce_delaunay=args.enforce_delaunay)

    # elevate to 3D
    dem_profile, dem = workflow.get_raster_on_shape(sources['DEM'], hucs.exterior(), crs)
    mesh_points3 = workflow.elevate(mesh_points2, crs, dem, dem_profile)

    return hucs, rivers, (mesh_points3, mesh_tris)


if __name__ == '__main__':
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)
    hucs, rivers, triangulation = mesh_hucs(args)
    fig, ax = workflow.bin_utils.plot_with_triangulation(args, hucs, rivers, triangulation)
    workflow.bin_utils.save(args, triangulation)
    logging.info("SUCESS")
    plt.show()
        
