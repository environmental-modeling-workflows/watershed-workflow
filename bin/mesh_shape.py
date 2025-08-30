#!/usr/bin/env python3
"""Downloads and meshes shapes based upon hydrography data."""

from matplotlib import pyplot as plt
import logging

import watershed_workflow
import watershed_workflow.ui
import watershed_workflow.sources
import watershed_workflow.bin_utils

def get_args():
    # set up parser
    parser = watershed_workflow.ui.get_basic_argparse(__doc__+'\n\n'+
                                                      watershed_workflow.sources.__doc__)
    watershed_workflow.ui.projection(parser)
    watershed_workflow.ui.inshape_args(parser)
    watershed_workflow.ui.huc_hint_options(parser)
    watershed_workflow.ui.outmesh_args(parser)

    watershed_workflow.ui.simplify_options(parser)
    watershed_workflow.ui.triangulate_options(parser)
    watershed_workflow.ui.plot_options(parser)

    data_ui = parser.add_argument_group('Data Sources')
    watershed_workflow.ui.huc_source_options(data_ui)
    watershed_workflow.ui.hydro_source_options(data_ui)
    watershed_workflow.ui.dem_source_options(data_ui)
    
    # parse args, log
    return parser.parse_args()

def mesh_shapes(args):
    sources = watershed_workflow.sources.get_sources(args)

    logging.info("")
    logging.info("Meshing shapes from: {}".format(args.input_file))
    logging.info("  with index: {}".format(args.shape_index))
    logging.info("="*30)
    try:
        logging.info('Target projection: "{}"'.format(args.projection['init']))
    except TypeError:
        pass
    
    # collect data
    # -- get the shapes
    crs, shapes = watershed_workflow.get_split_form_shapes(args.input_file, args.shape_index, out_crs=args.projection)
    args.projection = crs

    # -- get the containing huc
    hucstr = watershed_workflow.find_huc(sources['HUC'], shapes.exterior(), crs, args.hint, shrink_factor=0.1)
    logging.info("found shapes in HUC %s"%hucstr)

    # -- get reaches of that huc
    _, reaches = watershed_workflow.get_reaches(sources['hydrography'], hucstr, shapes.exterior().bounds, crs, crs)
    rivers = watershed_workflow.simplify_and_prune(shapes, reaches, args.simplify, args.prune_reach_size, args.cut_intersections)
    
    # make 2D mesh
    mesh_points2, mesh_tris, _, _ = watershed_workflow.triangulate(shapes, rivers,
                                                   verbosity=args.verbosity,
                                                   refine_max_area=args.refine_max_area,
                                                   refine_distance=args.refine_distance,
                                                   refine_max_edge_length=args.refine_max_edge_length,
                                                   refine_min_angle=args.refine_min_angle,
                                                   enforce_delaunay=args.enforce_delaunay)

    # elevate to 3D
    dem_profile, dem = watershed_workflow.get_raster_on_shape(sources['DEM'], shapes.exterior(), crs)
    mesh_points3 = watershed_workflow.elevate(mesh_points2, crs, dem, dem_profile)

    return shapes, rivers, (mesh_points3, mesh_tris)


if __name__ == '__main__':
    args = get_args()
    watershed_workflow.ui.setup_logging(args.verbosity, args.logfile)

    shapes, rivers, triangulation = mesh_shapes(args)
    fig, ax = watershed_workflow.bin_utils.plot_with_triangulation(args, shapes, rivers, triangulation)
    watershed_workflow.bin_utils.save(args, triangulation)
    logging.info("SUCESS")
    plt.show()
