#!/usr/bin/env python3
"""Downloads and meshes shapes based upon hydrography data."""

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
    workflow.ui.inshape_args(parser)
    workflow.ui.huc_hint_options(parser)
    
    workflow.ui.outmesh_args(parser)
    workflow.ui.center_options(parser)

    workflow.ui.simplify_options(parser)
    workflow.ui.refine_options(parser)

    data_ui = parser.add_argument_group('Data Sources')
    workflow.ui.huc_source_options(data_ui)
    workflow.ui.hydro_source_options(data_ui)
    workflow.ui.dem_source_options(data_ui)
    
    # parse args, log
    return parser.parse_args()

def mesh_shapes(args):
    workflow.ui.setup_logging(args.verbosity, args.logfile)
    sources = workflow.source_list.get_sources(args)

    logging.info("")
    logging.info("Meshing shapes from: {}".format(args.input_file))
    logging.info("  with index: {}".format(args.shape_index))
    logging.info("="*30)
    logging.info('Target projection: "{}"'.format(args.projection['init']))
    
    # collect data
    shapes, centroid = workflow.get_split_form_shapes(args.input_file, args.shape_index, args.projection, args.center)
    shapes_boundary = shapes.exterior()
    hucstr = workflow.find_huc(sources['HUC'], shapes.exterior(), args.projection, args.hint)
    logging.info("found shapes in HUC %s"%hucstr)    
    rivers, centroid = workflow.get_rivers_by_bounds(sources['hydrography'], shapes_boundary.bounds, args.projection, hucstr, centering=centroid)
    rivers = workflow.simplify_and_prune(shapes, rivers, args)
    
    # make 2D mesh
    mesh_points2, mesh_tris = workflow.triangulate(shapes, rivers, args)

    # elevate to 3D
    if args.center:
        mesh_points2_uncentered = mesh_points2 + np.expand_dims(np.array(centroid.coords[0]),0)
    else:
        mesh_points2_uncentered = mesh_points2

    dem_profile, dem = workflow.get_raster_on_shape(sources['DEM'], shapes_boundary, args.projection)
    mesh_points3_uncentered = workflow.elevate(mesh_points2_uncentered, dem, dem_profile)

    if args.center:
        mesh_points3 = np.empty(mesh_points3_uncentered.shape,'d')
        mesh_points3[:,0:2] = mesh_points2
        mesh_points3[:,2] = mesh_points3_uncentered[:,2]
    else:
        mesh_points3 = mesh_points3_uncentered

    return centroid, shapes, rivers, (mesh_points3, mesh_tris)



if __name__ == '__main__':
    try:
        args = get_args()
        centroid, shapes, rivers, triangulation = mesh_shapes(args)
        workflow.bin_utils.plot(args, shapes, rivers, triangulation)
        workflow.bin_utils.save(args, centroid, triangulation)
        logging.info("SUCESS")
        plt.show()
        sys.exit(0)
    except KeyboardInterrupt:
        logging.error("Keyboard Interupt, stopping.")
        sys.exit(0)
    except Exception as err:
        logging.error('{}'.format(str(err)))
        #sys.exit(1)
        raise err
        
