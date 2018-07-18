#!/usr/bin/env python3
"""Downloads and meshes HUC and hydrography data.

Default data for HUCs comes from The National Map's Watershed Boundary Dataset (WBD).
Default data for hydrography comes from The National Map's National Hydrography Dataset (NHD).
See: "https://nhd.usgs.gov/"

Default DEMs come from the National Elevation Dataset (NED).
See: "https://lta.cr.usgs.gov/NED"
"""

import os,sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
import shapely

import workflow.hilev
import workflow.ui


if __name__ == '__main__':
    # set up parser
    parser = workflow.ui.get_basic_argparse(__doc__)
    workflow.ui.outmesh_options(parser)
    workflow.ui.simplify_options(parser)
    workflow.ui.refine_options(parser)
    workflow.ui.huc_args(parser)

    # parse args, log
    args = parser.parse_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)
    
    # collect data
    containing_huc, hucs, centroid = workflow.hilev.get_hucs(args.HUC, True)
    rivers = workflow.hilev.get_rivers(args.HUC)
    dem_profile, dem = workflow.hilev.get_dem(containing_huc)

    # make 2D mesh
    rivers = [shapely.affinity.translate(r, -centroid.coords[0][0], -centroid.coords[0][1]) for r in rivers]
    rivers = workflow.hilev.simplify_and_prune(hucs, rivers, args)
    mesh_points2, mesh_tris = workflow.hilev.triangulate(hucs,rivers, args)

    # elevate to 3D
    mesh_points2_uncentered = mesh_points2 + np.expand_dims(np.array(centroid.coords[0]),0)
    mesh_points3_uncentered = workflow.hilev.elevate(mesh_points2_uncentered, dem, dem_profile)
    mesh_points3 = np.empty(mesh_points3_uncentered.shape,'d')
    mesh_points3[:,0:2] = mesh_points2
    mesh_points3[:,2] = mesh_points3_uncentered[:,2]

    # plot the result
    if args.verbosity > 0:
        plt.figure()
        workflow.plot.triangulation(mesh_points3, mesh_tris, linewidth=0.5)
        workflow.plot.hucs(hucs, 'k')
        workflow.plot.rivers(rivers)
        plt.gca().set_aspect('equal', 'datalim')    
        plt.show()

    # save mesh
    if args.outfile is None:
        outdir = "data/meshes"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, 'huc_%s.vtk'%args.HUC)
    else:
        outfile = args.outfile            
    workflow.hilev.save(outfile, mesh_points3_uncentered, mesh_tris)
    sys.exit(0)
