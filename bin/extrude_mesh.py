#!/usr/bin/env python3
"""Vertically extrudes a VTK surface mesh.

Note that this script is quite simple, and is not as rich as the
possible use of the extrude capability here.  It is quite likely that
you don't really want to use this, and should instead write your own
script or use a Jupyter notebook to do your own extrusion process.
The command line simply isn't rich enough to express things
efficiently.

Mostly this script exists for testing and debugging.
"""

import sys,os
import logging

import workflow.ui
import workflow.extrude

def get_args():
    parser = workflow.ui.get_basic_argparse(__doc__)
    parser.add_argument("-n", "--num-cells", default=10, type=int,
                        help="number of cells to extrude")
    parser.add_argument("-d", "--depth", default=40.0, type=float,
                        help="depth to extrude")
    parser.add_argument("-p", "--plot", default=False, action="store_true",
                        help="plot the 2D mesh")
    parser.add_argument("input_file", type=workflow.ui.vtkfile,
                        help="input filename of surface mesh (expects VTK)")
    parser.add_argument("output_file", type=str,
                        help="output filename (expects EXO)")

    args = parser.parse_args()

    if os.path.isfile(args.output_file):
        print('Output file "%s" exists, cowardly not overwriting.'%args.output_file)
        sys.exit(1)

    return args
    

if __name__ == "__main__":
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)
        
    logging.info("Reading file: {}".format(args.input_file))
    m2 = workflow.extrude.Mesh2D.read_VTK(args.input_file)
    if args.plot:
        m2.plot()

    logging.info("Extruding:")
    extrusion = ['constant',], [args.depth,], [args.num_cells,], [101,]
    workflow.extrude.Mesh3D.summarize_extrusion(*extrusion)
    m3 = workflow.extrude.Mesh3D.extruded_Mesh2D(m2, *extrusion)

    logging.info("Writing file: {}".format(args.output_file))
    m3.write_exodus(args.output_file)
