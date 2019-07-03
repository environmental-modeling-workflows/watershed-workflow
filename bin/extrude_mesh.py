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
import argparse
import workflow.extrude

def commandline_options():
    parser = argparse.ArgumentParser(description='Extrude a 2D mesh to make a 3D mesh')
    parser.add_argument("-n", "--num-cells", default=10, type=int,
                        help="number of cells to extrude")
    parser.add_argument("-d", "--depth", default=40.0, type=float,
                        help="depth to extrude")
    parser.add_argument("-p", "--plot", default=False, action="store_true",
                        help="plot the 2D mesh")
    parser.add_argument("input_file", type=str,
                        help="input filename of surface mesh (expects VTK)")
    parser.add_argument("output_file", type=str,
                        help="output filename (expects EXO)")

    options = parser.parse_args()

    if options.output_file is None:
        options.output_file = ".".join(options.input_file.split(".")[:-1])+".exo"
    

    if os.path.isfile(options.output_file):
        print('Output file "%s" exists, cowardly not overwriting.'%options.output_file)
        sys.exit(1)

    if not os.path.isfile(options.input_file):
        print('No input file provided')
        parser.print_usage()
        sys.exit(1)

    return options
    

if __name__ == "__main__":
    options = commandline_options()
        
    m2 = workflow.extrude.Mesh2D.read_VTK(options.input_file)
    if options.plot:
        m2.plot()
    m3 = workflow.extrude.Mesh3D.extruded_Mesh2D(m2, ['constant',], [options.depth,], [options.num_cells,], [10000,], )
    m3.write_exodus(options.output_file)
