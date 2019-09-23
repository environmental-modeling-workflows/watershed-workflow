#!/usr/bin/env python3
"""Conditions unstructured DEMs from VTK."""


import argparse
import logging

import workflow.ui
import workflow.condition
import workflow.extrude

def get_args():
    parser = workflow.ui.get_basic_argparse(__doc__)
    parser.add_argument('input_file',
                        type=workflow.ui.vtkfile, help='Input VTK file')
    parser.add_argument('output_file',
                        type=str, help='Output VTK file')
    parser.add_argument('--outlet_node', type=int,
                        help='Outlet node index (default searches boundary for lowest point).')

    # parse args, log
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    workflow.ui.setup_logging(args.verbosity, args.logfile)

    logging.info("Reading file: {}".format(args.input_file))
    m2 = workflow.extrude.Mesh2D.read_VTK(args.input_file)

    coords_old = m2.coords[:]
    workflow.condition.condition(m2, args.outlet_node)
    coords_new = m2.coords
    
    logging.info("max/min coordinate difference: {} / {}".format((coords_new[:,2] - coords_old[:,2]).max(),  (coords_new[:,2] - coords_old[:,2]).min()))

    logging.info("Writing file: {}".format(args.output_file))
    m2.write_VTK(args.output_file)
