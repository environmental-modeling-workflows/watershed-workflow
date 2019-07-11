#!/usr/bin/env python3
"""
Conditions DEMs from VTK files

python condition_dem.py my_file.vtk my_conditioned_file.vtk
"""


import workflow.condition
import workflow.find_neighbor
import argparse
import numpy as np

from workflow_tpls import vtk_io


if __name__ == '__main__':
    mesh = vtk_io.read('coweeta_basin.vtk')
    npoints = len(mesh[0])
    neighbors = workflow.find_neighbor.findNeighbors(npoints, mesh[1])

    coords = mesh[0].copy()
    points = dict((i, workflow.condition.Point(c, neighbors[i])) for i, c in enumerate(coords))
    workflow.condition.condition(points, 34, 3)

    coords_old = mesh[0]
    coords_new = np.array([points[i].coords for i in range(npoints)])

    print("shape = ", coords_old.shape)
    print("coordinate difference (xy): ", (np.abs(coords_new[:,0:2] - coords_old[:,0:2])).max())
    print("coordinate difference (z): ", (coords_new[:,2] - coords_old[:,2]).max(),  (coords_new[:,2] - coords_old[:,2]).min())

    vtk_io.write('coweeta_basin_conditioned.vtk',coords_new, mesh[1])
