
import vtk
import numpy as np
import sortedcontainers
import attr
import time

reader = vtk.vtkUnstructuredGridReader()

reader.SetFileName('coweeta_basin.vtk')
reader.Update()
mesh = reader.GetOutput()
ncells = mesh.GetNumberOfCells()
npoints = mesh.GetNumberOfPoints()

net = {c : [] for c in range(0,npoints)}
for c in range(npoints): #npts loop over npoints? or ncells?
    cell = mesh.GetCell(c) #load data for THIS cell in loop
    PointId0 = cell.GetPointId(0) #get the corner IDs for a cell
    PointId1 = cell.GetPointId(1)
    PointId2 = cell.GetPointId(2)

    if PointId1 not in net[PointId0]: # case 0 'do not continue on edges which we have already encountered'
        net[PointId0] += [PointId1] # for each corner ID, we make a list
    if PointId2 not in net[PointId0]:
        net[PointId0] += [PointId2] # in these lists we append the ID of the neighbors, one cell at a time
    if PointId0 not in net[PointId1]:
        net[PointId1] += [PointId0] #add 0 and 2 as neighbors to 1
    if PointId2 not in net[PointId1]:
        net[PointId1] += [PointId2]
    if PointId0 not in net[PointId2]:
        net[PointId2] += [PointId0] #add 0 and 1 as neighbors to 2
    if PointId1 not in net[PointId2]:
        net[PointId2] += [PointId1]
        
print(net)
