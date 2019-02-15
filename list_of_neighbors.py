#This could be rewritten to use set-union operators. The solution to this can possibly be gleaned from here: https://stackoverflow.com/questions/2151517/pythonic-way-to-create-union-of-all-values-contained-in-multiple-lists
import vtk
import numpy as np
import sortedcontainers
import attr
import time

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName('coweeta_basin2.vtk')
reader.Update()
mesh = reader.GetOutput()
ncells = mesh.GetNumberOfCells()
npoints = mesh.GetNumberOfPoints()

#net is a set of lists, one list per point
net = {p : [] for p in range(0,npoints)} 

#iterate over all cells
for c in range(ncells): 
    #get data in given cell 'c'
    cell = mesh.GetCell(c) 

    #triangle means that three corners of a cell 'c' are stored
    PointId0 = cell.GetPointId(0) 
    PointId1 = cell.GetPointId(1)
    PointId2 = cell.GetPointId(2)

    #internal edges will be visited twice, so to prevent data duplication we use conditions    
    #conditions and appends are used in place of 'union' operators. could be improved
    #if it's not already there, we store the ID for the connected points, in one cell.
    if PointId1 not in net[PointId0]: 
        net[PointId0] += [PointId1] 
    if PointId2 not in net[PointId0]:
        net[PointId0] += [PointId2] 
    if PointId0 not in net[PointId1]:
        net[PointId1] += [PointId0] 
    if PointId2 not in net[PointId1]:
        net[PointId1] += [PointId2]
    if PointId0 not in net[PointId2]:
        net[PointId2] += [PointId0] 
    if PointId1 not in net[PointId2]:
        net[PointId2] += [PointId1]
    else:continue

#post condition check to make sure there's no disconnected points (which would be a problem)
for i in range (npoints):
    if len(net[i])==0:
         print("DISCONNECTED POINT:", i)

print(net)
