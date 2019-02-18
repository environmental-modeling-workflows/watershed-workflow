#This could be rewritten to use set-union operators. The solution to this can possibly be gleaned from here: https://stackoverflow.com/questions/2151517/pythonic-way-to-create-union-of-all-values-contained-in-multiple-lists

from vtk_io import read
import argparse    
import vtk
import numpy as np
import sortedcontainers
import attr
import time

meshId = read("coweeta_basin.vtk")

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName('coweeta_basin.vtk')
reader.Update()
mesh = reader.GetOutput()
ncells = mesh.GetNumberOfCells()
npoints = mesh.GetNumberOfPoints()



#NeighboringPointId is a set of lists, one list per point
NeighboringPointId = {p : [] for p in range(0,npoints)} 

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
    if PointId1 not in NeighboringPointId[PointId0]: 
        NeighboringPointId[PointId0] += [PointId1] 
    if PointId2 not in NeighboringPointId[PointId0]:
        NeighboringPointId[PointId0] += [PointId2] 
    if PointId0 not in NeighboringPointId[PointId1]:
        NeighboringPointId[PointId1] += [PointId0] 
    if PointId2 not in NeighboringPointId[PointId1]:
        NeighboringPointId[PointId1] += [PointId2]
    if PointId0 not in NeighboringPointId[PointId2]:
        NeighboringPointId[PointId2] += [PointId0] 
    if PointId1 not in NeighboringPointId[PointId2]:
        NeighboringPointId[PointId2] += [PointId1]
    else:continue

#post condition check to make sure there's no disconnected points (which would be a problem)
for i in range (npoints):
    if len(NeighboringPointId[i])==0:
         print("DISCONNECTED POINT:", i)
    else:
         print("This Id:",i," Neighbor Ids: ", NeighboringPointId[i])
         for neighbor in range(len(NeighboringPointId[i])):
             print("This Val:", meshId[0][i][2],"Neighbor Vals:",meshId[0][neighbor][2],"\n")

#print(NeighboringPointId)
