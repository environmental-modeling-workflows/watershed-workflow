
import vtk
import numpy as np
import sortedcontainers
import attr
import time

reader = vtk.vtkUnstructuredGridReader()

reader.SetFileName('coweeta_basin2.vtk')
reader.Update()
mesh = reader.GetOutput()
ncells = mesh.GetNumberOfCells() #I want number of points, though
net = {c : [[],[]] for c in range(0,10851)} #needs to be allowed to grow! fix this.
for c in range(10851): #npts
    cell = mesh.GetCell(c)
    #print("cell",c)

 #   net = {c : [] for c in range(0,10)}
    cell = mesh.GetCell(c)
 #   print(net)
    PointId0 = cell.GetPointId(0) #get the corner IDs for a cell
    PointId1 = cell.GetPointId(1)
    PointId2 = cell.GetPointId(2)

    #if len(net[PointId0]) :
    net[PointId0][0] = (PointId1) #add 1 and 2 as neighbors to 0
    net[PointId0][1] = (PointId2)
    print("id0",net[PointId0])
    #if PointId1 not in ListOfLists:
    net[PointId1][0] = (PointId0) #add 0 and 2 as neighbors to 1
    net[PointId1][1] = (PointId2)
    print("id1",net[PointId1])
    #if PointId2 not in ListOfLists:
    net[PointId2][0] = (PointId0) #add 0 and 1 as neighbors to 2
    net[PointId2][1] = (PointId1)
    print("id2",net[PointId2])
    
print(net)
