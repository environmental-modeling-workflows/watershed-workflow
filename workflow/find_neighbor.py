#This could be rewritten to use set-union operators. The solution to this can possibly be gleaned from here: https://stackoverflow.com/questions/2151517/pythonic-way-to-create-union-of-all-values-contained-in-multiple-lists

import vtk_io
import argparse    
import vtk
import numpy as np
import sortedcontainers
import attr
import time

meshId = vtk_io.read("coweeta_basin.vtk")

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName('coweeta_basin.vtk')
reader.Update()
mesh = reader.GetOutput()
ncells = mesh.GetNumberOfCells()
npoints = mesh.GetNumberOfPoints()

def constructPointStructure(npoints):
    PointStructure = {p : [] for p in range(0,npoints)}
    return PointStructure

def GetNeighborValues(PointId):
    neighborValue = []
    neighborIds = NeighboringPointId[PointId]
    for j in range(len(neighborIds)): #iterate over all neighbors
        thisNeighborId = neighborIds[j]
        neighborValue += [meshId[0][thisNeighborId][2]] # [2] is hardcoded to only provide z axis
    return(neighborValue)

def GetValue(PointId):
    value = [meshId[0][PointId][2]]
    return value    

def GetNeighborIdsByPnt(point):
    neighborIds = NeighboringPointId[point]
    return neighborIds

def GetNeighborIdsByCell(cell):
    PointId0 = cell.GetPointId(0) # get Ids for the three corners of a triangle
    PointId1 = cell.GetPointId(1)
    PointId2 = cell.GetPointId(2)

    #internal edges will be visited twice, so to prevent data duplication we use conditions    
    #conditions and appends are used in place of 'union' operators. could be improved
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

def GetIdSortedByZ():
    for thisPointId in range(npoints):
        Id_z_matrix[thisPointId] = [thisPointId,GetValue(thisPointId)] # get the Z axis value for an id 'i' 
    Id_z_matrix.sort(key=lambda x: x[1]) #sort the tuples by Z (stored in 1)
    return Id_z_matrix

#end functions 

#NeighboringPointId is a set of lists, one list per point
NeighboringPointId = constructPointStructure(npoints)
NeighborMatrix = [[0 for x in range(10)] for y in range(npoints)] 
Id_z_matrix = [0 for y in range(npoints)] 

#iterate over all cells
for c in range(ncells): 
    #get data in given cell 'c'
    cell = mesh.GetCell(c) 
    GetNeighborIdsByCell(cell)

#post condition check to make sure there's no disconnected points (which would be a problem)
#for i in range (npoints):
    #if len(NeighboringPointId[i])<=1:
        #print("DISCONNECTED POINT:", i)
    #if len(NeighboringPointId[i])>=10:
    #     print("This mesh is looking very strange...i (pointid) = ", i, "neighboringpointid=",NeighboringPointId[i])
    #else:
Id_z_matrix = GetIdSortedByZ()
#         Neighbor_Z_Values = GetNeighborValues(i)
      #  print("i=",i,"neighbor z values:",Neighbor_Z_Values,"\n")
      #   print(Neighbor_Z_Values)

#         Id_z_matrix[i] = [i,GetValue(i)]
#         NeighborMatrix[i]=[i,Neighbor_Z_Values]

#Id_z_matrix.sort(key=lambda x: x[1])
#print("ID Matrix:",Id_z_matrix)
