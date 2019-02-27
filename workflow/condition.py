import numpy as np
import sortedcontainers
import attr
import time
import vtk
import find_neighbor

@attr.s
class Point:
    """POD struct that stores coords, a np array of length 3 (x,y,z) and neighbors, 
    a list of IDs of neighboring points.
    """
    coords = attr.ib()
    neighbors = attr.ib()


def condition(points, outletID):
    elev = sortedcontainers.SortedList(list(points.items()), key=lambda id_p:id_p[1].coords[2])
    waterway = set([outletID,])
    counter = 0
    while counter != 1000:
    #while len(elev) is not 0:
        counter += 1
        #print("len(elev), counter",len(elev),counter)
        spacer, current_point_metadata = elev.pop(0) #spacer is a placeholder. clean this up
        current = int(current_point_metadata.coords[0]) #current is the point
        current_neighbors = find_neighbor.GetNeighborIdsByPnt(current)
        #print("spacer",spacer,current)# we gotta fix this current issue... spacer vs current leads to loops
        if current in waterway:
            waterway.update(current_neighbors)

        else:
            ww_neighbors = [n for n in current_neighbors if n in waterway]
            #print ("ww_neighbors",ww_neighbors)
            if len(ww_neighbors) != 0: #update z for a point to: minimum neighboring z (from ww_neighbor)
                print("trying to fill")
                current_point_metadata.coords[2] = min(points[n].coords[2] for n in ww_neighbors)
                print("filling:",current,current_point_metadata.coords[2])

            else:
                current_point_metadata.coords[2] = min(points[n].coords[2] for n in current_neighbors)
            print("elev.add")
            elev.add((current, current_point_metadata))
    return

if __name__ == "__main__":
    def make_points_1D(elevs):
        points = {}
        IdSortedByZ = find_neighbor.GetIdSortedByZ()
        
        for i,e in enumerate(elevs):
            thisID = IdSortedByZ[i][0]
            coords = np.array([thisID,0,e])
            neighbors = find_neighbor.GetNeighborIdsByPnt(i)
            points[i] = Point(coords, neighbors)
        return points

    def run_test_1D(elev_in, elev_out):
        points = make_points_1D(elev_in)
        condition(points, 175)

    full_input = find_neighbor.GetIdSortedByZ()
    test_input = []

    for item in range(len(full_input)):
        test_input += full_input[item][1]

    input_list = [0,1,3,2,1,4]

    run_test_1D(test_input, [0,1,3,3,3,4])
