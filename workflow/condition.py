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
    elev = sortedcontainers.SortedList(list(points.items()), key=lambda id_p:id_p[1].coords[2]) # modify this to be x value is Id
    #print(elev)
    waterway = set([outletID,])
    counter = 0
    #while counter is not 10:
    while len(elev) is not 0:
        counter += 1
        current, current_point_metadata = elev.pop(0)
        if current in waterway:# if current is already known to be in waterway, add current.neighbors to waterway
            waterway.update(current_point_metadata.neighbors) 
        else: #determine if current.neighbors are connected to a waterway
            ww_neighbors = []
            for n in range(len(current_point_metadata.neighbors)): #construct waterway neighbors dict
                if current_point_metadata.neighbors[n] in waterway:
                    ww_neighbors.append(current_point_metadata.neighbors[n])
                else: continue #continue and deconstruct waterway neighbors dict
            if len(ww_neighbors) != 0: #update z for a point to: minimum neighboring z (from ww_neighbor)
                PId = (int(current_point_metadata.coords[0])) #the ID of the point (converted to int)
                old_z = (current_point_metadata.coords[2]) #before
                current_point_metadata.coords[2] = min(points[n].coords[2] for n in ww_neighbors)
                new_z = current_point_metadata.coords[2] #after
                print("Filling: PId",PId,"| ",old_z, " -> ", new_z)
            else:#not in waterway - still we update pit to neighboring elevations
                neighboring_values = []
                for n in range(len(current_point_metadata.neighbors)):
                    neighboring_values += current_point_metadata.neighbors #data duplication may occur here
                min_neighbor_id = min(neighboring_values)
                min_neighbor_value_pre = find_neighbor.GetValue(min_neighbor_id)
                min_neighbor_value = str(min_neighbor_value_pre)[1:-1] #this is to strip brackets.  i.e. str[brackets][1:-1] = brackets
                current_point_metadata.coords[2] = min_neighbor_value
    elev.add( (current,current_point_metadata) )
    return

if __name__ == "__main__":
    def make_points_1D(elevs):
        points = {}
        IdSortedByZ = find_neighbor.GetIdSortedByZ()
        
        for i,e in enumerate(elevs):
            #print(e)
            thisID = IdSortedByZ[i][0]

            coords = np.array([thisID,0,e])
            neighbors = find_neighbor.GetNeighborIdsByPnt(i)
            points[i] = Point(coords, neighbors)
        return points

    def run_test_1D(elev_in, elev_out):
        points = make_points_1D(elev_in)
        condition(points, 24)

        #for i in range(len(elev_in)):
        #    assert(points[i].coords[2] == elev_out[i])
        #print("Run 2")
        #condition(points, 180) #look  below for more options!

    full_input = find_neighbor.GetIdSortedByZ()
    test_input = []

    for item in range(len(full_input)):
        test_input += full_input[item][1]

    input_list = [0,1,3,2,1,4]

    run_test_1D(test_input, [0,1,3,3,3,4])
