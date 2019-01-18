import numpy as np
import sortedcontainers
import attr
import time

@attr.s
class Point:
    """POD struct that stores coords, a np array of length 3 (x,y,z) and neighbors, 
    a list of IDs of neighboring points.
    """
    coords = attr.ib()
    neighbors = attr.ib()


def condition(points, outletID):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet
    """

    # create a sorted list of elevations, from largest to smallest
    elev = sortedcontainers.SortedList(list(points.items()), key=lambda id_p:id_p[1].coords[2])
    waterway = set([outletID,])

    # loop over elevation list from small to large
    while len(elev) is not 0:
        current, current_p = elev.pop(0)
        if current in waterway:
            # still in the waterway
            waterway.update(current_p.neighbors)
        else:
            # not in the waterway, fill
            ww_neighbors = [n for n in current_p.neighbors if n in waterway]
            if len(ww_neighbors) != 0:
                current_p.coords[2] = min(points[n].coords[2] for n in ww_neighbors)
            else:
                current_p.coords[2] = min(points[n].coords[2] for n in current_p.neighbors)
            
            # push back into elev list with new, higher elevation
            elev.add( (current,current_p) )
    return


# if __name__ == "__main__":
#     def make_points_1D(elevs):
#         points = {}
#         for i,e in enumerate(elevs):
#             coords = np.array([i,0,e])
#             if i == 0:
#                 neighbors = [1,]
#             elif i == len(elevs)-1:
#                 neighbors = [i-1,]
#             else:
#                 neighbors = [i-1,i+1]
#             points[i] = Point(coords, neighbors)
#         return points

#     def run_test_1D(elev_in, elev_out):
#         points = make_points_1D(elev_in)
#         condition(points, 0)

#         for i in range(len(elev_in)):
#             assert(points[i].coords[2] == elev_out[i])

#     run_test_1D([0,1,3,2,1,4], [0,1,3,3,3,4])

    
