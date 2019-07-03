#This could be rewritten to use set-union operators. The solution to this can possibly be gleaned from here: https://stackoverflow.com/questions/2151517/pythonic-way-to-create-union-of-all-values-contained-in-multiple-lists

import collections
    
def findNeighbors(npoints, simplex_dict):
    """Document me!"""
    NeighboringPointId = collections.defaultdict(set)

    for simplex_set in simplex_dict.values():
        for cell in simplex_set:
            PointId0, PointId1, PointId2 = tuple(cell)

            NeighboringPointId[PointId0].add(PointId1)
            NeighboringPointId[PointId0].add(PointId2)            

            NeighboringPointId[PointId1].add(PointId0)
            NeighboringPointId[PointId1].add(PointId2)            

            NeighboringPointId[PointId2].add(PointId0)
            NeighboringPointId[PointId2].add(PointId1)

            # generalize!
            # for i,p in enumerate(cell):
            #     if i == len(cell)-1:
            #         q = cell[0]
            #     else:
            #         q = cell[i+1]

                # edge p,q is one edge of the cell
                # stick p in q's neighbors
                # stick q in p's neighbors
                
    
    return NeighboringPointId


def findNeighborsAndBoundary(npoints, simplex_dict):
    """Document me!"""
    NeighboringPointId = collections.defaultdict(set)
    EdgeCount = collections.defaultdict(int)

    for simplex_set in simplex_dict.values():
        for cell in simplex_set:
            PointId0, PointId1, PointId2 = tuple(cell)


            NeighboringPointId[PointId0].add(PointId1)
            NeighboringPointId[PointId0].add(PointId2)            

            NeighboringPointId[PointId1].add(PointId0)
            NeighboringPointId[PointId1].add(PointId2)            

            NeighboringPointId[PointId2].add(PointId0)
            NeighboringPointId[PointId2].add(PointId1)

            # generalize!
            # for i,p in enumerate(cell):
            #     if i == len(cell)-1:
            #         q = cell[0]
            #     else:
            #         q = cell[i+1]

                # edge p,q is one edge of the cell
                # stick p in q's neighbors
                # stick q in p's neighbors
                # stick (p,q) or (q,p) into EdgeCount

            if PointId0 < PointId1:
                e = PointId0, PointId1
            else:
                e = PointId1, PointId0
            EdgeCount[e] += 1
            # and add the other two edges!
            # generalize this too!

                
    boundary_edges = [e for e,count in EdgeCount.items() if count == 1]
    boundary_nodes = set([p for e in boundary_edges for p in e])
    return NeighboringPointId, boundary_nodes


