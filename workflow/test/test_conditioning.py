import pytest
import numpy as np
import logging

import workflow.condition

def make_points_1D(elevs):
    points = {}
    for i,e in enumerate(elevs):
        coords = np.array([i,0,e])
        if i == 0:
            neighbors = [1,]
        elif i == len(elevs)-1:
            neighbors = [i-1,]
        else:
            neighbors = [i-1,i+1]
        points[i] = workflow.condition.Point(coords, neighbors)
    return points

def run_test_1D(elev_in, elev_out, alg):
    points = make_points_1D(elev_in)
    if alg == 1:
        workflow.condition.condition1(points, 0)
    elif alg == 2:
        workflow.condition.condition2(points, 0)
    elif alg == 3:
        workflow.condition.condition3(points, 0)

    print("GOT COORDS:")
    print(([points[i].coords[2] for i in range(len(points))]))
    
    for i in range(len(elev_in)):
        assert(points[i].coords[2] == elev_out[i])

def test_null():
    run_test_1D([0,1,2,3,4,5], [0,1,2,3,4,5], 1)
    run_test_1D([0,1,2,3,4,5], [0,1,2,3,4,5], 2)
    run_test_1D([0,1,2,3,4,5], [0,1,2,3,4,5], 3)

def test_one_pit():
    run_test_1D([0,1,3,2,4,5], [0,1,3,3,4,5], 1)
    run_test_1D([0,1,3,2,4,5], [0,1,3,3,4,5], 2)
    run_test_1D([0,1,3,2,4,5], [0,1,3,3,4,5], 3)

def test_two_pit():
    run_test_1D([0,1,3,2,1,4], [0,1,3,3,3,4], 1)
    run_test_1D([0,1,3,2,1,4], [0,1,3,3,3,4], 2)
    run_test_1D([0,1,3,2,1,4], [0,1,3,3,3,4], 3)

def test_two_pit_backwards():
    run_test_1D([0,1,3,1,2,4], [0,1,3,3,3,4], 1)
    run_test_1D([0,1,3,1,2,4], [0,1,3,3,3,4], 2)
    run_test_1D([0,1,3,1,2,4], [0,1,3,3,3,4], 3)

def test_double_pit():
    run_test_1D([0,1,2,1,3,1,3,5], [0,1,2,2,3,3,3,5], 1)
    run_test_1D([0,1,2,1,3,1,3,5], [0,1,2,2,3,3,3,5], 2)
    run_test_1D([0,1,2,1,3,1,3,5], [0,1,2,2,3,3,3,5], 3)

def test_bad_outlet_pit():
    run_test_1D([0,1,3,-1,4,5], [0,1,3,3,4,5], 1)
    run_test_1D([0,1,3,-1,4,5], [0,1,3,3,4,5], 2)
    run_test_1D([0,1,3,-1,4,5], [0,1,3,3,4,5], 3)
    


    
