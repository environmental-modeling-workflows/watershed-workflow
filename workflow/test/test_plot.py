import os
import pytest
import shapely
from matplotlib import pyplot as plt
import cartopy.crs
import numpy.testing as npt

from crs_fixtures import point, point_ak, shift, points, lines, polygons

import workflow.plot
import workflow.warp
import workflow.conf
import pickle

crss = [(None, None),
        (4269, None),
        (5070, None),
        (4269, 4269),
        (5070, 4269),
        (5070, 5070),
        (4269, 5070),
        (26913, 5070)]

crss_ak = [(3338, 3338),
           (5070, 3338),
           (4269, 3338)]

show = False
new_gold = False
check_gold = True
fig = None
if not show:
    fig = plt.figure()


import collections
def default_dict():
    return collections.defaultdict(default_dict)
pickle_file_name = os.path.join('workflow','test', 'test_plot_gold.pickle')

if new_gold:
    gold = default_dict()
else:
    import pickle
    with open(pickle_file_name, 'rb') as fid:
        gold = pickle.load(fid)



def point():
    return shapely.geometry.Point(-90, 38)

def point_ak():
    return shapely.geometry.Point(-147, 65)
    
def shift(p, t):
    return shapely.geometry.Point(p.xy[0][0] + t[0], p.xy[1][0] + t[1])

@pytest.fixture
def points():
    def _points(p):
        ps = [
            p,
            shift(p, (2,0)),
            shift(p, (1,1)),              
            ]
        return ps
    return _points

@pytest.fixture
def lines():
    def _lines(p):
        ls = [
            shapely.geometry.LineString([p, shift(p,(0,1)), shift(p,(0,2))]),
            shapely.geometry.LineString([p, shift(p,(1,0)), shift(p,(2,0))]),
            shapely.geometry.LineString([p, shift(p,(1,1)), shift(p,(2,2))]),
        ]
        return ls
    return _lines
    
@pytest.fixture
def polygons():
    def _polygons(p):
        polys = [
            shapely.geometry.Polygon([[p.x, p.y] for p in [ p, shift(p, (-1,0)), shift(p, (-1,-1)), shift(p, (0,-1)), p]]),
            shapely.geometry.Polygon([[p.x, p.y] for p in [ p, shift(p,  (1,0)), shift(p, (1,-1)), shift(p, (0,-1)), p]]),
            shapely.geometry.Polygon([[p.x, p.y] for p in [ p, shift(p, (-1,1)), shift(p, (0,2)), shift(p, (1,1)), p]]),
        ]
        return polys
    return _polygons    

def run_test(start_p, obj_gen, epsg_data, epsg_ax):
    print("Running test from {} to {}".format(epsg_data, epsg_ax))
    if show:
        fig = plt.figure()
    else:
        fig = globals()['fig']

    if epsg_ax is not None:
        epsg_ax = workflow.crs.from_epsg(epsg_ax)
    ax = workflow.plot.get_ax(epsg_ax, fig)
    if epsg_ax is not None:
        ax.stock_img()

    if epsg_data is not None:
        crs = workflow.crs.from_epsg(epsg_data)
        objs = workflow.warp.shplys(obj_gen(start_p), workflow.crs.latlon_crs(), crs)
    else:
        epsg_data = 'None'
        crs = None
        objs = obj_gen(start_p)
    res = workflow.plot.shply(objs, crs, 'r', ax=ax)

    if new_gold:
        if hasattr(res, 'get_paths'):
            for i,p in enumerate(res.get_paths()):
                gold[str(start_p)][obj_gen.__name__][epsg_data][i] = p.vertices
        else:
            gold[str(start_p)][obj_gen.__name__][epsg_data] = res.get_path().vertices
            
        with open(pickle_file_name, 'wb') as fid:
            pickle.dump(gold, fid)
    elif check_gold:
        if hasattr(res, 'get_paths'):
            for i,p in enumerate(res.get_paths()):
                npt.assert_allclose(gold[str(start_p)][obj_gen.__name__][epsg_data][i], p.vertices)
        else:
            npt.assert_allclose(gold[str(start_p)][obj_gen.__name__][epsg_data], res.get_path().vertices)

    if not show:
        fig.clear()
    else:
        plt.show()


def test_points(points):
    for epsg_data, epsg_ax in crss:
        run_test(point(), points, epsg_data, epsg_ax)
    for epsg_data, epsg_ax in crss_ak:
        run_test(point_ak(), points, epsg_data, epsg_ax)
    
def test_lines(lines):
    for epsg_data, epsg_ax in crss:
        run_test(point(), lines, epsg_data, epsg_ax)
    for epsg_data, epsg_ax in crss_ak:
        run_test(point_ak(), lines, epsg_data, epsg_ax)

def test_polygons(polygons):
    for epsg_data, epsg_ax in crss:
        run_test(point(), polygons, epsg_data, epsg_ax)
    for epsg_data, epsg_ax in crss_ak:
        run_test(point_ak(), polygons, epsg_data, epsg_ax)


    




    
