import pytest
import shapely
from matplotlib import pyplot as plt
import cartopy.crs

import workflow.plot
import workflow.warp
import workflow.conf

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

show = True

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
    ax = workflow.plot.get_ax(workflow.conf.get_crs(epsg_ax))
    if epsg_ax is not None:
        ax.stock_img()

    if epsg_data is not None:
        crs = workflow.conf.get_crs(epsg_data)
        objs = workflow.warp.warp_shapelys(obj_gen(start_p), workflow.conf.latlon_crs(), crs)
    else:
        crs = None
        objs = obj_gen(start_p)
    workflow.plot.shply(objs, crs, 'r', ax=ax)

def test_points(points):
    for epsg_data, epsg_ax in crss:
        run_test(point(), points, epsg_data, epsg_ax)
    for epsg_data, epsg_ax in crss_ak:
        run_test(point_ak(), points, epsg_data, epsg_ax)
    if show:
        plt.show()
    
def test_lines(lines):
    for epsg_data, epsg_ax in crss:
        run_test(point(), lines, epsg_data, epsg_ax)
    for epsg_data, epsg_ax in crss_ak:
        run_test(point_ak(), lines, epsg_data, epsg_ax)
    if show:
        plt.show()


def test_polygons(polygons):
    for epsg_data, epsg_ax in crss:
        run_test(point(), polygons, epsg_data, epsg_ax)
    for epsg_data, epsg_ax in crss_ak:
        run_test(point_ak(), polygons, epsg_data, epsg_ax)
    if show:
        plt.show()

    




    
