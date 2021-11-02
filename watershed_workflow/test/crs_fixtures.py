import pytest
import shapely


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
