import pytest
import numpy as np
import math
import shapely 


import watershed_workflow.mesh
import watershed_workflow.utils
import watershed_workflow.densify_rivers_hucs
import watershed_workflow.river_tree
import watershed_workflow.split_hucs

def assert_list_same(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    assert(len(l1) == len(l2))
    for a,b in zip(l1,l2):
        assert(a == b)

def elevate_open_book(points):
    points3=np.zeros((len(points),3))
    points3[:,:2]=points
    points3[:,2]=9+abs(points[:,0]-200)/800+points[:,1]/500 
    return points3


def watershed_poly(): 
    """Create watershed polygon, mocking NHDPLus dataset"""
    ws1= shapely.geometry.Polygon(100*np.array([[0,0],[1,0],[3,0],[4,0],[4,1],[4,2],[4,3],[4,4],[3,4.5],[2,5],[1,4.5],[0,4], [0,3], [0,2] ,[0,1]], 'd'))
    return ws1


def reaches():
    """Create a list of reaches, mocking NHDPLus dataset"""
    reach1=shapely.geometry.LineString([(200,200),(200,0)])
    reach2=shapely.geometry.LineString([ (50,300), (100,300), (100,200),(200,200)])
    reach3=shapely.geometry.LineString([ (350,400),(350,300),(300,300),(300,200),(200,200)])
    reach4=shapely.geometry.LineString([(100,400), (200,300)])
    reaches=[reach1, reach2, reach3, reach4]
    return reaches

@pytest.fixture
def watershed_rivers():
    my_hucs=[watershed_poly()]
    watershed = watershed_workflow.split_hucs.SplitHUCs(my_hucs) 
    rivers = watershed_workflow.construct_rivers(watershed, reaches(), method='geometry')   
    return watershed, rivers


def test_river_meshing(watershed_rivers):
    watershed, rivers = watershed_rivers
    watershed_workflow.simplify(watershed, rivers, simplify_rivers=10, snap=False, cut_intersections=True)    

    #densify watershed and river
    watershed=watershed_workflow.densify_rivers_hucs.densify_hucs(watershed=watershed, rivers=rivers, use_original= False,limit_scales=[0,25,100,50])
    rivers=watershed_workflow.densify_rivers_hucs.densify_rivers(rivers, rivers, limit=14, use_original=False, treat_collinearity=True)

    assert(58 == len(watershed.segments[0].coords))
    assert(16 == len(rivers[0].segment.coords))
    assert(12 == len(rivers[1].segment.coords))

    # creating river corridor and defining quads
    quads, corrs = watershed_workflow.river_tree.create_rivers_meshes(rivers=rivers, widths=5, junction_treatment=True)

    assert(74 == len(quads))
    assert_list_same([137, 138, 139], quads[-1])
    assert_list_same([14, 15, 55, 111, 112], quads[14])
    assert(math.isclose(3925.0000, corrs[0].area, rel_tol=1e-4))
    assert(math.isclose(674.9656, corrs[1].area, rel_tol=1e-4))
    assert(128 == len(corrs[0].exterior.coords))

    # triangulating with rivers as holes
    d0 = 12; d1 = 18; A0 = 82; A1 = 500
    points, elems= watershed_workflow.triangulate(watershed, rivers, corrs, mesh_rivers=True,
                                              refine_min_angle=32, refine_distance=[d0,A0,d1,A1], diagnostics=False)
    areas = np.array([watershed_workflow.utils.triangle_area(points[e]) for e in elems])

    assert(703 == len(points))
    assert(1219 == len(elems))
    assert(math.isclose(20.711983, min(areas), rel_tol=1e-4))
    assert(math.isclose(759.140041, max(areas), rel_tol=1e-4))

    # elevate in 3d
    points3=elevate_open_book(points)

    # adding quads to elem list
    elems_list=[elem.tolist() for elem in list(elems)]
    for elem in quads:
        elems_list.append(elem)
    
    m2 = watershed_workflow.mesh.Mesh2D(points3.copy(), elems_list)

    assert(1293 == m2.num_cells)
    assert(1995 == m2.num_edges)
    assert(703 == m2.num_nodes)
    assert(39 == len(m2.boundary_edges))
    assert(39 == len(m2.boundary_nodes))
    assert(5 == len(m2.conn[1233]))
    assert(np.allclose(np.array([130.17369346, 359.37985407, 9.47270926]), m2.compute_centroid(0), 1.e-6))
    assert(np.allclose(np.array([35.72548705, 16.33090195,  9.23800495]), m2.compute_centroid(1), 1.e-6))    
