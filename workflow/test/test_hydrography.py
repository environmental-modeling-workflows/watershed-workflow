import pytest
import shapely.geometry

from workflow.test.shapes import *

import workflow.utils
import workflow.hucs
import workflow.hydrography
import workflow.tree

def test_null_cleanup(rivers):
    """Tests that cleanup on nice river network does nothing"""
    riversc = workflow.hydrography.quick_cleanup(rivers)
    print(type(rivers))
    print(type(riversc))
    assert_close(riversc, rivers)

def test_close_cleanup(rivers):
    """Tests that cleanup can remove close points"""
    extra = shapely.geometry.LineString([(15,-3.00000001), (15,-3)])
    rivers_wextra = shapely.geometry.MultiLineString(list(rivers)+[extra,])
    rivers_clean = workflow.hydrography.quick_cleanup(rivers_wextra)
    assert_close(rivers_clean, rivers, 0.1)

def test_pruning_cleanup(two_boxes, rivers):
    """Tests that cleanup can remove close points"""
    extra = shapely.geometry.LineString([(12,-3.01), (12,-3)])
    rivers_wextra = shapely.geometry.MultiLineString(list(rivers)+[extra,])
    rivers_clean = workflow.hydrography.quick_cleanup(rivers_wextra)
    assert_close(rivers_clean, rivers_wextra, 0.000001)

    rivers_wextra_clean = workflow.tree.forests_to_list(workflow.hydrography.cleanup(two_boxes, rivers_wextra))
    rivers_clean = workflow.tree.forests_to_list(workflow.hydrography.cleanup(two_boxes, rivers))
    assert_close(rivers_clean, rivers_wextra_clean)

def test_cut_and_bin_one_contained_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)

    # one perfectly contained segment
    s1 = [shapely.geometry.LineString([(0.1,0.1), (4.3,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == s1[0])

def test_cut_and_bin_one_boundary_touching_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that touches the boundary
    s1 = [shapely.geometry.LineString([(0.,0.1), (4.3,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == s1[0])

def test_cut_and_bin_one_inter_touching_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that touches the inter
    s1 = [shapely.geometry.LineString([(0.1,0.1), (10.0,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == s1[0])

def test_cut_and_bin_one_leaving_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that leaves the domain
    s1 = [shapely.geometry.LineString([(-0.1,3.3), (9.0,3.3)]),]
    #with pytest.raises(RuntimeError):
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == shapely.geometry.LineString([(0,3.3), (9,3.3)]))

def test_cut_and_bin_one_spanning_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that spans the two boxes
    s1 = [shapely.geometry.LineString([(0.1,3.3), (11.0,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 1)
    assert(bins[0][0] == shapely.geometry.LineString([(0.1,3.3), (10.0,3.3)]))
    assert(bins[1][0] == shapely.geometry.LineString([(10.0,3.3), (11.0,3.3)]))

def test_cut_and_bin_one_spanning_existingpoint_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that spans the two boxes at a already-there point
    s1 = [shapely.geometry.LineString([(0.1,3.3), (10.0,3.3), (11.0,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 1)
    assert(bins[0][0] == shapely.geometry.LineString([(0.1,3.3), (10.0,3.3)]))
    assert(bins[1][0] == shapely.geometry.LineString([(10.0,3.3), (11.0,3.3)]))

def test_cut_and_bin_one_spanning_multipoint_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one multisegment that spans the two boxes
    s1 = [shapely.geometry.LineString([(0.1,3.3), (5.0,3.3), (7.0,3.3), (9.0,3.3), (11.0,3.3), (13.0,3.3), (15.0,3.3)])]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 1)
    assert(bins[0][0] == shapely.geometry.LineString([(0.1,3.3), (5.0,3.3), (7.0,3.3), (9.0,3.3), (10.0,3.3)]))
    assert(bins[1][0] == shapely.geometry.LineString([(10.0,3.3), (11.0,3.3), (13.0,3.3), (15.0,3.3)]))
    
def test_cut_and_bin_two_contained_segs(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # two perfectly contained segments
    s1 = [shapely.geometry.LineString([(0.1,0.1), (4.3,3.3)]),
          shapely.geometry.LineString([(11.1,0.1), (15.3,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 1)
    assert(bins[0][0] == s1[0])
    assert(bins[1][0] == s1[1])

    

def data(poly_hucs,river_segs):
    hucs = workflow.hucs.HUCs(poly_hucs)
    rivers = workflow.hydrography.make_global_tree(river_segs)
    for tree in rivers:
        assert(workflow.tree.is_consistent(tree))
    return hucs,rivers

def check1(hucs,rivers):
    assert(len(hucs) is 1)
    poly0 = hucs.polygon(0)
    print(poly0.boundary.coords[:])
    assert(workflow.utils.close(poly0, shapely.geometry.Polygon([(0,-5), (10,-5), (10,0), (10,5), (0,5)])))

    riverlist = list(rivers[0].dfs())
    assert(len(riverlist) is 1)
    print(riverlist[0].coords[:])
    assert(workflow.utils.close(riverlist[0], shapely.geometry.LineString([(5,0), (10,0)])))

    for tree in rivers:
        assert(workflow.tree.is_consistent(tree))
    
def check2(hucs,rivers):
    assert(len(hucs) is 2)
    poly0 = hucs.polygon(0)
    print(poly0.boundary.coords[:])
    assert(workflow.utils.close(poly0, shapely.geometry.Polygon([(0,-5), (10,-5), (10,0), (10,5), (0,5)])))
    poly1 = hucs.polygon(1)
    print(poly1.boundary.coords[:])
    assert(workflow.utils.close(poly1, shapely.geometry.Polygon([(10,-5), (20,-5), (20,5), (10,5), (10,0)])))

    riverlist = list(rivers[0].dfs()) # dfs, preOrdered
    assert(len(riverlist) is 2)
    print(riverlist[0].coords[:])
    assert(workflow.utils.close(riverlist[0], shapely.geometry.LineString([(10,0), (15,0)])))
    print(riverlist[1].coords[:])
    assert(workflow.utils.close(riverlist[1], shapely.geometry.LineString([(5,0), (10,0)])))

    for tree in rivers:
        assert(workflow.tree.is_consistent(tree))
    
def check3(hucs,rivers):
    assert(len(hucs) is 3)
    poly0 = hucs.polygon(0)
    assert(workflow.utils.close(poly0, shapely.geometry.Polygon([(0,-5),(10,-5),(10,5),(0,5)])))
    poly1 = hucs.polygon(1)
    assert(workflow.utils.close(poly1, shapely.geometry.Polygon([(10,-5),(20,-5),(20,5),(10,5)])))
    poly2 = hucs.polygon(2)
    assert(workflow.utils.close(poly2, shapely.geometry.Polygon([(0,5),(10,5),(20,5),(20,10),(10,10),(0,10)])))

    assert(len(rivers[0]) is 3)
    riverlist = list(rivers[0].dfs())
    print(riverlist[0].coords[:])
    assert(workflow.utils.close(riverlist[0], shapely.geometry.LineString([(10.,5.), (10.,10.)])))
    print(riverlist[1].coords[:])
    assert(workflow.utils.close(riverlist[1], shapely.geometry.LineString([(5.,0.), (10.,5.)])))
    print(riverlist[2].coords[:])
    assert(workflow.utils.close(riverlist[2], shapely.geometry.LineString([(15.,0.), (10.,5.)])))

    for tree in rivers:
        assert(workflow.tree.is_consistent(tree))
    
def test_snap0():
    # snap a river endpoint onto a huc
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5.,0.), (10.,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check1(hucs,rivers)

def test_snap0b():
    # snap a river endpoint onto a huc
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5.,0.), (9.999,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check1(hucs,rivers)

def test_snap0c():
    # snap a river endpoint onto a huc
    b1 = [(0, -5), (10,-5), (10,0.), (10,5), (0,5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5.,0.), (9.999,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check1(hucs,rivers)
    
def test_snap1():
    # snap a river endpoint onto a huc
    b1 = [(0, -5), (10,-5), (10, 0.001), (10,5), (0,5)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    rs = [shapely.geometry.LineString([(5.,0.), (9.99,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check1(hucs,rivers)

    
def test_snap2():
    # generic intersection with no movement only additions
    tb = two_boxes()
    rs = [shapely.geometry.LineString([(5.,0.), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check2(hucs,rivers)
        
def test_snap3():
    # move a river point onto the huc boundary
    tb = two_boxes()
    rs = [shapely.geometry.LineString([(5.,0.), (10.001,0), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    poly0 = hucs.polygon(0)
    check2(hucs,rivers)
    
def test_snap4():
    # move a huc point onto the river
    b1 = [(0, -5), (10,-5), (10, 0.001), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5), (10, 0.001)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    rs = [shapely.geometry.LineString([(5.,0.), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check2(hucs,rivers)
    
def test_snap5():
    # move both
    b1 = [(0, -5), (10,-5), (10, 0.001), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5), (10, 0.001)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    rs = [shapely.geometry.LineString([(5.,0.), (10.001,0), (15,0)]),]
    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check2(hucs,rivers)
    
def test_snap6():
    # three boxes, intersection at the junction, do nothing
    b1 = [(0, -5), (10,-5), (10,5), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10,5)]
    b3 = [(0, 5), (10,5), (20,5), (20,10), (0,10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))

    rs = [shapely.geometry.LineString([(5.,0.), (10.,5),]),
          shapely.geometry.LineString([(15.,0.), (10.,5),]),
          shapely.geometry.LineString([(10.,5.), (10,10)]),
    ]

    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check3(hucs,rivers)

def test_snap7():
    # three boxes, intersection at the junction, move the huc to the river
    b1 = [(0, -5), (10,-5), (10.001,5.001), (0,5)]
    b2 = [(10, -5), (20,-5), (20,5), (10.001,5.001)]
    b3 = [(0, 5), (10.001,5.001), (20,5), (20,10), (0,10)]
    tb = []
    tb.append(shapely.geometry.Polygon(b1))
    tb.append(shapely.geometry.Polygon(b2))
    tb.append(shapely.geometry.Polygon(b3))

    rs = [shapely.geometry.LineString([(5.,0.), (10.,5),]),
          shapely.geometry.LineString([(15.,0.), (10.,5),]),
          shapely.geometry.LineString([(10.,5.), (10,10)]),]

    hucs, rivers = data(tb, rs)
    workflow.hydrography.snap(hucs, rivers, 0.1)
    check3(hucs,rivers)
