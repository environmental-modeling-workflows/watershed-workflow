import pytest
import shapely.geometry

from workflow.test.shapes import *

import workflow.utils
import workflow.hucs
import workflow.hydrography

def test_bin(two_ys, two_boxes):
    """Tests simple binnning with seperate graphs"""
    bins = workflow.hydrography.bin_rivers(two_boxes, two_ys)
    assert(len(bins[0][0]) == 2)
    assert(len(bins[0][1]) == 1)
    assert(len(bins[1][0]) == 3)
    assert(len(bins[1][1]) == 0)


def test_bin_rivers(rivers, two_boxes):
    """Tests simple binnning with nice river network"""
    bins = workflow.hydrography.bin_rivers(two_boxes, rivers)
    assert(len(bins[0][0]) == 1)
    assert(len(bins[0][1]) == 2)
    assert(len(bins[1][0]) == 2)
    assert(len(bins[1][1]) == 1)

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


def test_cut_and_bin(two_boxes):
    seg1 = shapely.geometry.LineString([(1,0), (15,0)])
    ml = shapely.geometry.MultiLineString([seg1,])
    bins = workflow.hydrography.split_and_bin(two_boxes, ml)

    sub1 = shapely.geometry.LineString([(1,0), (10,0)])
    sub2 = shapely.geometry.LineString([(10,0), (15,0)])
    assert(len(bins) is 2)
    assert(len(bins[0]) is 1)
    assert(len(bins[1]) is 1)
    assert_close(bins[0][0], sub1)
    assert_close(bins[1][0], sub2)

def test_pruning_cleanup(two_boxes, rivers):
    """Tests that cleanup can remove close points"""
    extra = shapely.geometry.LineString([(12,-3.01), (12,-3)])
    rivers_wextra = shapely.geometry.MultiLineString(list(rivers)+[extra,])
    rivers_clean = workflow.hydrography.quick_cleanup(rivers_wextra)
    assert_close(rivers_clean, rivers_wextra, 0.000001)

    rivers_wextra_clean = workflow.tree.forests_to_list(workflow.hydrography.cleanup(two_boxes, rivers_wextra))
    rivers_clean = workflow.tree.forests_to_list(workflow.hydrography.cleanup(two_boxes, rivers))
    assert_close(rivers_clean, rivers_wextra_clean)



def test_cut_and_bin2_one_contained_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)

    # one perfectly contained segment
    s1 = [shapely.geometry.LineString([(0.1,0.1), (4.3,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == s1[0])

def test_cut_and_bin2_one_boundary_touching_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that touches the boundary
    s1 = [shapely.geometry.LineString([(0.,0.1), (4.3,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == s1[0])

def test_cut_and_bin2_one_inter_touching_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that touches the inter
    s1 = [shapely.geometry.LineString([(0.1,0.1), (10.0,3.3)]),]
    bins = workflow.hydrography.cut_and_bin(tb, s1)
    assert(len(bins) == 2)
    assert(len(bins[0]) == 1)
    assert(len(bins[1]) == 0)
    assert(bins[0][0] == s1[0])

def test_cut_and_bin2_one_leaving_seg(two_boxes):
    """Test of binning"""
    tb = workflow.hucs.HUCs(two_boxes)
    
    # one segment that leaves the domain
    s1 = [shapely.geometry.LineString([(-0.1,3.3), (9.0,3.3)]),]
    with pytest.raises(RuntimeError):
        bins = workflow.hydrography.cut_and_bin(tb, s1)
    # assert(len(bins) == 2)
    # assert(len(bins[0]) == 1)
    # assert(len(bins[1]) == 0)
    # assert(bins[0][0] == shapely.geometry.LineString([(0,3.3), (9,3.3)]))

def test_cut_and_bin2_one_spanning_seg(two_boxes):
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

def test_cut_and_bin2_one_spanning_existingpoint_seg(two_boxes):
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

def test_cut_and_bin2_one_spanning_multipoint_seg(two_boxes):
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
    
def test_cut_and_bin2_two_contained_segs(two_boxes):
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

    

    


