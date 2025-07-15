from watershed_workflow.sources.manager_nhd import ManagerNHD

from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs

def test_nhd_get(coweeta):
    # download
    nhd = ManagerNHD('NHDPlus MR v2.1')
    reaches = nhd.getShapesByGeometry(coweeta.geometry[0], coweeta.crs).to_crs(watershed_workflow.crs.latlon_crs)
    assert 7 == len(reaches)  # note this is different from NHDPlus

def test_nhd_mr_get(coweeta):
    # download
    nhd = ManagerNHD('NHD MR')
    reaches = nhd.getShapesByGeometry(coweeta.geometry[0], coweeta.crs).to_crs(watershed_workflow.crs.latlon_crs)
    assert 7 == len(reaches)  # note this is different from NHDPlus

def test_nhd_hr_get(coweeta):
    # download
    nhd = ManagerNHD('NHDPlus HR')
    reaches = nhd.getShapesByGeometry(coweeta.geometry[0], coweeta.crs).to_crs(watershed_workflow.crs.latlon_crs)
    assert 21 == len(reaches)  # note this is different from NHDPlus
    
