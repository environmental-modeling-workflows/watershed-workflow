import pytest
import geopandas as gpd

from watershed_workflow.sources.manager_nhd import ManagerNHD
from watershed_workflow.sources.test.fixtures import coweeta
import watershed_workflow.crs
import watershed_workflow.sources.standard_names as names


@pytest.fixture
def nhd_waterdata():
    """ManagerNHD for WaterData protocol"""
    return ManagerNHD('NHDPlus MR v2.1', catchments=True)


@pytest.fixture
def nhd_mr():
    """ManagerNHD for NHD MR protocol"""
    return ManagerNHD('NHD MR', catchments=True)


@pytest.fixture  
def nhd_hr():
    """ManagerNHD for NHDPlus HR protocol"""
    return ManagerNHD('NHDPlus HR', catchments=True)


def test_constructor_properties():
    """Test that constructor sets properties correctly"""
    nhd = ManagerNHD('NHDPlus MR v2.1')
    assert nhd.name == 'NHDPlus MR v2.1'  # Name is overridden to be user-friendly
    assert nhd.source == 'HyRiver.WaterData'
    assert nhd.native_crs_in == watershed_workflow.crs.latlon_crs
    assert nhd._protocol_name == 'WaterData'  # Protocol name is the string
    assert nhd._protocol.__name__ == 'WaterData'  # _protocol is the class
    assert nhd._catchments == True  # default


def test_constructor_no_catchments():
    """Test constructor with catchments disabled"""
    nhd = ManagerNHD('NHDPlus MR v2.1', catchments=False)
    assert nhd._catchments == False


def test_invalid_protocol():
    """Test that invalid protocol raises error"""
    with pytest.raises(ValueError, match="Invalid ManagerNHD dataset_name"):
        ManagerNHD('Invalid Protocol')


def test_nhd_waterdata_get_by_geometry(nhd_waterdata, coweeta):
    """Test WaterData protocol getShapesByGeometry"""
    reaches = nhd_waterdata.getShapesByGeometry(coweeta)
    
    # Basic checks
    assert isinstance(reaches, gpd.GeoDataFrame)
    assert len(reaches) == 7  # Expected count for Coweeta
    
    # Check standard naming
    assert names.ID in reaches.columns
    assert names.NAME in reaches.columns
    
    # Check that some NHD-specific columns are present
    expected_cols = [names.LENGTH, names.CATCHMENT_AREA, names.ORDER, names.DRAINAGE_AREA]
    for col in expected_cols:
        if col in reaches.columns:  # Some may not be present depending on data
            assert reaches[col].notna().any()


def test_nhd_mr_get_by_geometry(nhd_mr, coweeta):
    """Test NHD MR protocol getShapesByGeometry"""
    reaches = nhd_mr.getShapesByGeometry(coweeta)
    
    assert isinstance(reaches, gpd.GeoDataFrame)
    assert len(reaches) == 7  # Expected count for Coweeta
    assert names.ID in reaches.columns
    assert names.NAME in reaches.columns


def test_nhd_hr_get_by_geometry(nhd_hr, coweeta):
    """Test NHDPlus HR protocol getShapesByGeometry"""
    reaches = nhd_hr.getShapesByGeometry(coweeta)
    
    assert isinstance(reaches, gpd.GeoDataFrame)
    assert len(reaches) == 21  # Expected count for Coweeta
    assert names.ID in reaches.columns
    assert names.NAME in reaches.columns


def test_getShapesByGeometry_with_shape(nhd_waterdata, coweeta):
    """Test getShapesByGeometry with GeoDataFrame input"""
    
    reaches = nhd_waterdata.getShapesByGeometry(coweeta.geometry[0], coweeta.crs)
    
    assert isinstance(reaches, gpd.GeoDataFrame)
    assert len(reaches) == 7
    assert names.ID in reaches.columns
    assert names.NAME in reaches.columns


def test_getShapesByID(nhd_waterdata, coweeta):
    """Test getShapesByID functionality"""
    # First get some reaches to get their IDs
    reaches = nhd_waterdata.getShapesByGeometry(coweeta)
    
    # Get first reach by ID
    first_id = reaches[names.ID].iloc[0]
    single_reach = nhd_waterdata.getShapesByID([first_id])
    
    assert isinstance(single_reach, gpd.GeoDataFrame)
    assert len(single_reach) == 1
    assert single_reach[names.ID].iloc[0] == first_id
    assert names.NAME in single_reach.columns
    
    # Test multiple IDs
    first_two_ids = reaches[names.ID].iloc[:2].tolist()
    two_reaches = nhd_waterdata.getShapesByID(first_two_ids)
    assert len(two_reaches) == 2


def test_getShapesByID_string_input(nhd_waterdata, coweeta):
    """Test getShapesByID with single string input"""
    reaches = nhd_waterdata.getShapesByGeometry(coweeta)
    first_id = reaches[names.ID].iloc[0]
    
    # Test with single string (not list)
    single_reach = nhd_waterdata.getShapesByID(first_id)
    assert len(single_reach) == 1
    assert single_reach[names.ID].iloc[0] == first_id


def test_catchments_functionality(coweeta):
    """Test that catchments are properly fetched when enabled"""
    # Test with catchments enabled
    nhd_with_catchments = ManagerNHD('NHDPlus MR v2.1', catchments=True)
    reaches_with_catchments = nhd_with_catchments.getShapesByGeometry(coweeta)
    
    # Test without catchments
    nhd_no_catchments = ManagerNHD('NHDPlus MR v2.1', catchments=False)
    reaches_no_catchments = nhd_no_catchments.getShapesByGeometry(coweeta)
    
    # With catchments should have more columns (catchment-related columns with _ca suffix)
    assert 'catchment' in reaches_with_catchments, "Should have catchment columns when catchments=True"
    
    # Without catchments should have fewer columns
    assert 'catchment' not in reaches_no_catchments, "Should not have catchment columns when catchments=False"


def test_standard_naming_applied(nhd_waterdata, coweeta):
    """Test that standard naming is properly applied"""
    reaches = nhd_waterdata.getShapesByGeometry(coweeta.geometry[0], coweeta.crs)
    
    # Check required standard columns
    assert names.ID in reaches.columns
    assert names.NAME in reaches.columns
    
    # Check that ID values are strings (as specified in _addStandardNames)
    assert reaches[names.ID].dtype == 'string'
    
    # Check metadata
    assert reaches.attrs['name'] == nhd_waterdata.name
    assert reaches.attrs['source'] == nhd_waterdata.source


def test_crs_handling(nhd_waterdata, coweeta):
    """Test that CRS is properly handled"""
    coweeta_utm = coweeta.to_crs(watershed_workflow.crs.default_crs)
    
    # Test that manager has correct native CRS
    assert nhd_waterdata.native_crs_in == watershed_workflow.crs.latlon_crs
    assert nhd_waterdata.native_crs_in != coweeta_utm.crs
    
    # Test with different input CRS
    reaches = nhd_waterdata.getShapesByGeometry(coweeta_utm)
    
    # Should have proper CRS set in output
    assert reaches.crs is not None
    assert len(reaches) == 7  # Should still get same results regardless of input CRS

    
