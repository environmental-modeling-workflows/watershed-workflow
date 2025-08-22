import pytest

import watershed_workflow.sources.manager_nrcs

from fixtures import coweeta


def test_nrcs2(coweeta):
    # get imgs
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS(force_download=True)
    df = nrcs.getShapesByGeometry(coweeta.geometry[0], coweeta.crs)

    # check df
    mukeys = set(df['ID'])
    assert len(df) == len(mukeys) # one per unique key
    assert len(df) == 38  # Updated expected count after refactoring
    assert df.crs is not None
    
    # Test that standard names are applied
    import watershed_workflow.sources.standard_names as names
    assert names.ID in df.columns
    assert names.NAME in df.columns
    
    # Check that all names follow NRCS-{mukey} pattern
    for name in df[names.NAME]:
        assert name.startswith('NRCS-')
    
    # Check metadata
    assert df.attrs['name'] == nrcs.name
    assert df.attrs['source'] == nrcs.source


def test_nrcs_constructor():
    """Test NRCS constructor properties"""
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    assert nrcs.name == 'National Resources Conservation Service Soil Survey (NRCS Soils)'
    assert nrcs.source == 'USDA NRCS SSURGO Database'
    assert nrcs.native_id_field == 'mukey'
    assert nrcs.force_download == False
    
    # Test with force_download=True
    nrcs_force = watershed_workflow.sources.manager_nrcs.ManagerNRCS(force_download=True)
    assert nrcs_force.force_download == True


def test_nrcs_geodataframe_input(coweeta):
    """Test getShapesByGeometry with GeoDataFrame input"""
    import geopandas as gpd
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    
    # Create GeoDataFrame from coweeta fixture
    gdf = gpd.GeoDataFrame([{'test': 1}], geometry=[coweeta.geometry[0]], crs=coweeta.crs)
    
    df = nrcs.getShapesByGeometry(gdf)
    
    assert isinstance(df, gpd.GeoDataFrame)
    assert len(df) == 38  # Same expected count
    import watershed_workflow.sources.standard_names as names
    assert names.ID in df.columns
    assert names.NAME in df.columns


def test_nrcs_getShapesByID_not_supported():
    """Test that getShapesByID raises NotImplementedError"""
    import pytest
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    
    with pytest.raises(NotImplementedError, match="ManagerNRCS doesn't support getShapesByID"):
        nrcs.getShapesByID(['123456'])
