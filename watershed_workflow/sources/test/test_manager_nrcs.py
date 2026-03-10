import pytest

import watershed_workflow.sources.manager_nrcs

@pytest.mark.network
def test_nrcs2(coweeta):
    # get imgs
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS(force_download=True)
    df = nrcs.getShapesByGeometry(coweeta.geometry[0], coweeta.crs)

    # check df
    mukeys = set(df['ID'])
    assert len(df) == len(mukeys) # one per unique key
    assert 50 > len(df) > 30
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


@pytest.mark.network
def test_nrcs_geodataframe_input(coweeta):
    """Test getShapesByGeometry with GeoDataFrame input"""
    import geopandas as gpd
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    
    # Create GeoDataFrame from coweeta fixture
    gdf = gpd.GeoDataFrame([{'test': 1}], geometry=[coweeta.geometry[0]], crs=coweeta.crs)
    
    df = nrcs.getShapesByGeometry(gdf)
    
    assert isinstance(df, gpd.GeoDataFrame)
    assert 50 > len(df) > 30
    import watershed_workflow.sources.standard_names as names
    assert names.ID in df.columns
    assert names.NAME in df.columns


@pytest.mark.network
def test_nrcs_properties(coweeta):
    """Test that soil properties are present and not truncated after cache roundtrip."""
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    df = nrcs.getShapesByGeometry(coweeta.geometry[0], coweeta.crs)

    expected_properties = [
        'residual saturation [-]',
        'van Genuchten alpha [Pa^-1]',
        'van Genuchten n [-]',
        'permeability [m^2]',
        'porosity [-]',
        'bulk density [g/cm^3]',
        'total sand pct [%]',
        'total silt pct [%]',
        'total clay pct [%]',
        'thickness [cm]',
    ]
    for col in expected_properties:
        assert col in df.columns, f'Missing property column: {col}'
        assert df[col].notna().any(), f'Property column {col} is all NaN'


def test_nrcs_getShapesByID_not_supported():
    """Test that getShapesByID raises NotImplementedError"""
    import pytest
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    
    with pytest.raises(NotImplementedError, match="ManagerNRCS doesn't support getShapesByID"):
        nrcs.getShapesByID(['123456'])
