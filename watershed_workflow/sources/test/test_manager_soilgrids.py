import pytest
import os
import numpy as np

import watershed_workflow.config
import watershed_workflow.sources.manager_soilgrids_2017 as manager_soilgrids
import watershed_workflow.crs


@pytest.fixture
def soilgrids():
    return manager_soilgrids.ManagerSoilGrids2017()


@pytest.fixture 
def soilgrids_us():
    return manager_soilgrids.ManagerSoilGrids2017('US')


def test_constructor(soilgrids):
    """Test basic constructor and properties."""
    assert soilgrids.name == 'SoilGrids2017'
    assert soilgrids.source == manager_soilgrids.ManagerSoilGrids2017.URL
    assert soilgrids.native_crs_in == watershed_workflow.crs.from_epsg(4326)
    assert soilgrids.native_crs_out == watershed_workflow.crs.from_epsg(4326)
    assert soilgrids.native_start is None
    assert soilgrids.native_end is None
    assert soilgrids.default_variables == ['BDTICM']


def test_constructor_us_variant(soilgrids_us):
    """Test US variant constructor."""
    assert soilgrids_us.name == 'SoilGrids2017_US'
    assert soilgrids_us.source == manager_soilgrids.ManagerSoilGrids2017.URL
    assert soilgrids_us.default_variables == ['BDTICM']


def test_valid_variables(soilgrids):
    """Test that all expected variables are present."""
    expected_vars = set(['BDTICM'])  # bedrock variable
    
    # Add all layer variables
    for base_var in manager_soilgrids.ManagerSoilGrids2017.BASE_VARIABLES:
        for layer in manager_soilgrids.ManagerSoilGrids2017.LAYERS:
            expected_vars.add(f'{base_var}_layer_{layer}')
    
    assert set(soilgrids.valid_variables) == expected_vars
    assert len(soilgrids.valid_variables) == 1 + 5 * 7  # BDTICM + 5 vars * 7 layers


def test_parse_variable(soilgrids):
    """Test variable name parsing."""
    # Test bedrock variable
    base_var, layer = soilgrids._parseVariable('BDTICM')
    assert base_var == 'BDTICM'
    assert layer is None
    
    # Test layered variables
    base_var, layer = soilgrids._parseVariable('BLDFIE_layer_3')
    assert base_var == 'BLDFIE'
    assert layer == 3
    
    # Test invalid variables
    with pytest.raises(ValueError):
        soilgrids._parseVariable('INVALID_VAR')
    
    with pytest.raises(ValueError):
        soilgrids._parseVariable('BLDFIE_layer_8')  # layer 8 doesn't exist
    
    with pytest.raises(ValueError):
        soilgrids._parseVariable('BLDFIE_layer_abc')  # invalid layer number


def test_variable_categories(soilgrids):
    """Test that variables are properly categorized."""
    # Check base variables
    expected_base_vars = ['BLDFIE', 'CLYPPT', 'SLTPPT', 'SNDPPT', 'WWP']
    assert soilgrids.BASE_VARIABLES == expected_base_vars
    
    # Check bedrock variable
    assert soilgrids.BEDROCK_VARIABLE == 'BDTICM'
    
    # Check layers
    assert soilgrids.LAYERS == list(range(1, 8))


