import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.sources.manager_glhymps
import watershed_workflow.sources.standard_names as names

from fixtures import coweeta


def test_glhymps_coweeta(coweeta):
    """Test GLHYMPS manager using Coweeta example data."""
    # Use the local Coweeta GLHYMPS file
    glhymps_file = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Coweeta', 'input_data', 'soil_structure', 'GLHYMPS', 'GLHYMPS.shp')
    glhymps_file = os.path.abspath(glhymps_file)
    
    # Skip test if file doesn't exist
    if not os.path.exists(glhymps_file):
        pytest.skip(f"GLHYMPS test file not found: {glhymps_file}")
    
    # Create manager with local file
    glhymps = watershed_workflow.sources.manager_glhymps.ManagerGLHYMPS(glhymps_file)
    
    # Test getShapesByGeometry with coweeta GeoDataFrame
    data = glhymps.getShapesByGeometry(coweeta)
    
    # Check basic properties
    assert len(data) > 0
    assert data.crs is not None
    assert names.ID in data.columns
    assert names.NAME in data.columns
    
    # Test that we get raw GLHYMPS data (OBJECTID_1 mapped to ID)
    assert 'OBJECTID_1' in data.columns  # Original GLHYMPS field should still be present
    
    # Test getShapesByID using the IDs we just retrieved
    test_ids = data[names.ID].iloc[:2].astype(str).tolist()  # Get first 2 IDs as strings
    data_by_id = glhymps.getShapesByID(test_ids)
    
    assert len(data_by_id) == len(test_ids)
    assert names.ID in data_by_id.columns
