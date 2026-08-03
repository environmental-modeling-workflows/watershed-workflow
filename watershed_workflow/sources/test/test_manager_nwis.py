"""Tests for ManagerNWIS — USGS National Water Information System.

Network tests use the western NC mountain region that contains multiple
stream gages with daily values, including site 03500000 (Little Tennessee
River near Prentiss, NC) which is the closest active gage to the
Coweeta Hydrological Laboratory area.
"""

import pytest
import shapely.geometry
import pandas as pd
import geopandas as gpd

import watershed_workflow.crs
import watershed_workflow.sources.manager_nwis
import watershed_workflow.sources.standard_names as names


# ---------------------------------------------------------------------------
# Non-network tests
# ---------------------------------------------------------------------------

def test_constructor():
    """Verify default constructor attributes."""
    mgr = watershed_workflow.sources.manager_nwis.ManagerNWIS()
    assert mgr.product == 'USGS NWIS'
    assert mgr.native_id_field == 'site_no'
    assert mgr.site_type == 'ST'
    assert mgr.has_data_type == 'dv'
    assert mgr.force_download == False


def test_constructor_custom():
    """Verify that custom constructor arguments are stored."""
    mgr = watershed_workflow.sources.manager_nwis.ManagerNWIS(site_type='GW')
    assert mgr.site_type == 'GW'


def test_getShapes_not_supported():
    """Verify that getShapes raises NotImplementedError."""
    mgr = watershed_workflow.sources.manager_nwis.ManagerNWIS()
    with pytest.raises(NotImplementedError):
        mgr.getShapes()


# ---------------------------------------------------------------------------
# Network tests (require live NWIS web service)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_getShapesByGeometry():
    """Retrieve NWIS stream gages in the western NC mountain region.

    Uses a bounding box spanning Macon and Swain counties, NC — the
    region surrounding the Coweeta Hydrological Laboratory — which
    contains several active USGS daily-value stream gages.
    """
    mgr = watershed_workflow.sources.manager_nwis.ManagerNWIS(force_download=True)

    # Box: western NC mountains, Macon–Swain counties
    box = shapely.geometry.box(-83.7, 35.0, -83.3, 35.5)
    crs = watershed_workflow.crs.from_epsg(4326)
    gages = mgr.getShapesByGeometry(box, crs)

    assert isinstance(gages, gpd.GeoDataFrame)
    assert len(gages) > 0
    assert names.ID in gages.columns
    assert names.NAME in gages.columns
    assert all(geom.geom_type == 'Point' for geom in gages.geometry)


@pytest.mark.network
def test_getShapesByID():
    """Look up a known western NC stream gage by site number.

    Site 03500000 is 'Little Tennessee River Near Prentiss, NC',
    an active USGS daily-value gage in the Coweeta region.
    """
    mgr = watershed_workflow.sources.manager_nwis.ManagerNWIS()
    gages = mgr.getShapesByID(['03500000'])

    assert isinstance(gages, gpd.GeoDataFrame)
    assert len(gages) == 1
    assert gages[names.ID].iloc[0] == '03500000'


@pytest.mark.network
def test_getStreamflow():
    """Retrieve daily streamflow for a known gage for January 2020."""
    mgr = watershed_workflow.sources.manager_nwis.ManagerNWIS()
    gages = mgr.getShapesByID(['03500000'])
    assert len(gages) == 1

    flow = mgr.getStreamflow(gages, ('2020-01-01', '2020-01-31'))

    assert isinstance(flow, pd.DataFrame)
    assert len(flow) == 31
    assert all(col.startswith('USGS-') for col in flow.columns)
