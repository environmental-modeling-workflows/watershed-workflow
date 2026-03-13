"""Tests for ManagerHFHydrodata.

Non-network tests (constructor, indicator table) run without credentials.
Network tests are marked with ``@pytest.mark.network`` and require valid
credentials in ``~/.watershed_workflowrc``.
"""
import pytest
import pandas as pd
import xarray as xr

from watershed_workflow.sources.manager_hf_hydrodata import ManagerHFHydrodata


def test_constructor():
    mgr = ManagerHFHydrodata()
    assert mgr.force_download == False
    assert 'pf_indicator' in mgr.valid_variables
    assert 'pf_flowbarrier' in mgr.valid_variables
    assert mgr.default_variables == ['pf_indicator']


def test_constructor_force_download():
    mgr = ManagerHFHydrodata(force_download=True)
    assert mgr.force_download == True


def test_native_resolution():
    mgr = ManagerHFHydrodata()
    assert mgr.native_resolution == pytest.approx(0.009)


def test_native_crs_out():
    mgr = ManagerHFHydrodata()
    proj4 = mgr.native_crs_out.to_proj4()
    assert '+proj=lcc' in proj4
    assert '+lat_1=30' in proj4
    assert '+lat_2=60' in proj4
    assert '+units=m' in proj4


def test_indicator_table():
    mgr = ManagerHFHydrodata()
    table = mgr.getIndicatorTable()
    assert isinstance(table, pd.DataFrame)
    assert 'porosity' in table.columns
    assert 'permeability_x' in table.columns
    assert 'permeability_y' in table.columns
    assert 'permeability_z' in table.columns
    assert 'vg_alpha' in table.columns
    assert 'vg_n' in table.columns
    assert 'sres' in table.columns
    assert 'specific_storage' in table.columns
    assert len(table) > 0
    assert table.index.name == 'indicator'


def test_indicator_table_values_reasonable():
    mgr = ManagerHFHydrodata()
    table = mgr.getIndicatorTable()
    # Porosity should be between 0 and 1
    assert (table['porosity'] > 0).all()
    assert (table['porosity'] < 1).all()
    # Permeability should be positive
    assert (table['permeability_x'] > 0).all()
    assert (table['permeability_z'] > 0).all()
    # van Genuchten n should be > 1 for most materials
    assert (table['vg_n'] > 1).all()
    # Residual saturation should be between 0 and 1
    assert (table['sres'] >= 0).all()
    assert (table['sres'] < 1).all()


def test_indicator_table_idempotent():
    """getIndicatorTable should return the same result on repeated calls."""
    mgr = ManagerHFHydrodata()
    t1 = mgr.getIndicatorTable()
    t2 = mgr.getIndicatorTable()
    pd.testing.assert_frame_equal(t1, t2)


def test_valid_variables_list():
    mgr = ManagerHFHydrodata()
    for var in ['pf_indicator', 'pf_flowbarrier', 'porosity',
                'permeability_x', 'permeability_y', 'permeability_z',
                'vg_alpha', 'vg_n', 'specific_storage']:
        assert var in mgr.valid_variables


def test_invalid_variable_raises():
    """Variable validation happens in _preprocessParameters, before credential check."""
    import watershed_workflow.crs
    import shapely.geometry
    mgr = ManagerHFHydrodata()
    geom = shapely.geometry.box(-84.0, 35.0, -83.0, 36.0)
    with pytest.raises(ValueError, match="Invalid variable"):
        mgr._preprocessParameters(
            geom, watershed_workflow.crs.latlon_crs,
            None, None, ['not_a_real_variable'], None, None,
        )


@pytest.mark.network
def test_get_dataset_indicator(coweeta):
    """Download pf_indicator and pf_flowbarrier for Coweeta watershed."""
    mgr = ManagerHFHydrodata(force_download=True)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs,
                        variables=['pf_indicator', 'pf_flowbarrier'])
    assert isinstance(ds, xr.Dataset)
    assert 'pf_indicator' in ds
    assert 'pf_flowbarrier' in ds
    # both fields are 3D: z, y, x (as returned by the conus2_domain API)
    assert ds['pf_indicator'].ndim == 3
    assert ds['pf_flowbarrier'].ndim == 3
    assert 'x' in ds['pf_indicator'].dims
    assert 'y' in ds['pf_indicator'].dims
    # Check CRS was written
    assert ds.rio.crs is not None


@pytest.mark.network
def test_get_dataset_porosity(coweeta):
    """Download porosity for Coweeta watershed and verify shape/values."""
    mgr = ManagerHFHydrodata(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs,
                        variables=['porosity'])
    assert 'porosity' in ds
    # Porosity should be between 0 and 1
    vals = ds['porosity'].values
    assert (vals[vals > 0] < 1).all()


@pytest.mark.network
def test_cache_reuse(coweeta, tmp_path, monkeypatch):
    """Second call with same bounds should reuse the cache, not re-download."""
    import watershed_workflow.utils.config
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'],
        'data_directory', str(tmp_path))

    mgr = ManagerHFHydrodata(force_download=False)
    # First call: downloads
    ds1 = mgr.getDataset(coweeta.geometry[0], coweeta.crs,
                         variables=['pf_flowbarrier'])
    # Second call: should hit cache
    ds2 = mgr.getDataset(coweeta.geometry[0], coweeta.crs,
                         variables=['pf_flowbarrier'])
    xr.testing.assert_equal(ds1, ds2)
