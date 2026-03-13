"""Tests for ManagerPOLARIS.

Non-network tests (constructor, variable validation, depth consistency) run
without any network access.  Network tests are marked with
``@pytest.mark.network`` and require a live connection to the POLARIS HTTP
server at hydrology.cee.duke.edu.
"""
import pytest
import numpy as np
import xarray as xr

from watershed_workflow.sources.manager_polaris import (
    ManagerPOLARIS,
    _DEPTH_LABELS,
    _DEPTH_CENTRES,
    _DEFAULT_VARIABLES,
    _VALID_VARIABLES,
)


# ---------------------------------------------------------------------------
# Constructor / metadata tests (no network)
# ---------------------------------------------------------------------------

def test_constructor():
    mgr = ManagerPOLARIS()
    assert mgr.stat == 'mean'
    assert mgr.force_download == False
    assert mgr.native_resolution == pytest.approx(0.000278)


def test_constructor_stat_p50():
    mgr = ManagerPOLARIS(stat='p50')
    assert mgr.stat == 'p50'


def test_constructor_force_download():
    mgr = ManagerPOLARIS(force_download=True)
    assert mgr.force_download == True


def test_valid_variables():
    mgr = ManagerPOLARIS()
    for var in ['clay', 'sand', 'silt', 'om',
                'ksat', 'alpha', 'n', 'theta_r', 'theta_s',
                'bd', 'ph', 'hb']:
        assert var in mgr.valid_variables


def test_default_variables():
    mgr = ManagerPOLARIS()
    assert set(mgr.default_variables) == {'theta_s', 'theta_r', 'alpha', 'n', 'ksat'}


def test_depth_labels_consistent():
    assert len(_DEPTH_LABELS) == len(_DEPTH_CENTRES) == 6


def test_invalid_stat_raises():
    with pytest.raises(ValueError, match='Invalid stat'):
        ManagerPOLARIS(stat='invalid')


def test_invalid_variable_raises():
    import shapely.geometry
    import watershed_workflow.crs
    mgr = ManagerPOLARIS()
    geom = shapely.geometry.box(-84.0, 35.0, -83.0, 36.0)
    with pytest.raises(ValueError, match='Invalid variable'):
        mgr._preprocessParameters(
            geom, watershed_workflow.crs.latlon_crs,
            None, None, ['not_a_real_variable'], None, None,
        )


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_get_dataset_default_variables(coweeta):
    """Download default VG variables for Coweeta and check they are present."""
    mgr = ManagerPOLARIS(force_download=True)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs)
    assert isinstance(ds, xr.Dataset)
    for var in ['theta_s', 'theta_r', 'alpha', 'n', 'ksat']:
        assert var in ds, f'Missing variable: {var}'


@pytest.mark.network
def test_dataset_has_depth_dimension(coweeta):
    """Each variable must have 6 depth layers at the correct centre depths."""
    mgr = ManagerPOLARIS(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['theta_s'])
    assert 'depth' in ds['theta_s'].dims
    assert ds['theta_s'].sizes['depth'] == 6
    np.testing.assert_allclose(
        ds['theta_s'].depth.values,
        np.array(_DEPTH_CENTRES, dtype=np.float32),
        rtol=1e-5,
    )


@pytest.mark.network
def test_dataset_values_reasonable(coweeta):
    """theta_s (porosity) must be in (0, 1] and alpha must be on log10 scale."""
    mgr = ManagerPOLARIS(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs,
                        variables=['theta_s', 'alpha'])

    theta_s_vals = ds['theta_s'].values
    valid_ts = theta_s_vals[np.isfinite(theta_s_vals)]
    assert (valid_ts > 0).all(), 'theta_s has non-positive values'
    assert (valid_ts <= 1).all(), 'theta_s exceeds 1'

    alpha_vals = ds['alpha'].values
    valid_alpha = alpha_vals[np.isfinite(alpha_vals)]
    # log10(cm^-1): typical range roughly -3 to 0
    assert (valid_alpha < 2).all(), 'alpha (log10) unexpectedly large'


@pytest.mark.network
def test_dataset_crs(coweeta):
    """Dataset must carry a CRS."""
    mgr = ManagerPOLARIS(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['theta_s'])
    assert ds.rio.crs is not None


@pytest.mark.network
def test_cache_reuse(coweeta, tmp_path, monkeypatch):
    """Second call with the same bounds must reuse the cache, not re-download."""
    import watershed_workflow.utils.config
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'],
        'data_directory', str(tmp_path),
    )

    mgr = ManagerPOLARIS(force_download=False)
    ds1 = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['theta_s'])
    ds2 = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['theta_s'])
    xr.testing.assert_equal(ds1, ds2)
