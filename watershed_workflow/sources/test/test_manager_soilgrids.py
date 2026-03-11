"""Tests for ManagerSoilGrids (SoilGrids v2.0).

Non-network tests (constructor, variable validation) run without any network
access.  Network tests are marked with ``@pytest.mark.network`` and require a
live connection to the ISRIC WCS.
"""
import pytest
import numpy as np
import xarray as xr

from watershed_workflow.sources.manager_soilgrids import (
    ManagerSoilGrids, _DEPTH_LABELS, _DEPTH_CENTRES,
)


# ---------------------------------------------------------------------------
# Constructor / metadata tests
# ---------------------------------------------------------------------------

def test_constructor():
    mgr = ManagerSoilGrids()
    assert mgr.force_download == False
    assert mgr.native_resolution == pytest.approx(0.002)


def test_constructor_force_download():
    mgr = ManagerSoilGrids(force_download=True)
    assert mgr.force_download == True


def test_valid_variables():
    mgr = ManagerSoilGrids()
    for var in ['clay', 'sand', 'silt', 'bdod', 'soc', 'phh2o', 'nitrogen', 'cfvo', 'ocd']:
        assert var in mgr.valid_variables


def test_default_variables():
    mgr = ManagerSoilGrids()
    assert set(mgr.default_variables) == {'clay', 'sand', 'silt', 'bdod'}


def test_depth_labels_and_centres_consistent():
    assert len(_DEPTH_LABELS) == len(_DEPTH_CENTRES) == 6


def test_rosetta_raster_unit():
    """computeVanGenuchtenModelFromRasters returns correct variables and shapes."""
    import xarray as xr
    import watershed_workflow.properties.soil as sp

    depth = np.array([0.025, 0.10], dtype=np.float32)
    y     = np.array([35.1, 35.2], dtype=np.float64)
    x     = np.array([-84.0, -83.9], dtype=np.float64)

    def _make(val):
        return xr.DataArray(
            np.full((2, 2, 2), val, dtype=np.float32),
            dims=['depth', 'y', 'x'],
            coords={'depth': depth, 'y': y, 'x': x},
        )

    ds = xr.Dataset({
        'sand': _make(40.0),
        'silt': _make(35.0),
        'clay': _make(25.0),
        'bdod': _make(1.3),
    })
    ds_out = sp.computeVanGenuchtenModelFromRasters(ds)

    expected_new = {
        'residual saturation [-]', 'porosity [-]',
        'van Genuchten alpha [Pa^-1]', 'van Genuchten n [-]', 'permeability [m^2]',
    }
    assert expected_new.issubset(set(ds_out.data_vars))
    for v in expected_new:
        assert ds_out[v].shape == (2, 2, 2), f'{v} wrong shape'
        assert np.all(np.isfinite(ds_out[v].values)), f'{v} has non-finite values'

    # Sanity-check physical ranges
    assert (ds_out['porosity [-]'].values > 0).all()
    assert (ds_out['porosity [-]'].values < 1).all()
    assert (ds_out['residual saturation [-]'].values >= 0).all()
    assert (ds_out['van Genuchten n [-]'].values > 1).all()
    assert (ds_out['permeability [m^2]'].values > 0).all()


def test_rosetta_raster_nan_propagation():
    """NaN pixels must propagate through Rosetta without crashing."""
    import xarray as xr
    import watershed_workflow.properties.soil as sp

    depth = np.array([0.025], dtype=np.float32)
    y     = np.array([35.0, 35.1], dtype=np.float64)
    x     = np.array([-84.0], dtype=np.float64)

    sand_data = np.array([[[40.0]], [[np.nan]]], dtype=np.float32)
    silt_data = np.array([[[35.0]], [[np.nan]]], dtype=np.float32)
    clay_data = np.array([[[25.0]], [[np.nan]]], dtype=np.float32)
    bdod_data = np.array([[[1.3]],  [[np.nan]]], dtype=np.float32)

    ds = xr.Dataset({
        'sand': xr.DataArray(sand_data, dims=['y', 'depth', 'x'],
                             coords={'depth': depth, 'y': y, 'x': x}).transpose('depth','y','x'),
        'silt': xr.DataArray(silt_data, dims=['y', 'depth', 'x'],
                             coords={'depth': depth, 'y': y, 'x': x}).transpose('depth','y','x'),
        'clay': xr.DataArray(clay_data, dims=['y', 'depth', 'x'],
                             coords={'depth': depth, 'y': y, 'x': x}).transpose('depth','y','x'),
        'bdod': xr.DataArray(bdod_data, dims=['y', 'depth', 'x'],
                             coords={'depth': depth, 'y': y, 'x': x}).transpose('depth','y','x'),
    })
    ds_out = sp.computeVanGenuchtenModelFromRasters(ds)
    poro = ds_out['porosity [-]'].values  # shape (depth=1, y=2, x=1)
    assert np.isfinite(poro[0, 0, 0])
    assert np.isnan(poro[0, 1, 0])


def test_invalid_variable_raises():
    import shapely.geometry
    import watershed_workflow.crs
    mgr = ManagerSoilGrids()
    geom = shapely.geometry.box(-84.0, 35.0, -83.0, 36.0)
    with pytest.raises(ValueError, match='Invalid variable'):
        mgr._preprocessParameters(
            geom, watershed_workflow.crs.latlon_crs,
            None, None, ['not_a_real_variable'],
        )


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_get_dataset_default_variables(coweeta):
    """Download default variables (clay, sand, silt, bdod) for Coweeta."""
    mgr = ManagerSoilGrids(force_download=True)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs)
    assert isinstance(ds, xr.Dataset)
    for var in ['clay', 'sand', 'silt', 'bdod']:
        assert var in ds


@pytest.mark.network
def test_dataset_has_depth_dimension(coweeta):
    """Each variable must have 6 depth layers."""
    mgr = ManagerSoilGrids(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['clay'])
    assert 'depth' in ds['clay'].dims
    assert ds['clay'].sizes['depth'] == 6
    np.testing.assert_allclose(ds['clay'].depth.values, _DEPTH_CENTRES, rtol=1e-5)


@pytest.mark.network
def test_dataset_values_reasonable(coweeta):
    """Clay content should be in [0, 100] % after scaling."""
    mgr = ManagerSoilGrids(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['clay'])
    vals = ds['clay'].values
    valid = vals[np.isfinite(vals)]
    assert (valid >= 0).all()
    assert (valid <= 100).all()


@pytest.mark.network
def test_dataset_crs(coweeta):
    """Dataset must carry a CRS."""
    mgr = ManagerSoilGrids(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['clay'])
    assert ds.rio.crs is not None


@pytest.mark.network
def test_rosetta_variables_present(coweeta):
    """Default download (clay+sand+silt+bdod) must produce VG parameter variables."""
    mgr = ManagerSoilGrids(force_download=False)
    ds = mgr.getDataset(coweeta.geometry[0], coweeta.crs)
    for v in ['residual saturation [-]', 'porosity [-]',
              'van Genuchten alpha [Pa^-1]', 'van Genuchten n [-]', 'permeability [m^2]']:
        assert v in ds, f'Missing variable: {v}'
        assert ds[v].sizes['depth'] == 6


@pytest.mark.network
def test_cache_reuse(coweeta, tmp_path, monkeypatch):
    """Second call with the same bounds must reuse the cache, not re-download."""
    import watershed_workflow.utils.config
    monkeypatch.setitem(
        watershed_workflow.utils.config.rcParams['DEFAULT'],
        'data_directory', str(tmp_path))

    mgr = ManagerSoilGrids(force_download=False)
    ds1 = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['silt'])
    ds2 = mgr.getDataset(coweeta.geometry[0], coweeta.crs, variables=['silt'])
    xr.testing.assert_equal(ds1, ds2)
