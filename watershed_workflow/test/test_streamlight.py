
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pytest

import watershed_workflow.stream_light

@pytest.fixture
def sl_data():
    sl = watershed_workflow.stream_light.StreamLight()
    path_data_test = './watershed_workflow/test/data_test_streamlight/'
    
    ## Files with the drivers and results
    data_ts = input_data_test_streamlight_df = pd.read_csv(os.path.join(path_data_test,'data_test_streamlight_time_series.csv'))

    data_params = input_data_test_streamlight_df = pd.read_csv(os.path.join(path_data_test,'data_test_streamlight_parameters.csv'))

    ## Channel characteristics 
    lat = data_params['lat'].values
    lon = data_params['lon'].values
    channel_azimuth = data_params['channel_azimuth'].values
    bottom_width = data_params['bottom_width'].values
    bank_height = data_params['bh'].values
    bank_slope = data_params['bs'].values
    water_depth = data_params['wl'].values
    tree_height = data_params['th'].values
    overhang = data_params['overhang'].values
    overhang_height = data_params['overhang_height'].values
    x_LAD = data_params['x_LAD'].values

    # Drivers for test simulation

    # Extract information from test drivers
    doy = data_ts['DOY'].values
    hour = data_ts['Hour'].values
    tz_offset = data_ts['offset'].values
    sw_inc = data_ts['SW_inc'].values
    lai = data_ts['LAI'].values

    # Run pyton implementation of StreamLight

    ## Channel characteristics 
    sl.set_channel_properties(lat = lat,lon = lon,channel_azimuth = channel_azimuth,bottom_width = bottom_width, bank_height = bank_height, bank_slope = bank_slope, water_depth = water_depth,
    tree_height = tree_height, overhang = overhang, overhang_height = overhang_height, x_LAD = x_LAD)

    # Energy drivers
    sl.set_energy_drivers(doy = doy, hour = hour, tz_offset = tz_offset, sw_inc = sw_inc, lai = lai)

    # Run StreamLight
    sl.run_streamlight()
    #sl.data_comparison = data_comparison

    return sl, data_ts, path_data_test

def plot_comparison(var_estimate, var_reference, var_name, ax):
    rmse_solar_dec = np.linalg.norm2(var_reference - var_estimate)
    ax.plot(var_reference,var_reference,'-k', lw = 0.5)
    ax.scatter(var_reference,var_estimate, c = 'red', alpha=.1, s=2)
    ax.set_aspect('equal')
    the_title = '{vname}, (RMSE = {vrmse:.0e})'
    ax.set_title(the_title.format(vname=var_name,vrmse=rmse_solar_dec), loc='left')
    ax.set_xlabel('From StreamLight in R')
    ax.set_ylabel('From StreamLight in Python')

def test_solar_dec(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):
    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'solar_dec'
    var_name_test = 'solar_dec'

    assert np.allclose(sl.solar_angles[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.solar_angles[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)
    
def test_solar_altitude(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):
    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'solar_altitude'
    var_name_test = 'solar_altitude'

    assert np.allclose(sl.solar_angles[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.solar_angles[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)

def test_sza(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):

    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'sza'
    var_name_test = 'SZA'

    assert np.allclose(sl.solar_angles[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.solar_angles[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)


def test_solar_azimuth(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):

    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'solar_azimuth'
    var_name_test = 'solar_azimuth2'

    assert np.allclose(sl.solar_angles[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.solar_angles[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)

def test_total_trans_PAR_ppfd(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):

    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'total_trans_PAR_ppfd'
    var_name_test = 'PAR_bc'

    assert np.allclose(sl.energy_response[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.energy_response[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)

def test_fraction_shade_veg(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):

    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'fraction_shade_veg'
    var_name_test = 'veg_shade'

    assert np.allclose(sl.shading_response[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.shading_response[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)

def test_fraction_shade_bank(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):

    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'fraction_shade_bank'
    var_name_test = 'bank_shade'

    assert np.allclose(sl.shading_response[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.shading_response[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)

def test_energy_total_surface_PAR_ppfd(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08, plot_scatter = False):

    sl, data_ts, path_data_test = sl_data

    var_name_ww = 'energy_total_surface_PAR_ppfd'
    var_name_test = 'PAR_surface'

    assert np.allclose(sl.energy_response[var_name_ww], data_ts[var_name_test].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.energy_response[var_name_ww], data_ts[var_name_test].values,var_name_ww,ax)

        plt.tight_layout()
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".pdf")), transparent=True)
        plt.savefig(os.path.join(path_data_test,(var_name_ww + ".png")), dpi=300, transparent=True)
