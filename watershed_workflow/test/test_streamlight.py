
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import pytest

import watershed_workflow.stream_light


@pytest.fixture
def sl_data():
    sl = watershed_workflow.stream_light.StreamLight()
    path_data_test = './watershed_workflow/test/data_test_streamlight/'
    
    ## Channel characteristics 
    lat = 35.9925
    lon = -79.0460
    channel_azimuth = 330
    bottom_width = 18.9
    bank_height = 0.1
    bank_slope = 100
    water_depth = 0.05
    tree_height = 23
    overhang = 2.3
    overhang_height = 0.1*tree_height
    x_LAD = 1

    ## Files with the drivers and results
    input_data_test_streamlight_df = pd.read_csv(os.path.join(path_data_test,'input_data_test_streamlight.csv'))
    
    results_test_streamlight_df = pd.read_csv(os.path.join(path_data_test,'results_test_streamlight.csv'))
    
    test_solar_geo_calc_df = pd.read_csv(os.path.join(path_data_test,'input_data_test_solar_geo_calc.csv'))
    
    test_rt_cn_1998_df = pd.read_csv(os.path.join(path_data_test,'input_data_test_RT_CN_1998.csv'))
    
    test_shade2_df = pd.read_csv(os.path.join(path_data_test,'input_data_test_SHADE2.csv'))
    
    test_stream_light_df = pd.read_csv(os.path.join(path_data_test,'input_data_test_stream_light.csv'))
    
    data = pd.concat([input_data_test_streamlight_df, results_test_streamlight_df, test_solar_geo_calc_df, test_rt_cn_1998_df, test_shade2_df, test_stream_light_df],axis =1)

    # Drivers for test simulation

    # Extract information from test drivers
    doy = data['DOY'].values
    hour = data['Hour'].values
    tz_offset = data['offset'].values
    sw_inc = data['SW_inc'].values
    lai = data['LAI'].values

    # Run pyton implementation of StreamLight

    ## Channel characteristics 
    sl.set_channel_properties(lat = lat,lon = lon,channel_azimuth = channel_azimuth,bottom_width = bottom_width, bank_height = bank_height, bank_slope = bank_slope, water_depth = water_depth,
    tree_height = tree_height, overhang = overhang, overhang_height = overhang_height, x_LAD = x_LAD)

    # Energy drivers
    sl.set_energy_drivers(doy = doy, hour = hour, tz_offset = tz_offset, sw_inc = sw_inc, lai = lai)

    # Run StreamLight
    sl.run_streamlight()
    #sl.data_comparison = data_comparison

    return sl, data, path_data_test


def test_solar_dec(sl_data, relative_tolerance = 1e-05, absolute_tolerance = 1e-08,plot_scatter = False):

    sl, data, path_data_test = sl_data

    #assert np.allclose(sl.solar_angles['solar_dec'], data['solar_dec'].values, rtol=relative_tolerance, atol=absolute_tolerance, equal_nan=True)
    
    print(os.getcwd())

    assert len(sl.solar_angles['solar_dec']) == len(data['solar_dec'].values)

    plot_scatter = False
    if plot_scatter:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_comparison(sl.solar_angles['solar_dec'], data['solar_dec'].values,'solar_dec',ax)

        plt.tight_layout()#(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(os.path.join(path_data_test,"test_StreamLight.pdf"), transparent=True)
        plt.savefig(os.path.join(path_data_test,"test_StreamLight.png"), dpi=300, transparent=True)
    

def plot_comparison(var_estimate, var_reference, var_name, ax):

    rmse_solar_dec = mean_squared_error(var_reference,var_estimate, squared=False)
    ax.plot(var_reference,var_reference,'-k', lw = 0.5)
    ax.scatter(var_reference,var_estimate, c = 'red', alpha=.1, s=2)
    ax.set_aspect('equal')
    the_title = '{vname}, (RMSE = {vrmse:.0e})'
    ax.set_title(the_title.format(vname=var_name,vrmse=rmse_solar_dec), loc='left')
    ax.set_xlabel('From StreamLight in R')
    ax.set_ylabel('From StreamLight in Python')
    