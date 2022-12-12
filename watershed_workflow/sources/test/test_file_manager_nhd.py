import pytest

import os
from distutils import dir_util
import shapely
import numpy as np
import fiona

import watershed_workflow.config
import watershed_workflow.warp
import watershed_workflow.sources.manager_nhd
import watershed_workflow.sources.utils as sutils

bounds4_ll = np.array(
    [-85.19249725424618, 34.903371461448046, -81.26111840201514, 37.24054551094525])
bounds8_ll = np.array(
    [-83.82022857720955, 34.903371461448046, -83.14341380430176, 35.58000945935606])

#bounds4_crs = np.array(list(watershed_workflow.warp.xy(bounds4[0], bounds4[1], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())) + list(watershed_workflow.warp.xy(bounds4[2], bounds4[3], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())))

#bounds8_crs = np.array(list(watershed_workflow.warp.xy(bounds8[0], bounds8[1], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())) + list(watershed_workflow.warp.xy(bounds8[2], bounds8[3], watershed_workflow.crs.latlon_crs(), watershed_workflow.crs.default_crs())))


@pytest.fixture
def nhd():
    return watershed_workflow.sources.manager_nhd.FileManagerNHD()


def test_nhd_url(nhd):
    url = nhd._url('02040101')
    assert (
        'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/GDB/NHD_H_02040101_HU8_GDB.zip'
        == url)


def test_nhd_url2(nhd):
    url = nhd._url('06010202')
    assert (
        'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/GDB/NHD_H_06010202_HU8_GDB.zip'
        == url)


def test_nhd_url3(nhd):
    url = nhd._url('14020001')
    assert (
        'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/GDB/NHD_H_14020001_HU8_GDB.zip'
        == url)


def test_nhd_url4(nhd):
    url = nhd._url('19060402')
    assert (
        'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/GDB/NHD_H_19060402_HU8_GDB.zip'
        == url)


def test_nhd_url_fail(nhd):
    with pytest.raises(ValueError):
        url = nhd._url('02010000')  # this huc is not a real huc


def test_nhd_url_invalid(nhd):
    with pytest.raises(ValueError):
        h = nhd.get_hucs('0201', 2)  # not allowed, on 8 only


def test_nhd_download(nhd):
    # download
    hfile = nhd._download('06010202', force=True)
    assert (hfile == nhd.name_manager.file_name('06010202'))


def test_nhd_get(nhd):
    # download
    profile, huc = nhd.get_huc('06010202')
    bounds = watershed_workflow.utils.create_shply(huc['geometry']).bounds
    assert (np.allclose(bounds8_ll, np.array(bounds), 1.e-6))


# hydro tests
def test_nhd_get_hydro(nhd):
    profile, rivers = nhd.get_hydro('060102020103')
    assert (198 == len(rivers))  # note this is different from NHDPlus
