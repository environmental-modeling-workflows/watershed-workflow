"""Test url generation"""

import os
import pytest
import workflow.files

import workflow.conf

@pytest.fixture
def source():
    return workflow.files.HucNames(digits=4,
                                   name='NHD Plus High Resoluton Hydrography',
                                   url='https://www.url.gov',
                                   base_folder='hydrography',
                                   folder_template='NHDPlus_H_{0}_GDB',
                                   file_template='NHDFlowline.shp')

def test_parse_huc(source):
    dd = workflow.conf.rcParams['data dir']
    
    assert(source.folder_name('0601') == os.path.join(dd,'hydrography/NHDPlus_H_0601_GDB'))
    assert(source.folder_name('601') == os.path.join(dd,'hydrography/NHDPlus_H_0601_GDB'))
    assert(source.folder_name('060101') == os.path.join(dd,'hydrography/NHDPlus_H_0601_GDB'))
    with pytest.raises(ValueError):
        source.folder_name('06')

def test_derived(source):
    dd = workflow.conf.rcParams['data dir']
    assert(source.url(0, '0601') == "https://www.url.gov/NHDPlus_H_0601_GDB.zip")
    assert(source.file_name('0601') == os.path.join(dd,'hydrography/NHDPlus_H_0601_GDB/NHDFlowline.shp'))
    assert(source.download(0, '0601') == os.path.join(dd,'hydrography/zips/NHDPlus_H_0601_GDB.zip'))


def test_download():
    fm = workflow.files.NHDHucOnlyFileManager()
    src = fm.names
    print(src.download(0,'0601'))
    print(src.url(0,'0601'))
    workflow.files._download(src.url(0,'0601'),src.download(0,'0601'))
    workflow.files._unzip(src.download(0,'0601'), src.folder_name('0601'))


    
    
