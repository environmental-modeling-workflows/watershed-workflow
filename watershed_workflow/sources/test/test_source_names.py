import pytest

import watershed_workflow.sources.names
import watershed_workflow.conf

def test_names():
    ddir = watershed_workflow.conf.rcParams['DEFAULT']['data_directory']
    watershed_workflow.conf.rcParams['DEFAULT']['data_directory'] = '/my'
    
    names = watershed_workflow.sources.names.Names('mynames', 'hydrography', 'rivers_{}', 'rivers_{}.gdb')
    assert('rivers_0102.gdb' == names.file_name_base('0102'))
    assert('/my/hydrography' == names.data_dir())
    assert('/my/hydrography/rivers_0102' == names.folder_name('0102'))
    assert('/my/hydrography/rivers_0102/raw' == names.raw_folder_name('0102'))
    assert('/my/hydrography/rivers_0102/rivers_0102.gdb' == names.file_name('0102'))
    watershed_workflow.conf.rcParams['DEFAULT']['data_directory'] = ddir
