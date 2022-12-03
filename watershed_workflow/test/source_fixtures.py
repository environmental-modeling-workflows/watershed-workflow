import os
from distutils import dir_util
import pytest
import watershed_workflow.source_list
import fiona

import pickle


class FileManagerMockNHDPlusSave(watershed_workflow.source_list.FileManagerNHDPlus):
    """A mock class for logging which HUCs are used in tests, for saving and mocking

    See watershed_workflow/test/source_fixtures_helpers.py
    """
    def get_hucs(self, huc, level, *args, **kwargs):
        # note, do this first, in case it errors
        result = super(FileManagerMockNHDPlusSave, self).get_hucs(huc, level, *args, **kwargs)

        with open('/tmp/log.txt', 'a') as fid:
            fid.write(f'writing {huc} level {level}\n')
        try:
            with open('/tmp/my.pkl', 'rb') as fid:
                d = pickle.load(fid)
        except FileNotFoundError:
            d = dict()

        if huc not in d:
            d[huc] = dict()
        d[huc][level] = True

        with open('/tmp/my.pkl', 'wb') as fid:
            pickle.dump(d, fid)

        return result

    def get_hydro(self, huc, *args, **kwargs):
        result = super(FileManagerMockNHDPlusSave, self).get_hydro(huc, *args, **kwargs)

        with open('/tmp/log.txt', 'a') as fid:
            fid.write(f'writing {huc} hydro\n')
        try:
            with open('/tmp/my.pkl', 'rb') as fid:
                d = pickle.load(fid)
        except FileNotFoundError:
            d = dict()

        if huc not in d:
            d[huc] = dict()
        d[huc]['hydro'] = True

        with open('/tmp/my.pkl', 'wb') as fid:
            pickle.dump(d, fid)

        return result


class FileManagerMockNHDPlusRestore:
    """A second mock class that restores the files created using the above save.

    See watershed_workflow/test/source_fixtures_helpers.py
    """
    lowest_level = 12
    name = 'FileManagerMockNHDPlusRestore'

    def __init__(self, dirname):
        self._dirname = dirname
        with fiona.open(os.path.join(dirname, 'hucs.shp'), 'r') as fid:
            self._profile = fid.profile
            self._hucs = list(fid)

    def get_hucs(self, huc, level, *args, **kwargs):
        this_hucs = [
            h for h in self._hucs
            if len(h['properties']['HUC']) == level and h['properties']['HUC'].startswith(huc)
        ]
        for h in this_hucs:
            h['properties'][f'huc{level}'] = h['properties']['HUC']

        if len(this_hucs) == 0:
            raise ValueError(f'Cannot read level {level} HUCs in {huc}')
        return self._profile.copy(), this_hucs

    def get_hydro(self, huc, *args, **kwargs):
        with fiona.open(os.path.join(self._dirname, f'river_{huc}.shp'), 'r') as fid:
            profile = fid.profile
            reaches = list(fid)
        return profile, reaches


@pytest.fixture
def sources(tmpdir, request):
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    shared_test_data = os.path.join(test_dir, 'fixture_data')
    sources = dict()
    #sources['HUC'] = FileManagerMockNHDPlusSave()
    sources['HUC'] = FileManagerMockNHDPlusRestore(
        os.path.join('watershed_workflow', 'test', 'fixture_data'))
    sources['hydrography'] = sources['HUC']
    return sources


@pytest.fixture
def datadir(tmpdir, request):
    """Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


@pytest.fixture
def sources_download(request):
    return watershed_workflow.source_list.get_default_sources()
