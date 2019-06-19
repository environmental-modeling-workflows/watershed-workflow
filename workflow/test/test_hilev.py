import os
from distutils import dir_util
import pytest
import fiona
import shapely.geometry
import workflow.hilev
import workflow.sources

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
def sources():
    sources = dict()
    #sources['HUC08'] = workflow.files.NHDFileManager()
    sources['HUC'] = workflow.sources.FileManagerNHDPlus()
    sources['DEM'] = workflow.sources.FileManagerNED()
    return sources


def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]
    return profile,shapely.geometry.shape(shp['geometry'])


