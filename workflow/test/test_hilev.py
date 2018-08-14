import os
from distutils import dir_util
import pytest
import fiona
import shapely.geometry
import workflow.hilev
import workflow.files

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
    sources['HUC'] = workflow.files.NHDHucOnlyFileManager()
    sources['DEM'] = workflow.files.NEDFileManager()
    return sources


def get_fiona(filename):
    with fiona.open(str(filename), 'r') as fid:
        profile = fid.profile
        shp = fid[0]
    return profile,shapely.geometry.shape(shp['geometry'])

def test_12(datadir, sources):
    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    shp = shp.buffer(-0.03)
    print(shp.area)
    assert('060300010402' == workflow.hilev.find_huc(profile, shp, sources['HUC'], '06'))

def test_12_exact(datadir, sources):
    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    shp = shp.buffer(-0.03)
    print(shp.area)
    assert('060300010402' == workflow.hilev.find_huc(profile, shp, sources['HUC'], '060300010402'))

def test_12_raises(datadir, sources):
    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    shp = shp.buffer(-0.03)
    print(shp.area)
    with pytest.raises(RuntimeError):
         workflow.hilev.find_huc(profile, shp, sources['HUC'], '060300010403')
    
def test_08(datadir, sources):
    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    assert('06030001' == workflow.hilev.find_huc(profile, shp, sources['HUC'], '06'))

def test_08_exact(datadir, sources):
    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    assert('06030001' == workflow.hilev.find_huc(profile, shp, sources['HUC'], '06030001'))

def test_08_raises(datadir, sources):
    testshpfile = datadir.join('test_polygon.shp')
    profile, shp = get_fiona(testshpfile)
    with pytest.raises(RuntimeError):
        workflow.hilev.find_huc(profile, shp, sources['HUC'], '0604')

