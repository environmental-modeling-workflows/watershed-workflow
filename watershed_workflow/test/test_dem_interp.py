import pytest

import rasterio.transform
import rasterio.crs
import numpy as np

import watershed_workflow
import watershed_workflow.conf

@pytest.fixture
def dem_and_points():
    # create a dem
    dem = np.ones((2,2))
    dem[1,0] = 2
    dem[1,1] = 10

    # with a profile
    dem_profile = dict()
    dem_profile['crs'] = rasterio.crs.CRS.from_epsg(5070)
    dem_profile['transform'] = rasterio.transform.Affine(1,0,0,0,1,0)
    dem_profile['height'] = 2
    dem_profile['width'] = 2
    dem_profile['offset'] = (0,0)

    # create some points to sample
    xy = np.array([ (0.000001,0.000001),
                    (.5,.5),
                    ( .9999999,  .9999999),
                    (1,1),
                    (1.0000001, 1.0000001),
                    (1.5,1.5),
                    (1.9999999, 1.9999999),
                    (.5, 1.5),
                    (1.5,.5),
    ])

    return dem, dem_profile, xy


def test_nearest(dem_and_points):
    dem, dem_profile, xy = dem_and_points
    vals = watershed_workflow.values_from_raster(xy, watershed_workflow.crs.from_rasterio(dem_profile['crs']), dem, dem_profile,'nearest')
    assert(np.allclose(np.array([1,1,1,10,10,10,10,2,1]), vals))


def test_interp(dem_and_points):
    dem, dem_profile, xy = dem_and_points
    vals = watershed_workflow.values_from_raster(xy, watershed_workflow.crs.from_rasterio(dem_profile['crs']), dem, dem_profile,'piecewise bilinear')
    assert(np.allclose(np.array([1,1,3.5,3.5,3.5,10,10,2,1]), vals, 1.e-4))
    
