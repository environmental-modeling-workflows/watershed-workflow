"""Used to warp shapefiles and rasters into new coordinate systems."""

import shutil
import numpy as np
import logging

import pyproj
import rasterio.warp
import shapely.geometry

import warnings
import watershed_workflow.crs

pyproj_version = int(pyproj.__version__[0])


def xy(x, y, old_crs, new_crs):
    """Warp a set of points from old_crs to new_crs."""
    if watershed_workflow.crs.equal(old_crs, new_crs):
        return x, y

    old_crs_proj = watershed_workflow.crs.to_proj(old_crs)
    new_crs_proj = watershed_workflow.crs.to_proj(new_crs)

    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    x1, y1 = transformer.transform(x, y)
    return x1, y1


def bounds(bounds, old_crs, new_crs):
    """Warp a bounding box from old_crs to new_crs."""
    return shply(shapely.geometry.box(*bounds), old_crs, new_crs).bounds


def shply(shp, old_crs, new_crs):
    """Warp a shapely object from old_crs to new_crs."""
    if watershed_workflow.crs.equal(old_crs, new_crs):
        return shp
    old_crs_proj = watershed_workflow.crs.to_proj(old_crs)
    new_crs_proj = watershed_workflow.crs.to_proj(new_crs)
    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    shp_out = shapely.ops.transform(transformer.transform, shp)
    if hasattr(shp, 'properties'):
        shp_out.properties = shp.properties
    return shp_out


def shplys(shps, old_crs, new_crs):
    """Warp a collection of shapely objects from old_crs to new_crs."""
    if watershed_workflow.crs.equal(old_crs, new_crs):
        return shps
    old_crs_proj = watershed_workflow.crs.to_proj(old_crs)
    new_crs_proj = watershed_workflow.crs.to_proj(new_crs)
    transformer = pyproj.Transformer.from_crs(old_crs_proj, new_crs_proj, always_xy=True)
    shps_out = [shapely.ops.transform(transformer.transform, shp) for shp in shps]
    for sout, sin in zip(shps_out, shps):
        if hasattr(sin, 'properties'):
            sout.properties = sin.properties
    return shps_out


def shape(feature, old_crs, new_crs):
    """Warp a fiona shape from old_crs to new_crs, in place."""
    # find the dimension -- can't trust the shape
    dim = -1
    ptr = feature['geometry']['coordinates']
    done = False
    while not done:
        if hasattr(ptr, '__len__'):
            assert (len(ptr) != 0)
            dim += 1
            ptr = ptr[0]
        else:
            done = True

    if dim == 0:
        # point
        x, y = xy(np.array([feature['geometry']['coordinates'][0], ]),
                  np.array([feature['geometry']['coordinates'][1], ]), old_crs, new_crs)
        feature['geometry']['coordinates'] = (x[0], y[0])

    elif dim == 1:
        # line-like or polygon with no holes
        coords = np.array(feature['geometry']['coordinates'], 'd')
        assert (len(coords.shape) == 2 and coords.shape[1] in [2, 3])
        x, y = xy(coords[:, 0], coords[:, 1], old_crs, new_crs)
        new_coords = [xy for xy in zip(x, y)]
        feature['geometry']['coordinates'] = new_coords

    elif dim == 2:
        # multi-line or polygon with holes
        new_rings = []
        for i in range(len(feature['geometry']['coordinates'])):
            coords = np.array(feature['geometry']['coordinates'][i], 'd')
            assert (len(coords.shape) == 2 and coords.shape[1] in [2, 3])
            x, y = xy(coords[:, 0], coords[:, 1], old_crs, new_crs)
            new_coords = list(zip(x, y))
            new_rings.append(new_coords)
        feature['geometry']['coordinates'] = new_rings

    elif dim == 3:
        # multi-polygon
        for i in range(len(feature['geometry']['coordinates'])):
            for j in range(len(feature['geometry']['coordinates'][i])):
                coords = np.array(feature['geometry']['coordinates'][i][j], 'd')
                assert (len(coords.shape) == 2 and coords.shape[1] in [2, 3])
                x, y = xy(coords[:, 0], coords[:, 1], old_crs, new_crs)
                new_coords = [xy for xy in zip(x, y)]
                feature['geometry']['coordinates'][i][j] = new_coords
    return feature


def raster(src_profile,
           src_array,
           dst_crs=None,
           dst_resolution=None,
           dst_height=None,
           dst_width=None,
           resampling_method=rasterio.warp.Resampling.nearest):
    """Warp a raster from src_profile to dst_profile, or resample resolution."""
    if watershed_workflow.crs.is_native(src_profile['crs']):
        src_crs = src_profile['crs']
        src_crs_rio = watershed_workflow.crs.to_rasterio(src_crs)
    else:
        src_crs_rio = src_profile['crs']
        src_crs = watershed_workflow.crs.from_rasterio(src_crs_rio)

    if dst_resolution is None and dst_height is None and dst_width is None and \
       (dst_crs is None or watershed_workflow.crs.equal(dst_crs, src_crs)):
        # nothing to do
        return src_profile, src_array

    if dst_resolution is None and dst_height is None:
        dst_height = src_profile['height']
    if dst_resolution is None and dst_width is None:
        dst_width = src_profile['width']

    if dst_crs is None:
        dst_crs = src_crs
    if watershed_workflow.crs.is_native(dst_crs):
        dst_crs_rio = watershed_workflow.crs.to_rasterio(dst_crs)
    else:
        dst_crs_rio = dst_crs

    src_bounds = rasterio.transform.array_bounds(src_profile['height'], src_profile['width'],
                                                 src_profile['transform'])
    logging.debug('Warping raster with bounds: {} to CRS: {}'.format(src_bounds, dst_crs))

    # Calculate the ideal dimensions and transformation in the new crs
    dst_profile = src_profile.copy()
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs_rio,
        dst_crs_rio,
        src_profile['width'],
        src_profile['height'],
        *src_bounds,
        resolution=dst_resolution,
        dst_height=dst_height,
        dst_width=dst_width)

    # update the relevant parts of the profile
    dst_profile.update({
        'crs': dst_crs_rio,
        'transform': dst_transform,
        'width': dst_width,
        'height': dst_height,
        'dtype': src_profile['dtype']
    })

    # Reproject and return
    logging.debug(f'  src_array shape: {src_array.shape}')
    if src_array.ndim == 3:
        nband = src_array.shape[0]
        # src_array must have dim (nband, nrow, cols) or (nband, height, width)
        assert (src_array.shape == (nband, src_profile['height'], src_profile['width']))
        dst_array = np.empty((nband, dst_height, dst_width), dtype=src_array.dtype)
    else:
        nband = 1
        assert (src_array.shape == (src_profile['height'], src_profile['width']))
        dst_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)
    logging.debug(f'  dst_array shape: {dst_array.shape}')

    dst_profile.update({ 'count': nband })
    rasterio.warp.reproject(src_array,
                            dst_array,
                            src_transform=src_profile['transform'],
                            src_crs=src_crs_rio,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs_rio,
                            dst_nodata=dst_profile['nodata'],
                            resampling=resampling_method)

    return dst_profile, dst_array


def dataset(dataset, **kwargs):
    """Warp a dataset collection.  Note this works in place!"""
    new_dataset = None
    for k, v in dataset.items():
        new_profile, new_vals = raster(v.profile, v.data, **kwargs)
        if new_dataset is None:
            new_dataset = watershed_workflow.datasets.Dataset(new_profile, v.times)
        new_dataset[k] = watershed_workflow.datasets.Data(new_profile, v.times, new_vals)
    return new_dataset


def state(state, **kwargs):
    """Warp a state."""
    new_state = watershed_workflow.datasets.State()
    for col in state.collections:
        new_col = dataset(col, **kwargs)
        new_state.collections.append(new_col)
    return new_state
