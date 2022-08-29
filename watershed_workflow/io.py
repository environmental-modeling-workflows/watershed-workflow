"""I/O Utilities"""

import fiona
import rasterio
import shapely.geometry
import collections

import watershed_workflow.crs


def write_to_raster(filename, profile, array):
    """Write a numpy array to raster file."""
    assert (len(array.shape) >= 2 and len(array.shape) <= 3)
    if len(array.shape) == 2:
        array = array.reshape((1, ) + array.shape)

    profile = profile.copy()
    profile.update(count=array.shape[2], compress='lzw')

    with rasterio.open(filename, 'w', **profile) as fout:
        for i in range(array.shape[0]):
            fout.write(array[i, :, :], i + 1)


def write_to_shapefile(filename, shps, crs, extra_properties=None):
    """Write a collection of shapes to a file using fiona"""
    if len(shps) == 0:
        return

    # set up the schema
    schema = dict()
    if type(shps[0]) is shapely.geometry.Polygon:
        schema['geometry'] = 'Polygon'
    elif type(shps[0]) is shapely.geometry.LineString:
        schema['geometry'] = 'LineString'
    else:
        raise RuntimeError('Currently this function only writes Polygon or LineString types')
    schema['properties'] = collections.OrderedDict()

    # set up the properties schema
    def register_type(key, atype):
        if atype is int:
            schema['properties'][key] = 'int'
        elif atype is str:
            schema['properties'][key] = 'str'
        elif atype is float:
            schema['properties'][key] = 'float'
        else:
            pass

    if extra_properties is None:
        extra_properties = dict()
    for key, val in extra_properties.items():
        register_type(key, type(val))

    try:
        item_properties = shps[0].properties
    except AttributeError:
        pass
    else:
        for key, val in item_properties.items():
            register_type(key, type(val))

    with fiona.open(filename,
                    'w',
                    driver='ESRI Shapefile',
                    crs=watershed_workflow.crs.to_fiona(crs),
                    crs_wkt=watershed_workflow.crs.to_wkt(crs),
                    schema=schema) as c:
        for shp in shps:
            props = extra_properties.copy()
            try:
                props.update(shp.properties)
            except AttributeError:
                pass

            for key in list(props.keys()):
                if key not in schema['properties']:
                    props.pop(key)

            c.write({ 'geometry': shapely.geometry.mapping(shp), 'properties': props })
