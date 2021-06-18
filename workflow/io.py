"""I/O Utilities"""

import fiona
import shapely.geometry
import collections

import workflow.crs

# write the shapefile
def write_to_shapefile(filename, shps, crs, extra_properties=None):
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

    with fiona.open(filename, 'w', 
                    driver='ESRI Shapefile', 
                    crs=workflow.crs.to_fiona(crs), 
                    crs_wkt=workflow.crs.to_wkt(crs),
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
                  
            c.write({'geometry': shapely.geometry.mapping(shp),
                     'properties': props })

        
