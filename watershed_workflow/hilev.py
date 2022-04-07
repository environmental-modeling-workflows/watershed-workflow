"""High-level routines -- these are imported into the top level `"watershed_workflow`" namespace.

This top level module provides functionality for getting shapes and rasters
representing watershed boundaries, river networks, digital elevation models,
and other GIS datasets and then processing those data sets for use in
simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import scipy

import rasterio
import rasterio.transform
import rasterio.features

import shapely

import watershed_workflow.config
import watershed_workflow.triangulation
import watershed_workflow.warp
import watershed_workflow.plot
import watershed_workflow.river_tree
import watershed_workflow.split_hucs
import watershed_workflow.hydrography
import watershed_workflow.sources.utils 
import watershed_workflow.sources.manager_shape
import watershed_workflow.utils

__all__ = ['get_huc', 'get_hucs', 'get_split_form_hucs',
           'get_shapes', 'get_split_form_shapes', 'find_huc',
           'get_reaches', 'construct_rivers', 'simplify', 'simplify_and_prune',
           'get_waterbodies',
           'triangulate', 'get_raster_on_shape', 'values_from_raster',
           'elevate', 'color_raster_from_shapes', 'color_existing_raster_from_shapes']

#
# functions for getting objects
# -----------------------------------------------------------------------------

def get_huc(source, huc, out_crs=None, digits=None):
    """Get a HUC shape object from a given code.

    Parameters
    ----------
    source : source-type
        source object providing `get_hucs()`
    huc : str
        hydrologic unit code
    out_crs : crs-type
        Output coordinate system.  Default is source's crs.
    digits : int, optional
        Number of digits to round coordinates to.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of `out`.
    out : Polygon
        shapely polygon for the hydrologic unit.

    """
    huc = watershed_workflow.sources.utils.huc_str(huc)

    logging.info("")
    logging.info(f"Loading HUC {huc}")
    logging.info("-"*30)

    out_crs, hu_shapes = get_hucs(source, huc, len(huc), out_crs, digits)
    logging.info(f"... found {len(hu_shapes)}")
    assert(len(hu_shapes) == 1)
    return out_crs, hu_shapes[0]


def get_hucs(source, huc, level, out_crs=None, digits=None):
    """Get shape objects for all HUCs at a level contained in huc.

    Parameters
    ----------
    source : source-type
        source object providing `get_hucs()`
    huc : str
        hydrologic unit code
    level : int
        HUC level of the requested sub-basins
    out_crs : crs-type
        Output coordinate system.  Default is source's crs.
    digits : int, optional
        Number of digits to round coordinates to.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of all entries in `out`.
    out : list(Polygon)
        List of shapely polygons for the subbasins.

    """
    # get the hu from source
    huc = watershed_workflow.sources.utils.huc_str(huc)
    if level is None:
        level = len(huc)

    logging.info("")
    logging.info(f"Loading level {level} HUCs in {huc}")
    logging.info("-"*30)
    
    profile, hus = source.get_hucs(huc, level)
    logging.info("... found {} HUCs".format(len(hus)))
    for hu in hus:
        logging.info('  -- {}'.format(watershed_workflow.sources.utils.get_code(hu,level)))
    
    # convert to destination crs
    native_crs = watershed_workflow.crs.from_fiona(profile['crs'])
    if out_crs and not watershed_workflow.crs.equal(out_crs, native_crs):
        logging.info("Converting to out_crs")
        for hu in hus:
            watershed_workflow.warp.shape(hu, native_crs, out_crs)
    else:
        out_crs = native_crs

    # round
    if digits != None:
        logging.info("Rounding coordinates")
        watershed_workflow.utils.round_shapes(hus, digits)

    # convert to shapely
    logging.info("Converting to shapely")
    hu_shapes = [watershed_workflow.utils.shply(hu) for hu in hus]
    return out_crs, hu_shapes


def get_split_form_hucs(source, huc, level=None, out_crs=None, digits=None):
    """Get a SplitHUCs object for all HUCs at level contained in huc.

    A SplitHUCs object is an object which stores a collection of polygons which
    share boundaries in a format that makes changing those shared boundaries
    possible without having to update all shapes that share the boundary.

    Parameters
    ----------
    source : source-type
        source object providing `get_hucs()`
    huc : str
        hydrologic unit code
    level : int
        HUC level of the requested sub-basins
    out_crs : crs-type
        Output coordinate system.  Default is source's crs.
    digits : int, optional
        Number of digits to round coordinates to.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of `out`.
    out : SplitHUCs
        Split-form HUCs object containing subbasins.

    """
    out_crs, hu_shapes = get_hucs(source, huc, level, out_crs, digits)
    logging.info("Creating split form")
    return out_crs, watershed_workflow.split_hucs.SplitHUCs(hu_shapes)


def get_shapes(source, index_or_bounds=None, in_crs=None, out_crs=None, digits=None, properties=False, **kwargs):
    """Read a shapefile.

    If index_or_bounds is a bounding box, in_crs must not be None and is the crs
    of the bounding box.

    Parameters
    ----------
    source : str or source-type
        Filename to parse, or a source object providing the get_shapes()
        method.
    index_or_bounds : int or [x_min, y_min, x_max, y_max] bounds, optional
        Filter the file, either by selecting a specific shape by index of the
        requested shape in the file, or providing a bounds-type tuple to select
        only shapes that intersect with the bounding box.
    in_crs : crs-type, optional
        Coordinate system of the bounding box.
    out_crs : crs-type, optional
        Coordinate system to which shapes will be warped.  If not
        provided, the native crs of the file will be used.
    digits : int, optional
        Number of digits to round coordinates to.
    properties : bool, optional
        If true, also get properties from the source.
    kwargs : dict
        All extra parameters are passed to the source manager's function.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of `out`.
    out : list(shapely)
        List of shapely objects in the shapefile meeting the criteria.
    out_properties : pandas dataframe
        Only if properties == True, a dataframe of corresponding properties

    """
    logging.info("")
    logging.info("Loading shapes")
    logging.info("-"*30)

    # load shapefile
    if type(source) is str:
        source_filename = source
        logging.info(f"Loading file: '{source}'")      
        source = watershed_workflow.sources.manager_shape.FileManagerShape(source)

    if properties:
        profile, shps, out_props = source.get_shapes_and_properties(index_or_bounds, in_crs, **kwargs)
    else:
        profile, shps = source.get_shapes(index_or_bounds, in_crs, **kwargs)
    logging.info(f"... found {len(shps)} shapes")

    # convert to shapely
    logging.info("Converting to shapely")
    if len(shps) == 0:
        shplys = []
    elif type(shps[0]) is dict:
        shplys = [watershed_workflow.utils.shply(shp) for shp in shps]
    else:
        shplys = shps
    
    # convert to destination crs
    native_crs = watershed_workflow.crs.from_fiona(profile['crs'])
    if out_crs and not watershed_workflow.crs.equal(out_crs, native_crs):
        logging.info("Converting to requested CRS")
        shplys = watershed_workflow.warp.shplys(shplys, native_crs, out_crs)
    else:
        out_crs = native_crs
        
    # round
    if digits != None:
        logging.info("Rounding coordinates")
        watershed_workflow.utils.round_shplys(shplys, digits)

    if properties:
        return out_crs, shplys, out_props
    else:
        return out_crs, shplys

def get_split_form_shapes(source, index_or_bounds=-1, in_crs=None, out_crs=None, digits=None):
    """Read a shapefile.

    Note that if index_or_bounds is a bounding box, in_crs must not be
    None and is the crs of the bounding box.

    Parameters
    ----------
    source : str or source-type
        Filename to parse, or a source object providing the get_shapes() method.
    index_or_bounds : int or [x_min, y_min, x_max, y_max] bounds, optional
        Filter the shapes, either by selecting a specific shape by index of the
        requested shape in the file, or providing a bounds-type tuple to select
        only shapes that intersect with the bounding box.
    in_crs : crs-type, optional
        Coordinate system of out and/or the bounding box provided.
    out_crs : crs-type, optional
        Coordinate system to which shapes will be warped.  If not
        provided, the native crs of the file will be used.
    digits : int, optional
        Number of digits to round coordinates to.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of `out`.
    out : SplitHUCs
        Split-form polygons object containing subbasins.

    """
    crs, shapes = get_shapes(source, index_or_bounds, in_crs, out_crs, digits)
    return crs, watershed_workflow.split_hucs.SplitHUCs(shapes)


def get_reaches(source, huc, bounds_or_shp=None, in_crs=None, out_crs=None,
                digits=None, tol=None, long=None, merge=False, presimplify=None, properties=None,
                **kwargs):
    """Get reaches from hydrography source within a given HUC and/or bounding box.

    Collects reach datasets within a HUC and/or a bounding box.  If bounds are
    provided, a containing HUC must still be provided to give a hint for file
    downloads.  If bounds are not provided, then all reaches that intersect the
    HUC are included.

    If bounds is provided, crs must not be None and is the crs of the bounding box.

    Parameters
    ----------
    source : source-type
        Source object providing a get_hydro() method.
    huc : str
        HUC containing reaches.  If bounds are provided, a hint to help the
        source find the file containing the bounds.  For NHD, this is a HUC4 or
        smaller.
    bounds_or_shp : [xmin, ymin, xmax, ymax] bounds or shly object, optional
        Bounding box to filter the river network.
    in_crs : crs-type, optional
        Coordinate system of the bounds.
    out_crs : crs-type, optional
        Coordinate system of the output reaches.  Default is the
        native crs of the source.
    digits : int, optional
        Number of digits to round coordinates to.
    tol : float, optional
        Tolerance used in filtering the reaches to the provided shape.
    long : float, optional
        If a reach is longer than this value it gets filtered.  Some
        NHD data has impoundments or other artifacts which appear
        as very long, perfectly straight single segment reaches.
    merge : bool, optional
        If true, reaches are merged (via shapely.ops.linemerge), collapsing 
        connected, non-branching reaches into a single LineString.
    presimplify : double, optional
        If provided, reaches are simplified within the specified
        tolerance as soon as possible for big extents.  Units are that
        of out_crs.
    properties : a list of properties to be added to reaches 'catchment' for catchment geometry, and property alias names for NHDPlusFlowlineVAA and NHDPlusEROMMA table 
        (Table 16 and 17 NHDPlus user guide)
    **kwargs : dict, optional
        Other arguments are passed to the file manager's get_reaches() method.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of `out`.
    out : list(LineString)
        Reaches in the HUC and/or intersecting the bounds.

    """
    logging.info("")
    logging.info("Loading Hydrography")
    logging.info("-"*30)
    logging.info(f"Loading streams in HUC {huc}")

    if isinstance(bounds_or_shp, tuple):
        bounds = bounds_or_shp
        bounds_or_shp = None
    else:
        bounds = bounds_or_shp.bounds
    logging.info(f"         and/or bounds {bounds}")

    # get the reaches
    profile, reaches = source.get_hydro(huc, bounds, in_crs, properties=properties, **kwargs)
    logging.info("... found {} reaches".format(len(reaches)))
  
    # convert to shapely
    logging.info("Converting to shapely")
    reaches = [watershed_workflow.utils.shply(reach) for reach in reaches]
        
    # convert to destination crs
    native_crs = watershed_workflow.crs.from_fiona(profile['crs'])
    if out_crs and not watershed_workflow.crs.equal(out_crs, native_crs):
        logging.info("Converting to out_crs")
        reaches = watershed_workflow.warp.shplys(reaches, native_crs, out_crs)

        for reach in reaches:
            if 'catchment' in reach.properties and reach.properties['catchment'] != None:
                reach.properties['catchment'] = watershed_workflow.utils.shply(reach.properties['catchment'])
                reach.properties['catchment'] = watershed_workflow.warp.shply(reach.properties['catchment'], native_crs, out_crs)
    else:
        out_crs = native_crs

    # filter to the shape
    if bounds_or_shp is not None:
        if not watershed_workflow.crs.equal(in_crs, out_crs):
            bounds_or_shp = watershed_workflow.warp.shply(bounds_or_shp, in_crs, out_crs)
        num_reaches = len(reaches)
        reaches = watershed_workflow.utils.filter_to_shape(bounds_or_shp, reaches, tol)
        count = num_reaches - len(reaches)
        logging.info(f"Removed {count} of {num_reaches} reaches not in shape")
        
    if presimplify != None:
        logging.info("Pre-simplifying")
        # convert to shapely and simplify
        reaches_s = [r.simplify(presimplify) for r in reaches]
        for r1, r2 in zip(reaches, reaches_s):
            r2.properties = r1.properties
        reaches = reaches_s

    if merge:
        logging.info("Merging (warning: this loses properties)")
        reaches = list(shapely.ops.linemerge(shapely.geometry.MultiLineString(reaches)))
        
    # round
    if digits != None:
        logging.info("Rounding coordinates")
        watershed_workflow.utils.round_shapes(reaches, digits)

    # not too long
    if long != None:
        logging.info(f"Filtering extra-long reaches")
        n_r = len(reaches_s)
        reaches_s = [reach for reach in reaches_s if reach.length < long]
        logging.info("... filtered {} of {} due to length criteria {}".format(n_r - len(reaches_s), n_r, long))
        
    return out_crs, reaches


def get_waterbodies(source, huc, bounds_or_shp=None, in_crs=None, out_crs=None, digits=None,
                    tol=None, prune_by_area=None, **kwargs):
    """Get waterbodies from NHDPlus hydrography source within a given HUC and/or bounding box.

    Collects waterbody datasets within a HUC and/or a bounding box.  If bounds are
    provided, a containing HUC must still be provided to give a hint for file
    downloads.  If bounds are not provided, then all bodies that intersect the
    HUC are included.

    If bounds is provided, crs must not be None and is the crs of the bounding box.

    Parameters
    ----------
    source : source-type
        Source object providing a get_hydro() method.
    huc : str
        HUC containing reaches.  If bounds are provided, a hint to help the
        source find the file containing the bounds.  For NHD, this is a HUC4 or
        smaller.
    bounds_or_shp : [xmin, ymin, xmax, ymax] bounds or shply, optional
        Bounding box or shapely shape to filter the river network.
    in_crs : crs-type, optional
        Coordinate system of the bounds.
    out_crs : crs-type, optional
        Coordinate system of the output reaches.  Default is the
        native crs of the source.
    digits : int, optional
        Number of digits to round coordinates to.
    tol : float, optional
        Tolerance used in filtering the reaches to the provided shape.
    prune_by_area : float, optional
        If provided, remove bodies whose total area is
        less than this tol.
    **kwargs : dict, optional
        Other arguments are passed to the file manager's get_waterbodies() method.

    Returns
    -------
    out_crs : crs-type
        Coordinate system of `out`.
    out : list(LineString)
        Waterbodies in the HUC and/or intersecting the bounds.

    """
    logging.info("")
    logging.info("Loading Water Bodies")
    logging.info("-"*30)
    logging.info(f"Loading waterbodies in HUC {huc}")

    # get the wbs
    if not type(bounds_or_shp) is tuple:
        bounds = bounds_or_shp.bounds
    else:
        bounds = bounds_or_shp
        bounds_or_shp = None

    logging.info(f"         and/or bounds {bounds}")
    profile, bodies = source.get_waterbodies(huc, bounds, in_crs, **kwargs)
    logging.info(f"... found {len(bodies)} waterbodies")
  
    # convert to shapely
    logging.info("Converting to shapely")
    bodies = [watershed_workflow.utils.shply(b) for b in bodies]
        
    # convert to destination crs
    native_crs = watershed_workflow.crs.from_fiona(profile['crs'])
    if out_crs and not watershed_workflow.crs.equal(out_crs, native_crs):
        logging.info("Converting to out_crs")
        bodies = watershed_workflow.warp.shplys(bodies, native_crs, out_crs)
    else:
        out_crs = native_crs

    # filter to shape
    if bounds_or_shp is not None:
        if not watershed_workflow.crs.equal(in_crs, out_crs):
            bounds_or_shp = watershed_workflow.warp.shply(bounds_or_shp, in_crs, out_crs)
        num_bodies = len(bodies)
        bodies = watershed_workflow.utils.filter_to_shape(bounds_or_shp, bodies, tol)
        count = num_bodies - len(bodies)
        logging.info(f"Removed {count} of {num_bodies} water bodies not in shape")

    if prune_by_area is not None:
        num_bodies = len(bodies)
        bodies = [body for body in bodies if body.area > prune_by_area]
        count = num_bodies - len(bodies)
        logging.info(f"Pruned {count} of {num_bodies} water bodies")

    return out_crs, bodies


def get_raster_on_shape(source, shape, in_crs, out_crs=None,
                        buffer=0, mask=False, nodata=None, **kwargs):
    """Collects a raster that covers the requested shape.

    Parameters
    ----------
    source : str or source-type
        Filename to parse, or a source object providing the get_raster()
        method.
    shape : polygon
        shapely or fiona polygon on which to get the raster.
    in_crs : crs-type
        CRS of shape.
    out_crs : crs-type, optional
        CRS of the raster.  Defaults to the source's CRS.

    buffer : double, optional
        Size of a buffer, in units of the shape's CRS, added to shape to ensure
        pixels cover the entire shape.  Default is 0.
    mask : bool, optional=False
        If True, mask the raster outside of shape.
    nodata : dtype, optional=raster nodata
        Value to place outside of shape.
    kwargs
        All extra arguments are passed to the source's get_raster() method.

    Returns
    -------
    profile : dict
        Rasterio profile of the image including rasterio CRS and transform
    raster : ndarray
        The raster data in a 2D-array.

    """
    logging.info("")
    logging.info("Loading Raster")
    logging.info("-"*30)

    # load file
    if type(source) is str:
        logging.info(f"Loading file: '{source}'")
        source = watershed_workflow.sources.manager_raster.FileManagerRaster(source)

    if type(shape) is dict:
        shape = watershed_workflow.utils.shply(shape)
    if type(shape) is shapely.geometry.MultiPolygon:
        shape = shapely.ops.unary_union(shape)
    shape_original = shape
    shape = shape.buffer(buffer)

    logging.info("Collecting raster")
    profile, raster = source.get_raster(shape, in_crs, **kwargs)
    logging.info(f"... got raster of shape: {raster.shape}")

    # warp the raster to the requested output
    if out_crs:
        logging.info("Warping to out_crs")
        profile, raster = watershed_workflow.warp.raster(profile, raster, out_crs)

    out_crs = watershed_workflow.crs.from_rasterio(profile['crs'])
    
    if mask:
        # mask the raster
        logging.info("Masking to shape")
        if out_crs != in_crs:
            shape = watershed_workflow.warp.shply(shape, in_crs, out_crs)
        logging.info("  shape bounds: {}".format(shape.bounds))
        mask = rasterio.features.geometry_mask([shape,], raster.shape,
                                               profile['transform'], invert=True)
        if nodata is None:
            if profile['nodata'] is None:
                try:
                    # surely there is a better way to see if dtype can handle nan?
                    nodata = np.array([np.nan,], dtype=np.dtype(profile['dtype']))[0]
                except:
                    # surely there is a better way to get -1 as dtype?
                    nodata = np.array([-1,], dtype=np.dtype(profile['dtype']))[0]
                    profile['nodata'] = nodata
                else:
                    profile['nodata'] = nodata
            else:
                nodata = profile['nodata']
            
        logging.info(f"  casting mask of dtype: {profile['dtype']} to: {nodata}")
        raster[~mask] = nodata

    transform = profile['transform']
    x0 = transform * (0,0)
    x1 = transform * (profile['width'], profile['height'])
    logging.info("... got raster bounds: {}".format((x0[0], x0[1], x1[0], x1[1])))
    return profile, raster

#
# functions for relating objects
# -----------------------------------------------------------------------------
def find_huc(source, shape, in_crs, hint, shrink_factor=1.e-5):
    """Finds the smallest HUC containing shp.

    Parameters
    ----------
    source : source-type
        Source object providing a get_hucs() method.
    shape : Polygon
        Shapely or fiona polygon on which to get the raster.
    in_crs : crs-type
        CRS of shape.
    hint : str
        HUC in which to start searching.  This should be at least as long as
        the indexing file size -- HUC 2 or longer for WBD, 4 or longer for NHD
        Plus, or 8 or longer for NHD.
    shrink_factor : float, optional
        A fraction of the radius of shape to shrink prior for checking
        containment within HUCs.  This fixes cases where shape is on a HUC
        boundary with potentially some numerical error.

    Returns
    ------- 
    out : str
        The smallest containing HUC.

    """
    def _in_huc(shply, huc_shply):
        """Checks whether shp is in HUC"""
        if huc_shply.contains(shply):
            return 2
        elif huc_shply.intersects(shply):
            return 1
        else:
            return 0

    def _find_huc(source, shply, crs, hint):
        """Searches in hint to find shp."""
        logging.debug('searching: %s'%hint)
        hint_level = len(hint)
        search_level = hint_level + 2
        if search_level > source.lowest_level:
            return hint

        _, subhus = get_hucs(source, hint, search_level, crs)
        
        for subhu in subhus:
            inhuc = _in_huc(shply, subhu)

            if inhuc == 2:
                # fully contained in try_huc, recurse
                hname = watershed_workflow.sources.utils.get_code(subhu, search_level)
                logging.debug('  subhuc: %s contains'%hname)
                return _find_huc(source, shply, crs, hname)
            elif inhuc == 1:
                hname = watershed_workflow.sources.utils.get_code(subhu, search_level)
                logging.debug('  subhuc: %s partially contains'%hname)
                # partially contained in try_huc, return this
                return hint
            else:
                hname = watershed_workflow.sources.utils.get_code(subhu, search_level)
                logging.debug('  subhuc: %s does not contain'%hname)
        assert(False)

    if type(shape) is shapely.geometry.Polygon:
        shply = shape
    else:
        shply = watershed_workflow.utils.shply(shape)

    # must shrink the poly a bit in case it is close to or on a boundary
    radius = np.sqrt(shply.area/np.pi)
    shply_s = shply.buffer(-shrink_factor*radius)

    hint = watershed_workflow.sources.utils.huc_str(hint)

    _, hint_hu = get_huc(source, hint, in_crs)
    inhuc = _in_huc(shply_s, hint_hu)
    if inhuc != 2:
        raise RuntimeError("{}: shape not found in hinted HUC '{}'".format(source.name, hint))

    result = _find_huc(source, shply_s, in_crs, hint)
    return result


#
# functions for manipulating objects
# -----------------------------------------------------------------------------
def construct_rivers(hucs, reaches, method='geometry',
                     ignore_small_rivers=None,
                     prune_by_area=None, prune_by_area_fraction=None,
                     remove_diversions=False, remove_braided_divergences=False):
    """Create a river, which is a tree of reaches.

    Note, HUCs and rivers must be in the same crs.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object containing all reaches.
    reaches : list(LineString)
        A list of reaches.
    method : str, optional='geometry'
        Provide the method for constructing rivers.  Valid are:
        - 'geometry' looks at coincident coordinates
        - 'hydroseq' Valid only for NHDPlus data, this uses the
          NHDPlus VAA tables Hydrologic Sequence.  If using this
          method, get_reaches() must have been called with both
          'HydrologicSequence' and 'DownstreamMainPathHydroSeq'
          properties requested (or properties=True).
    ignore_small_rivers : int, optional
        If provided, removes rivers whose number of reaches is less
        than this value.
    prune_by_area : float, optional
        If provided, remove reaches whose total contributing area is
        less than this tol.  NOTE: only valid for reaches that include
        a contributing area property (e.g. NHDPlus).
    prune_by_area_fraction : float, optional
        If provided, remove reaches whose total contributing area, as
        a fraction of the area of hucs, is less than this tol.  NOTE:
        only valid for reaches that include a contributing area
        property (e.g. NHDPlus).
    remove_diversions : bool, optional=False
        If true, remove diversions (see documentation of
        modify_rivers_remove_divergences()).
    remove_braided_divergences : bool, optional=False
        If true, remove braided divergences (see documentation of
        modify_rivers_remove_divergences()).

    Returns
    ------- 
    out : list(River)
        A list of rivers, as River objects.
    """
    logging.info("")
    logging.info("Constructing river network")
    logging.info("-"*30)

    logging.info("Generating the river tree")
    rivers = watershed_workflow.hydrography.make_global_tree(reaches, method=method)
    logging.info(f" ... generated {len(rivers)} rivers")

    if ignore_small_rivers is not None:
        rivers = watershed_workflow.hydrography.filter_small_rivers(rivers, ignore_small_rivers)
        if len(rivers) == 0:
            return rivers

    if prune_by_area is not None:
        rivers = watershed_workflow.hydrography.prune_by_contributing_area(rivers, prune_by_area)
        if len(rivers) == 0:
            return rivers

    if prune_by_area_fraction is not None:
        rivers = watershed_workflow.hydrography.prune_by_fractional_contributing_area(
            rivers, prune_by_area_fraction)
        if len(rivers) == 0:
            return rivers
        
    if remove_diversions or remove_braided_divergences:
        rivers = watershed_workflow.hydrography.remove_divergences(
            rivers, remove_diversions, remove_braided_divergences)
    return rivers


def simplify(hucs,
             rivers,
             waterbodies=None,
             simplify_hucs=0,
             simplify_rivers=None,
             simplify_waterbodies=None,
             prune_tol=None,
             merge_tol=None,
             snap=False,
             cut_intersections=False):
    """Simplifies the HUC and river shapes.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object containing all reaches.
    rivers : list(River)
        A list of river objects.
    waterbodies : list(shply), optional
        A list of waterbodies.
    simplify_hucs : float, optional
        If provided, simply the hucs by moving points at most this
        many units (see also shapely.simplify).  Units are that of the
        CRS of shapes.
    simplify_rivers : float, optional
        If provided, simply the rivers by moving points at most this
        many units (see also shapely.simplify).  Units are that of the
        CRS of shapes.  If not provided, use the simplify_hucs value.
        Provide 0 to make no changes to the rivers.
    simplify_waterbodies : float, optional
        Simplify the waterbodies.  If not provided, uses the
        simplify_hucs value.  Provide 0 to make no changes to the
        rivers.
    prune_tol : float, optional
        Prune leaf reaches that are smaller than this tolerance.  If
        not provided, uses simplify_rivers value.  Provide 0 to not do
        this step.
    merge_tol : float, optional
        Merges reaches that are smaller than this tolerance with their
        downstream parent reach.  Note that if there is a branchpoint
        at the downstream node of the small reach, it will get moved
        to the upstream node.  If not provided, uses simplify_rivers
        value.  Provide 0 to not do this step.
    snap : bool, optional
        If true, snaps river and HUC nodes that are nearly coincident
        to be discretely coincident.
    cut_intersections : bool, optional
        If true, force intersections of the river network and the HUC
        boundary to occur at a coincident node by adding nodes as
        needed.

    Returns
    -------
    rivers : list(River)
       Snap may change the rivers, so this returns the list of updated
       rivers.

    .. note: 
        This also may modify the hucs and waterbody objects in-place.

    """
    assert(type(hucs) is watershed_workflow.split_hucs.SplitHUCs)
    assert(type(rivers) is list)
    assert(all(type(r) is watershed_workflow.river_tree.River for r in rivers))

    if simplify_rivers is None:
        simplify_rivers = simplify_hucs
    if simplify_waterbodies is None:
        simplify_waterbodies = simplify_hucs
    if prune_tol is None:
        prune_tol = simplify_rivers
    if merge_tol is None:
        merge_tol = simplify_rivers
    
    logging.info("")
    logging.info("Simplifying")
    logging.info("-"*30)
            
    if simplify_rivers > 0:
        logging.info("Simplifying rivers")
        watershed_workflow.hydrography.cleanup(rivers, simplify_rivers, prune_tol, merge_tol)

    if simplify_hucs > 0:
        logging.info("Simplifying HUCs")
        watershed_workflow.split_hucs.simplify(hucs, simplify_hucs)

    if simplify_waterbodies > 0 and waterbodies is not None:
        for i,wb in enumerate(waterbodies):
            waterbodies[i] = wb.simplify(simplify_waterbodies)

    if snap:
        logging.info("Snapping river and HUC (nearly) coincident nodes")
        rivers = watershed_workflow.hydrography.snap(hucs, rivers, simplify_rivers,
                                            3*simplify_rivers, cut_intersections)
    elif cut_intersections:
        logging.info("Cutting crossings and removing external segments")
        watershed_workflow.hydrography.cut_and_snap_crossings(hucs, rivers, simplify_rivers)
    
    logging.info("")
    logging.info("Simplification Diagnostics")
    logging.info("-"*30)
    if len(rivers) != 0:
        mins = []
        for river in rivers:
            for line in river.dfs():
                coords = np.array(line.coords[:])
                dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
                mins.append(np.min(dz))
        logging.info(f"  river min seg length: {min(mins)}")
        logging.info(f"  river median seg length: {np.median(np.array(mins))}")

    mins = []
    watershed_workflow.split_hucs.simplify(hucs, 0)
    for line in hucs.segments:
        coords = np.array(line.coords[:])
        dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
        mins.append(np.min(dz))
    logging.info(f"  HUC min seg length: {min(mins)}")
    logging.info(f"  HUC median seg length: {np.median(np.array(mins))}")
    return rivers


def simplify_and_prune(hucs, reaches,
                       simplify_hucs=None,
                       simplify_rivers=None,
                       ignore_small_rivers=None,
                       prune_by_area=0,
                       prune_by_area_fraction=0,
                       snap=False, cut_intersections=False):
    """DEPRECATED: simply calls construct_rivers() and simplify()"""
    rivers = construct_rivers(hucs, reaches, ignore_small_rivers, simplify_hucs,
                              prune_by_area, prune_by_area_fraction)
    simplify(hucs, rivers, simplify_hucs, simplify_rivers, snap, cut_intersections)
    return rivers

    
def triangulate(hucs, rivers,
                mesh_rivers=False, diagnostics=True, stream_outlet_width=None, verbosity=1, tol=1,
                refine_max_area=None, refine_distance=None, refine_max_edge_length=None,
                refine_min_angle=None, enforce_delaunay=False, river_region_dist=None):
    """Triangulates HUCs and rivers.

    Note, refinement of a given triangle is done if any of the provided
    criteria is met.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs()
    reaches : list(LineString)
        A list of reaches from, e.g., get_reaches()
    mesh_rivers : bool, optional
        Include stream network in the mesh discretely.
    diagnostics : bool, optional
        Plot diagnostics graphs of the triangle refinement.
    stream_outlet_width : float, optional
        If provided, adds edge sets within the provided distance from
        all outlets, used to track discharge observation regions.
    tol : float, optional
        Set tolerance for minimum distance between two nodes. The unit is the same as 
        that of the watershed's CRS. The default is 1.
    refine_max_area : float, optional
        Refine a triangle if its area is greater than this area.
    refine_distance : list(float), optional
        Refine a triangle if its area is greater than a function of its
        centroid's distance from the nearest point on the river network.  The
        argument is given by:

        [near_distance, near_area, far_distance, far_area]

        Defining d as the distance from triangle centroid to the nearest point
        on the river network and area as the area of the triangle in question,
        refinement occurs if:

        * d < near_distance and area > near_area
        * d > far_distance and area > far_area
        * otherwise, defining 
          d' = (d - near_distance) / (far_distance - near_distance),
          refining occurs if
          area > near_area + (far_area - near_area) * d'

        Effectively this simply writes a piecewise linear function of triangle
        distance from centroid and uses that as a max area criteria.
    refine_max_edge_length : float, optional
        Refine a triangle if its max edge length is greater than this length.
    refine_min_angle : float, optional
        Try to ensure that all triangles have a minimum edge length greater
        than this value.
    enforce_delaunay : bool,optional, experimental
        Attempt to ensure all triangles are proper Delaunay triangles.

        .. note:
            This requires a hacked version of meshpy.triangle that
            supports this option.  See the patch available at
            workflow_tpls/meshpy_triangle.patch
    river_region_dist: float, optional
        Create river region based on the distance from river networks. This is useful if explicit 
        representation of riverbed is desired. Default is None.

    Returns
    -------
    vertices : np.array((n_points, 2), 'd')
        Array of triangle vertices.
    triangles : np.array((n_tris, 3), 'i')
        For each triangle, a list of 3 indices into the vertex array that make
        up that triangle.
    areas : _only if diagnostics=True_, np.array((n_tris), 'd')
        Array of triangle areas.

    """
    verbose = verbosity > 2
    
    logging.info("")
    logging.info("Meshing")
    logging.info("-"*30)

    refine_funcs = []
    if refine_max_area != None:
        refine_funcs.append(watershed_workflow.triangulation.refine_from_max_area(refine_max_area))
    if refine_distance != None:
        refine_funcs.append(watershed_workflow.triangulation.refine_from_river_distance(*refine_distance, rivers))
    if refine_max_edge_length != None:
        refine_funcs.append(watershed_workflow.triangulation.refine_from_max_edge_length(refine_max_edge_length))
    def my_refine_func(*args):
        return any(rf(*args) for rf in refine_funcs)        

    if mesh_rivers:
        rivers_tri = rivers
    else:
        rivers_tri = None
    vertices, triangles = watershed_workflow.triangulation.triangulate(hucs, rivers_tri,
                                                             tol=tol, verbose=verbose,
                                                             refinement_func=my_refine_func,
                                                             min_angle=refine_min_angle,
                                                             enforce_delaunay=enforce_delaunay)

    if diagnostics or river_region_dist is not None:
        logging.info("Plotting triangulation diagnostics")
        river_multiline = shapely.geometry.MultiLineString([r for river in rivers for r in river])
        distances = []
        areas = []
        needs_refine = []
        for tri in triangles:
            verts = vertices[tri]
            bary = np.sum(np.array(verts), axis=0)/3
            bary_p = shapely.geometry.Point(bary[0], bary[1])
            distances.append(bary_p.distance(river_multiline))
            areas.append(watershed_workflow.utils.triangle_area(verts))
            needs_refine.append(my_refine_func(verts, areas[-1]))
        areas = np.array(areas)
        distances = np.array(distances)
        logging.info("  min area = {}".format(areas.min()))
        logging.info("  max area = {}".format(areas.max()))

        if verbosity > 0:
            plt.figure()
            plt.subplot(121)
            plt.hist(distances)
            plt.xlabel("distance from river of triangle centroids [m]")
            plt.ylabel("count [-]")
            plt.subplot(122)
            plt.scatter(distances, areas,c=needs_refine,marker='x')
            plt.xlabel("distance [m]")
            plt.ylabel("triangle area [m^2]")

        if river_region_dist is not None:
            river_idx = distances < river_region_dist
            river_tris = triangles[river_idx]

            plt.figure()
            plt.tripcolor(vertices[:,0], vertices[:,1], triangles, 
                            facecolors= np.array([0]*len(triangles)), 
                            cmap = None, edgecolors = 'w', linewidth=0.01)
            plt.tripcolor(vertices[:,0], vertices[:,1], river_tris, 
                            facecolors= np.array([1]*len(river_tris)), 
                            cmap = 'jet', edgecolors = 'w', linewidth=0.1)
            plt.title("river region")
      
            return vertices, triangles, areas, distances, river_idx
        return vertices, triangles, areas, distances
            
    return vertices, triangles

def elevate(mesh_points, mesh_crs, dem, dem_profile, algorithm='piecewise bilinear'):
    """Elevate mesh_points onto the provided dem.

    Parameters
    ----------
    mesh_points : np.array((n_points, 2), 'd')
        Array of triangle vertices.
    mesh_crs : crs-type
        Mesh coordinate system.
    dem : np.array
        2D array forming an elevation raster.
    dem_profile : dict
        rasterio profile for the elevation raster.
    algorithm : str, optional
        Algorithm used for interpolation.  One of:
        * "nearest" for nearest-neighbor pixels
        * "piecewise bilinear" for interpolation (default)

    Returns
    -------
    out : np.array((n_points, 3), 'd')
        Array of triangle vertices, including a z-dimension.

    """
    logging.info("")
    logging.info("Elevating Triangulation to DEM")
    logging.info("-"*30)

    # index the i,j of the points, pick the elevations
    elev = values_from_raster(mesh_points, mesh_crs, dem, dem_profile, algorithm)

    # create the 3D points
    out = np.zeros((len(mesh_points),3),'d')
    out[:,0:2] = mesh_points
    out[:,2] = elev
    return out


def values_from_raster(points, points_crs, raster, raster_profile, algorithm='nearest'):
    """Interpolate a raster onto a collection of unstructured points.

    Parameters
    ----------
    points : np.array((n_points, 2), 'd')
        Array of points to interpolate onto.
    points_crs : crs-type
        Coordinate system of the points.
    raster : np.array
        2D array forming the raster.
    raster_profile : dict
        rasterio profile for the raster.
    algorithm : str, optional
        Algorithm used for interpolation.  One of:
        * "nearest" for nearest neighbor pixels (default)
        * "piecewise bilinear" for interpolation

    Returns
    -------
    out : np.array((n_points,))
        Array of raster values interpolated onto the points.

    """
    raster_crs = watershed_workflow.crs.from_rasterio(raster_profile['crs'])
    points_raster_crs = np.array(watershed_workflow.warp.xy(points[:,0], points[:,1], points_crs, raster_crs)).transpose()
    if algorithm == 'nearest':
        out = raster[rasterio.transform.rowcol(raster_profile['transform'], points_raster_crs[:,0], points_raster_crs[:,1])]
    elif algorithm == 'piecewise bilinear':
        eps = 1.e-10
        
        # get the index of the point
        invtransform = ~raster_profile['transform']
        mybox = np.zeros((2,2),'d')
        out = np.zeros((len(points),),'d')
        for k,xy in enumerate(points_raster_crs):
            xy = tuple(xy)
            j,i = invtransform * xy

            # center on pixel
            i -= 0.5
            j -= 0.5
            
            i = max(eps, min(raster_profile['height']-1-eps, i))
            j = max(eps, min(raster_profile['width']-1-eps, j))

            mybox[0,0] = raster[math.floor(i), math.floor(j)]
            mybox[0,1] = raster[math.floor(i), math.ceil(j)]
            mybox[1,0] = raster[math.ceil(i), math.floor(j)]
            mybox[1,1] = raster[math.ceil(i), math.ceil(j)]
            ii = i%1
            jj = j%1

            up = mybox[0,0] + jj * (mybox[0,1] - mybox[0,0])
            dn = mybox[1,0] + jj * (mybox[1,1] - mybox[1,0])
            out[k] = up + (dn - up) * ii
    else:
        raise ValueError(f'Invalid algorithm "{algorithm}", valid are "nearest" and "piecewise bilinear"')
    return out
    

def color_raster_from_shapes(shapes, shapes_crs, shape_colors,
                             raster_bounds, raster_dx, raster_crs=None, nodata=None):
    """Color in a raster by filling in a collection of shapes.

    Given a canvas specified by bounds and pixel size, color a raster by, for
    each shape, finding the intersection of that shape with the canvas and
    coloring it by a provided value.  Paint by numbers.

    Note, if the shapes overlap, the last shape containing a pixel gives the
    color of that pixel.

    Parameters
    ----------
    shapes : list(Polygon)
        Collection of shapes (likely) overlapping the canvas.
    shapes_crs : crs-type
        Coordinate system of the shapes.
    shapes_colors : iterable[]
        Color to label the interior of each polygon with.
    raster_bounds : [xmin, ymin, xmax, ymax]
        Bounding box for the output raster, in the given CRS.
    raster_dx : float
        Pixel size (assumed the same in both x and y).
    raster_crs : crs-type, optional=shapes_crs
        Coordinate system of the raster.
    nodata : dtype, optional={-1 (int), nan (float)}
        Value to place in pixels which intersect no shape.  Note the type of
        this should be the same as the type of shape_colors.

    Returns
    -------
    out_profile : dict
        rasterio profile of the color raster.
    out : np.array(raster_bounds, dtype)
        Raster of colors.

    """
    assert(len(shapes) == len(shape_colors))
    if len(shapes) == 0:
        raise ValueError("Cannot generate raster for empty set of shapes")

    logging.info('Coloring shapes onto raster:')

    if not watershed_workflow.crs.equal(shapes_crs, raster_crs):
        shapes = watershed_workflow.warp.shplys(shapes, shapes_crs, raster_crs)
   
    dtype = np.dtype(type(shape_colors[0]))

    if nodata is None:
        try:
            nodata = dtype(np.nan)
        except ValueError:
            nodata = dtype(-1)

    raster_profile, raster = watershed_workflow.utils.create_empty_raster(raster_bounds, raster_crs, raster_dx, dtype, nodata)
    logging.info(f'  and {len(set(shape_colors))} independent colors')

    for p, p_id in zip(shapes, shape_colors):
        if not p.is_empty:
            p_list = watershed_workflow.utils.flatten([p,])
            mask = rasterio.features.geometry_mask(p_list, raster.shape, raster_profile['transform'], invert=True)
            raster[mask] = p_id
    return raster_profile, raster


def color_existing_raster_from_shapes(shapes, shapes_crs, shape_colors,
                                      raster, raster_profile):
    """Color in a raster by filling in a collection of shapes.

    Given a canvas, find the intersection of that shape with the canvas and
    coloring it by a provided value.  Paint by numbers.

    Note, if the shapes overlap, the last shape containing a pixel gives the
    color of that pixel.

    Parameters
    ----------
    shapes : list(Polygon)
        Collection of shapes (likely) overlapping the canvas.
    shapes_crs : crs-type
        Coordinate system of the shapes.
    shapes_colors : iterable[]
        Color to label the interior of each polygon with.
    raster : np.ndarray
        The canvas to color on.
    raster_profile : dict
        Rasterio style profile including at least CRS, nodata, and
        transform.

    """
    assert(len(shapes) == len(shape_colors))
    if len(shapes) == 0:
        raise ValueError("Cannot generate raster for empty set of shapes")

    logging.info('Coloring shapes onto raster:')
    logging.info(f'  and {len(set(shape_colors))} independent colors')

    if not watershed_workflow.crs.equal(shapes_crs, raster_profile['crs']):
        shapes = watershed_workflow.warp.shplys(shapes, shapes_crs, raster_profile['crs'])
    
    for p, p_id in zip(shapes, shape_colors):
        if not p.is_empty:
            p_list = watershed_workflow.utils.flatten([p,])
            mask = rasterio.features.geometry_mask(p_list, raster.shape, raster_profile['transform'], invert=True)
            raster[mask] = p_id

