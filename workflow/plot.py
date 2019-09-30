import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as pltc
import shapely
import rasterio
import descartes
from mpl_toolkits.mplot3d import Axes3D


import workflow.utils
import workflow.crs

def get_ax(crs=None, fig=None, nrow=1, ncol=1, index=1, window=None, **kwargs):
    """Returns an axis with a projection."""
    # make a figure
    if fig is None:
        fig = plt.figure(**kwargs)
        newfig = True
    else:
        newfig = False

    if window is None:
        if crs is None:
            # no crs, just get an ax -- you deal with it.
            ax = fig.add_subplot(nrow, ncol, index)
        elif crs == '3d':
            # 3d plot
            ax = fig.add_subplot(nrow, ncol, index, projection='3d')
        else:
            projection = workflow.crs.to_cartopy(crs)
            ax = fig.add_subplot(nrow, ncol, index, projection=projection)

    else:
        if crs is None:
            fig.add_axes(window=window)
            ax = fig.gca()

        elif crs == '3d':
            ax = Axes3D(fig, rect=window)

        else:
            projection = workflow.crs.to_cartopy(crs)
            fig.add_axes(window=window, projection=projection)
            ax = fig.gca()

    if newfig:
        return fig, ax
    else:
        return ax
        
def huc(huc, crs, color='k', ax=None, **kwargs):
    """Plot a HUC polygon (simply calls plot.shply())"""
    return shply([huc,], crs, color, ax, **kwargs)

def hucs(hucs, crs, color='k', ax=None, **kwargs):
    """Plot a SplitHUCs object, a wrapper for plot.shply()"""
    ps = list(hucs.polygons())
    return shply(ps, crs, color, ax, **kwargs)

def shapes(shps, crs, color='k', ax=None, **kwargs):
    """Plot an itereable collection of fiona shapes."""
    shplys = [workflow.utils.shply(shp) for shp in shps]
    shply(shplys, crs, color, ax, **kwargs)

def river(river, crs, color='b', ax=None, **kwargs):
    """Plot an itereable collection of reaches (LineStrings)."""
    shply(river, crs, color, ax, **kwargs)

def rivers(rivers, crs, color='b', ax=None,  **kwargs):
    """Plot an itereable collection of river Tree objects."""
    if type(rivers) is shapely.geometry.MultiLineString:
        return river(rivers, crs, color, ax, **kwargs)
    
    if type(color) is not str and len(color) == len(rivers):
        for r, c in zip(rivers, color):
            river(r, crs, c, ax, **kwargs)
    else:
        for r in rivers:
            river(r, crs, color, ax, **kwargs)
            
    
def shply(shps, crs, color=None, ax=None, style='-', **kwargs):
    """Plot shapely objects.

    Currently this assumes shps is an iterable collection of Points,
    Lines, or Polygons.  So while a single MultiPolygon is allowed,
    lists of MultiPolygons are not currently supported.  And
    heterogeneous collections are not supported.
    """
    if len(shps) is 0:
        return
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'none'

    if ax is None:
        ax = get_ax(crs)

    if not hasattr(ax, 'projection') or crs is None:
        projection = None
    else:
        projection = workflow.crs.to_cartopy(crs)
        print('got projection:', projection)
        
    if type(next(iter(shps))) is shapely.geometry.Point:
        # plot points
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'
        
        points = np.array([p.coords for p in shps])[:,0,:]
        if projection is None:
            res = ax.scatter(points[:,0], points[:,1], c=color, **kwargs)
        else:
            res = ax.scatter(points[:,0], points[:,1], c=color, transform=projection, **kwargs)
            
            
    elif type(next(iter(shps))) is shapely.geometry.LineString:
        # plot lines
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = style
        if 'colors' not in kwargs:
            kwargs['colors'] = color
        
        lines = [np.array(l.coords) for l in shps]
        lc = pltc.LineCollection(lines, **kwargs)
        if projection is not None:
            lc.set_transform(projection)

        if type(ax) is Axes3D:
            res = ax.add_collection3d(lc)
        else:
            res = ax.add_collection(lc)
            ax.autoscale()
        
    elif type(next(iter(shps))) is shapely.geometry.Polygon:
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = style

        try:
            color_len = len(color)
        except (AttributeError,TypeError):
            color_len = -1

        if type(color) is str or color_len != len(shps):
            # assume this is ONE color, and therefore can add as a multipolygon/polygon collection
            if 'edgecolor' not in kwargs:
                kwargs['edgecolor'] = color
            
            multi_poly = shapely.geometry.MultiPolygon(shps)
            patch = descartes.PolygonPatch(multi_poly, **kwargs)
            if projection is not None:
                patch.set_transform(projection)

            if type(ax) is Axes3D:
                res = ax.add_collection3d(patch)
            else:
                res = ax.add_patch(patch)

        else:
            # add polygons independently
            if color is None:
                res = []
                for shp in shps:
                    patch = descartes.PolygonPatch(shp, **kwargs)
                    # if projection is not None:
                    #     patch.set_transform(projection)
                    res.append(ax.add_patch(patch))
            else:
                res = []
                for c, shp in zip(color, shps):
                    patch = descartes.PolygonPatch(shp, edgecolor=c, **kwargs)
                    # if projection is not None:
                    #     patch.set_transform(projection)
                    res.append(ax.add_patch(patch))
        ax.autoscale()
    else:
        raise TypeError('Unknown shply type: {}'.format(type(next(iter(shps)))))
    #assert res is not None
    return res

def triangulation(points, tris, crs, color='gray', ax=None, **kwargs):
    """Plots a triangulation"""
    if ax is None:
        ax = get_ax(crs)
    
    if type(color) is str and color == 'elevation' and points.shape[1] != 3:
        color = 'gray'

    if type(ax) is Axes3D:
        if type(color) is str and color == 'elevation':
            col =  ax.plot_trisurf(points[:,0], points[:,1], points[:,2], tris, points[:,2], **kwargs)
        elif type(color) != str:
            col =  ax.plot_trisurf(points[:,0], points[:,1], points[:,2], tris, color, **kwargs)
        else:        
            col =  ax.plot_trisurf(points[:,0], points[:,1], points[:,2], tris, color=color, **kwargs)
    else:
        if type(color) is str and color == 'elevation':
            col =  ax.tripcolor(points[:,0], points[:,1], tris, points[:,2], **kwargs)
        elif type(color) != str:
            col =  ax.tripcolor(points[:,0], points[:,1], tris, color, **kwargs)
        else:        
            col =  ax.triplot(points[:,0], points[:,1], tris, color=color, **kwargs)
    return col


def dem(profile, data, ax=None, vmin=None, vmax=None, **kwargs):
    """Plots a raster"""
    if ax is None:
        ax = get_ax(profile['crs'])

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    logging.info('BOUNDS: {}'.format(bounds))
    return ax.imshow(data, origin='upper', extent=extent, vmin=vmin, vmax=vmax)


def basemap(crs, ax=None, resolution='50m', land_kwargs=None, ocean_kwargs=None, state_kwargs=None):
    """Add a basemap to an axis."""
    import cartopy.feature

    if ax is None:
        ax = get_ax(crs)

    if land_kwargs is None:
        land_kwargs = dict()
    if 'edgecolor' not in land_kwargs:
        land_kwargs['edgecolor'] = 'face'
    if 'facecolor' not in land_kwargs:
        land_kwargs['facecolor'] = cartopy.feature.COLORS['land']
    land = cartopy.feature.NaturalEarthFeature('physical', 'land', resolution, **land_kwargs)
    ax.add_feature(land)

    if state_kwargs is not None:
        kwargs = {'facecolor':'none', 'edgecolor':'k', 'linewidth':0.5}
        kwargs.update(**state_kwargs)
        states = cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines',
                                                     resolution, **kwargs)
        ax.add_feature(states)
    
    if ocean_kwargs is None:
        ocean_kwargs = dict()
    if 'edgecolor' not in ocean_kwargs:
        ocean_kwargs['edgecolor'] = 'face'
    if 'facecolor' not in ocean_kwargs:
        ocean_kwargs['facecolor'] = cartopy.feature.COLORS['water']
    ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', resolution, **ocean_kwargs)
    ax.add_feature(ocean)

    return 
        
