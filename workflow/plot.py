"""Plotting relies on cartopy to ensure that coordinate projections are dealt
with reasonably within matplotlib.  The preferred usage for plotting is similar
to the non-pylab interface to matplotlib -- first get a figure and axis object,
then call plotting functions passing in that ax object.

Note that we use the descartes package to plot shapely objects, which is a
simple wrapper to write a shapely polygon as a matplotlib patch.

Note that, for complex plots, it can be useful to manage the ordering of the
layers of objects.  In this case, all plotting functions accept matplotlib's
zorder argument, an int which controls the order of drawing, with larger being
later (on top) of smaller values.

.. code:: python
    import workflow.plot as wfp
    fig, ax = wfp.get_ax(crs, figsize=(6,6))
    wfp.hucs(my_hucs, crs, 'r', ax=ax)

"""


import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as pltc
from matplotlib import cm as pcm
import shapely
import rasterio
import descartes
from mpl_toolkits.mplot3d import Axes3D


import workflow.utils
import workflow.crs
import workflow.colors

def get_ax(crs, fig=None, nrow=1, ncol=1, index=1, window=None, **kwargs):
    """Returns an axis with a given projection.
    
    Note this forwards extra kwargs for plt.figure().

    Parameters
    ----------
    crs : CRS object, optional
      A **projected** CRS for the plot.  None can be given for a normal
      matplotlib axis (useful for plotting lat/lon or other non-projected
      coordinate systems.  Note that if you call plotting on an object with an
      unprojected CRS, it will project for you, or change coordinates if
      needed.  This can get a bit dicey, so prefer to plot objects all in the
      same CRS.  Defaults to None.
    fig : matplotlib figure, optional
      If you already have a figure, will create the axis on this figure.
      Defaults to None, at which point a figure will be created.
    nrow, ncol, index : int, optional
      Create a grid of axes of this shape.  Calls
      plt.add_subplot(nrow,ncol,index).  Default is 1.
    window : [xmin, ymin, width, height], optional
      Matplotlib patch arguments for call to fig.add_axes()
    figsize : (width, height), optional
      Figure size in inches.
    dpi : int, optional
      Dots per inch for figures.
    kwargs : dict
      Additional arguments to plt.figure()

    Returns
    -------
    *If fig is provided*
    ax : matplotlib axes object
    
    *If fig is not provided*
    fig : matplotlib figure object
    ax : matplotlib ax object
    """
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
    """Plot a HUC polygon.

    A wrapper for plot.shply()
    
    Parameters
    ----------
    huc : shapely polygon
      An object to plot.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      descartes.PolygonPatch.

    Returns
    -------
    patches : matplotlib PatchCollection
    """
    return shply([huc,], crs, color, ax, **kwargs)

def hucs(hucs, crs, color='k', ax=None, **kwargs):
    """Plot a SplitHUCs object.
    
    A wrapper for plot.shply()
    
    Parameters
    ----------
    hucs : workflow.split_hucs.SplitHucs object
      The collection of hucs to plot.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      descartes.PolygonPatch.

    Returns
    -------
    patches : matplotib PatchCollection
    """
    ps = list(hucs.polygons())
    return shply(ps, crs, color, ax, **kwargs)

def shapes(shps, crs, color='k', ax=None, **kwargs):
    """Plot an itereable collection of fiona shapes.

    A wrapper for plot.shply()

    Parameters
    ----------
    shapes : list(fiona shape objects)
      The collection of fiona shape objects to plot.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      descartes.PolygonPatch.

    Returns
    -------
    patches : matplotib PatchCollection
    """
    shplys = [workflow.utils.shply(shp) for shp in shps]
    shply(shplys, crs, color, ax, **kwargs)

def river(river, crs, color='b', ax=None, **kwargs):
    """Plot an itereable collection of reaches.

    A wrapper for plot.shply()

    Parameters
    ----------
    river : list(shapely.LineString)
      An iterable of shapely LineString reaches.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      matplotlib.collections.LineCollection.

    Returns
    -------
    lines : matplotib LineCollection
    """
    shply(river, crs, color, ax, **kwargs)

def rivers(rivers, crs, color='b', ax=None,  **kwargs):
    """Plot an itereable collection of river Tree objects.

    A wrapper for plot.shply()

    Parameters
    ----------
    rivers : list(river_tree.RiverTree)
      An iterable of river_tree.RiverTree objects.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      matplotlib.collections.LineCollection.

    Returns
    -------
    lines : matplotib LineCollection
    """
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

    Currently this assumes shps is an iterable collection of Points, Lines, or
    Polygons.  So while a single MultiPolygon is allowed, lists of
    MultiPolygons are not currently supported.  These can easily be unraveled.
    
    Heterogeneous collections are not supported.

    Parameters
    ----------
    shps : list(shapely shape objects)
      An iterable of shapely objects to plot.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to the plotting method, which can be:
      * pyplot.scatter() (if shps are Point objects)
      * matplotlib.collections.LineCollection() (if shps are LineStrings)
      * descartes.PolygonPatch() (if shps are Polygons)

    Returns
    -------
    col : collection of matplotlib points or lines or patches
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
        
        lines = [np.array(l.coords)[:,0:2] for l in shps]
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

                if type(color[0]) is not tuple and type(color[0]) is not np.ndarray:
                    # likely just an array -- map using cmap
                    try:
                        cmap = kwargs.pop('cmap')
                    except KeyError:
                        cmap = pcm.viridis

                    try:
                        cmap_norm = kwargs.pop('norm')
                    except KeyError:
                        cmap_norm = None
                        
                    cm = workflow.colors.cm_mapper(min(color), max(color), cmap, cmap_norm)
                    color = np.array([cm(c) for c in color])

                if kwargs['facecolor'] == 'color':
                    kwargs.pop('facecolor')
                    face_is_edge = True
                else:
                    face_is_edge = False
                
                for c, shp in zip(color, shps):
                    if face_is_edge:
                        patch = descartes.PolygonPatch(shp, facecolor=c, **kwargs)
                    else:
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
    """Plots a triangulation.

    A wrapper for matplotlib's plot_trisurf() or tripcolor()

    Parameters
    ----------
    points : np.ndarray(npoints, 3)
      Array of point coordinates, x,y,z.
    tris : list, np.ndarray(ntris, 3)
      List of lists or ndarray of indices into the points array for defining
      the triangle topology.
    crs : CRS object
      Coordinate system of the points.
    color : matplotlib color object or iterable or str, optional
      Either a matplotlib color object (for uniform colors), or a list of color
      objects (length equal to the length of tris), or 'elevation' to color by
      z coordinate.
    ax : matplotlib ax object
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to plot_trisurf() (for 3D axes) or tripcolor()
      (for 2D).

    Returns
    -------
    col : matplotlib collection
      Collection of patches representing the triangles.

    """
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


def mesh(m2, crs, color='gray', ax=None, **kwargs):
    """Plots a workflow.mesh.Mesh2D object.

    Parameters
    ----------
    m2 : Mesh2D
      The 2D mesh to plot.
    crs : CRS object
      Coordinate system of the points.
    color : matplotlib color object or iterable or str, optional
      Either a matplotlib color object (for uniform colors), or a list of color
      objects (length equal to the length of tris), or 'elevation' to color by
      z coordinate.
    ax : matplotlib ax object
      Axes to plot on.  Calls get_ax() if not provided.
    kwargs : dict
      Extra arguments passed to plot_trisurf() (for 3D axes) or tripcolor()
      (for 2D).

    Returns
    -------
    col : matplotlib collection
      Collection of patches representing the triangles.

    """
    shplys = [ shapely.geometry.Polygon(m2.coords[c,:]) for c in m2.conn ]
    return shply(shplys, crs, color, ax, **kwargs)



def dem(profile, data, ax=None, vmin=None, vmax=None, **kwargs):
    """Plots a raster.

    A wrapper for matplotlib imshow()

    Parameters
    ----------
    profile : rasterio profile
      Rasterio profile of the input raster.
    data : np.ndarray
      2D array of data.
    vmin,vmax : float
      Min and max value to limit extent of color values.
    kwargs : dict
      Dictionary of extra arguments passed to imshow().
    
    Returns
    -------
    im : matplotlib image object
      Return value of imshow()
    """
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


def basemap(crs=None, ax=None, resolution='50m', land_kwargs=None, ocean_kwargs=None, state_kwargs=None):
    """Add a basemap to the axis.

    Uses cartopy to add political and natural boundaries and shapes to the axes
    image.

    Parameters
    ----------
    crs : CRS object, optional
      Coordinate system to plot.  May be ignored if ax is provided.
    ax : matplotlib ax object, optional
      Matplotlib axes to plot on.  If not provided, get_ax() is called using
      crs.
    resolution : str
      Resolution of cartopy basemap.  One of '10m', '50m', or '110m'.
    land_kwargs : dict
      Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
      land polygons.
    ocean_kwargs : dict
      Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
      ocean polygons.
    state_kwargs : dict
      Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
      political state boundary polygons.
    """
    import cartopy.feature

    if ax is None:
        ax = get_ax(crs)

    if land_kwargs is not False:
        if land_kwargs is None:
            land_kwargs = dict()
        if 'edgecolor' not in land_kwargs:
            land_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in land_kwargs:
            land_kwargs['facecolor'] = cartopy.feature.COLORS['land']
        land = cartopy.feature.NaturalEarthFeature('physical', 'land', resolution, **land_kwargs)
        ax.add_feature(land)

    if state_kwargs is not None and state_kwargs is not False:
        kwargs = {'facecolor':'none', 'edgecolor':'k', 'linewidth':0.5}
        kwargs.update(**state_kwargs)
        states = cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines',
                                                     resolution, **kwargs)
        ax.add_feature(states)
        
    if ocean_kwargs is not False:
        if ocean_kwargs is None:
            ocean_kwargs = dict()
        if 'edgecolor' not in ocean_kwargs:
            ocean_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in ocean_kwargs:
            ocean_kwargs['facecolor'] = cartopy.feature.COLORS['water']
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', resolution, **ocean_kwargs)
        ax.add_feature(ocean)

    return 
        
