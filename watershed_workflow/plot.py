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
"""

import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as pltc
from matplotlib import cm as pcm
import shapely
import rasterio
from mpl_toolkits.mplot3d import Axes3D

import watershed_workflow.utils
import watershed_workflow.crs
import watershed_workflow.colors


def _is_iter(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


# plot reaches and modify...
#
# This uses the annotated axes
class Labeler:
    """A labeling widget that can be attached to matplotlib figures to display info on-click.

    When an geometry item (e.g. point, line, or polygon) is clicked on
    the figure, that is mapped into the original WW object that
    generated the geometry, and then run through a function to
    generate a label that is written to the title of the figure.

    Parameters
    ----------
    ax : matplotlib.Axes object
        The axes to work with.
    items : list[tuple[artist, metadata, formatter]]
        See documentation of the addItem() method.

    """
    def __init__(self, ax, items=None):
        self.ax = ax
        self.items = []
        for item in items:
            self.addItem(*item)

        self.ax.set_title("None")
        self.selected = None

    def addItem(self, data, artist, formatter):
        """Adds an item to the list of things to label.

        Parameters
        ----------
        data : List[*]
            A list of objects being labeled.  This is likely the
            underlying data, with properties, that was passed to
            a ww.plot function.
        artist : matplotlib collection
            A matplotlib Collection, likely the return value of
            a ww.plot call or similar.
        formatter : function or str
            A function that accepts an entry in data and returns a
            string to label the item selected.  If this is a string,
            it is assumed to be a formattable string to which the
            item's properties dictionary is passed.
        """

        if isinstance(formatter, str):

            def format_this(item):
                return formatter.format(**dict(item)), list()

            formatter = format_this

        assert (len(artist) == len(metadata))
        self.items.append((artist, metadata, formatter))
        self.metadata = metadata
        self._format = format
        self._selected = []

    def deselect(self):
        """Clears anything plotted in the last click"""
        for artist in self._selected:
            artist.clear()
            self._selected = []

    def select(self, i, j, xy):
        """Selects item i, collection index j, with a click at xy"""
        data, artist, formatter = self.items[i]

        if isinstance(data, list):
            dat = data[j]
            if isinstance(dat, shapely.geometry.base.BaseGeometry) and hasattr(dat, 'properties'):
                dat = dict(geometry=dat, **dat.properties)
            title = formatter(dat)
        elif isinstance(data, pandas.DataFrame):
            title = formatter(data.iloc[j])
        self.ax.set_title(title)

        # redraw LineStrings with markers
        if isinstance(artist, matplotlib.collections.LineCollection):
            line = artist.get_data()[i]
            color = artist.get_colors()[i]

            self._selected.append(self.ax.plot(line[:, 0], line[:, 1], '-x', color=color))

    def update(self, event):
        """Acts on click."""
        print('event loc:', event.mouseevent.x, event.mouseevent.y)
        print('event dict:', event.__dict__)

        i = next(i for (i, item) in enumerate(self.items) if item[0] is event.artist)
        self.select(i, 0, (event.mouseevent.x, event.mouseevent.y))
        self.ax.get_figure().canvas.draw_idle()


def get_ax(crs,
           fig=None,
           nrow=1,
           ncol=1,
           index=None,
           window=None,
           axgrid=None,
           ax_kwargs=None,
           **kwargs):
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
    ax_kwargs : dict
      Other arguments provided to the axes creation call.
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
    if ax_kwargs is None:
        ax_kwargs = dict()
    # make a figure
    if fig is None:
        fig = plt.figure(**kwargs)
        newfig = True
    else:
        newfig = False

    if window is None:

        def _get_ax(axargs, ax_kwargs):
            if crs is None:
                # no crs, just get an ax -- you deal with it.
                ax = fig.add_subplot(*axargs, **ax_kwargs)
            elif crs == '3d':
                # 3d plot
                ax_kwargs['projection'] = '3d'
                ax = fig.add_subplot(*axargs, **ax_kwargs)
            else:
                projection = watershed_workflow.crs.to_cartopy(crs)
                ax_kwargs['projection'] = projection
                ax = fig.add_subplot(*axargs, **ax_kwargs)
            return ax

        if axgrid is None:
            if nrow == 1 and ncol == 1:
                index = 1
            if index is None:
                ax = [[_get_ax([nrow, ncol, i * (ncol) + j + 1], ax_kwargs) for j in range(ncol)]
                      for i in range(nrow)]
                if nrow == 1:
                    ax = ax[0]
                elif ncol == 1:
                    ax = [a[0] for a in ax]
            else:
                ax = _get_ax([nrow, ncol, index], ax_kwargs)
        else:
            ax = _get_ax([axgrid, ], ax_kwargs)

    else:
        if crs is None:
            fig.add_axes(window)
            ax = fig.gca()

        elif crs == '3d':
            ax = Axes3D(fig, rect=window)

        else:
            projection = watershed_workflow.crs.to_cartopy(crs)
            fig.add_axes(window, projection=projection)
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
    return shply([huc, ], crs, color, ax, **kwargs)


def hucs(hucs,
         crs,
         color='k',
         ax=None,
         outlet_marker=None,
         outlet_markersize=100,
         outlet_markercolor='green',
         **kwargs):
    """Plot a SplitHUCs object.
    
    A wrapper for plot.shply()
    
    Parameters
    ----------
    hucs : watershed_workflow.split_hucs.SplitHucs object
      The collection of hucs to plot.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    outlet_marker : matplotlib marker string, optional
      If provided, also plots the actual points that make up the shape.
    outlet_markersize : float, optional
      Size of the outlet marker.
    outlet_markercolor : matplotlib color string, optional
      Color of the outlet marker.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      descartes.PolygonPatch.

    Returns
    -------
    patches : matplotib PatchCollection
    """
    ps = list(hucs.polygons())
    polys = shply(ps, crs, color, ax, **kwargs)

    if hucs.polygon_outlets is not None and ax is not None and outlet_marker is not None:
        x = np.array([
            p.xy[0][0] for p in hucs.polygon_outlets
            if not watershed_workflow.utils.is_empty_shapely(p)
        ])
        y = np.array([
            p.xy[1][0] for p in hucs.polygon_outlets
            if not watershed_workflow.utils.is_empty_shapely(p)
        ])
        c = [
            c for (c, p) in zip(outlet_markercolor, hucs.polygon_outlets)
            if not watershed_workflow.utils.is_empty_shapely(p)
        ]
        ax.scatter(x, y, s=outlet_markersize, marker=outlet_marker, c=c)


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
    shplys = [watershed_workflow.utils.create_shply(shp) for shp in shps]
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
    shplys(river, crs, color, ax, **kwargs)


def rivers(rivers, crs, color=None, ax=None, **kwargs):
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
    if color is None:
        color = watershed_workflow.colors.enumerated_colors(len(rivers))

    if type(color) is not str and len(color) == len(rivers):
        for r, c in zip(rivers, color):
            river(r, crs, c, ax, **kwargs)
    else:
        for r in rivers:
            river(r, crs, color, ax, **kwargs)


def shply(shp, *args, **kwargs):
    """Plot a single shapely object.  See shplys() for options."""
    if type(shp) is list:
        return shplys(shp, *args, **kwargs)
    else:
        return shplys([shp, ], *args, **kwargs)


def shplys(shps, crs, color=None, ax=None, marker=None, **kwargs):
    """Plot shapely objects.

    Currently this assumes shps is an iterable collection of Points, Lines, or
    Polygons.  So while a single MultiPolygon is allowed, lists of
    MultiPolygons are not currently supported.  These can easily be unraveled.
    
    Heterogeneous collections are not supported.

    Parameters
    ----------
    shps : shapely shape, list(shapely shape objects), or MultiShape object
      An iterable of shapely objects to plot.
    crs : CRS object
      The coordinate system to plot in.
    color : str, scalar, or array-like, optional
      See https://matplotlib.org/tutorials/colors/colors.html
    ax : matplotib axes object, optional
      Axes to plot on.  Calls get_ax() if not provided.
    marker : matplotlib marker string
      If provided, also plots the actual points that make up the shape.
    kwargs : dict
      Extra arguments passed to the plotting method, which can be:
      * pyplot.scatter() (if shps are Point objects)
      * matplotlib.collections.LineCollection() (if shps are LineStrings)
      * descartes.PolygonPatch() (if shps are Polygons)

    Returns
    -------
    col : collection of matplotlib points or lines or patches
    """
    import descartes

    try:
        if len(shps) == 0:
            return
    except TypeError:
        shps = [shps, ]

    # get an axis and projection to work on
    if ax is None:
        fig, ax = get_ax(crs)
    if not hasattr(ax, 'projection') or crs is None:
        projection = None
    else:
        projection = watershed_workflow.crs.to_cartopy(crs)

    # set default colors
    if color is None:
        color = watershed_workflow.colors.enumerated_colors(len(shps))

    # update keyword arguments
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'none'

    # markers cannot be used in collections, so we scatter them separately
    marker_kwargs = dict()
    if marker is not None:
        marker_kwargs['marker'] = marker
        if 'markersize' in kwargs:
            marker_kwargs['s'] = kwargs.pop('markersize')

    if type(next(iter(shps))) is shapely.geometry.Point:
        # plot points
        marker_kwargs.update(kwargs)
        if 'marker' not in marker_kwargs:
            marker_kwargs['marker'] = 'o'

        points = np.array([p.coords for p in shps])[:, 0, :]
        if projection is None:
            res = ax.scatter(points[:, 0], points[:, 1], c=color, **marker_kwargs)
        else:
            res = ax.scatter(points[:, 0], points[:, 1], c=color, transform=projection, **kwargs)

    elif type(next(iter(shps))) is shapely.geometry.LineString:
        # plot lines
        if 'colors' not in kwargs:
            kwargs['colors'] = color

        if _is_iter(kwargs['colors']) and \
           len(kwargs['colors']) == len(shps) and \
           not _is_iter(next(iter(kwargs['colors']))):
            # colormap!
            colors = np.array(kwargs.pop('colors'))

            if 'cmap' in kwargs: cmap = kwargs.pop('cmap')
            else: cmap = None

            if 'vmin' in kwargs: vmin = kwargs.pop('vmin')
            else: vmin = np.nanmin(colors)

            if 'vmax' in kwargs: vmax = kwargs.pop('vmax')
            else: vmax = np.nanmax(colors)

            cmapper = watershed_workflow.colors.cm_mapper(vmin, vmax, cmap)
            colors = [cmapper(c) for c in colors]
            kwargs['colors'] = colors

        lines = [np.array(l.coords)[:, 0:2] for l in shps]

        res = pltc.LineCollection(lines, **kwargs)
        if projection is not None:
            res.set_transform(projection)

        if type(ax) is Axes3D:
            res = ax.add_collection3d(res)
        else:
            res = ax.add_collection(res)
            ax.autoscale()

        if marker is not None:
            points = np.array([c for l in lines for c in l])
            if type(color) is str:
                point_colors = color
            else:
                point_colors = np.array([color[i] for (i, l) in enumerate(lines) for c in l])
            if projection is None:
                ax.scatter(points[:, 0], points[:, 1], c=point_colors, **marker_kwargs)
            else:
                ax.scatter(points[:, 0],
                           points[:, 1],
                           c=point_colors,
                           transform=projection,
                           **marker_kwargs)

    elif type(next(iter(shps))) in [shapely.geometry.Polygon, shapely.geometry.MultiPolygon]:
        if kwargs['facecolor'] in ['color', 'edge']:
            kwargs.pop('facecolor')
            face_is_edge = True
        else:
            face_is_edge = False

        if type(color) is str and color == 'elevation':
            # compute colors from the mean elevation
            color = [np.array(p.exterior.coords)[0:-1, 2].mean() for p in iter(shps)]
        elif type(color) is str and color == 'area':
            color = [p.area for p in iter(shps)]
        elif type(color) is str and color == 'log10area':
            color = np.log10(np.array([p.area for p in iter(shps)]))

        try:
            color_len = len(color)
        except (AttributeError, TypeError):
            color_len = -1

        if color is None or type(color) is str or color_len != len(shps):
            # assume this is ONE color, and therefore can add as a multipolygon/polygon collection
            if color is None:
                color = 'k'

            if 'edgecolor' not in kwargs:
                kwargs['edgecolor'] = color
            if face_is_edge:
                kwargs['facecolor'] = color

            # first must flatten
            def listify(thing):
                if type(thing) is shapely.geometry.MultiPolygon:
                    return list(thing.geoms)
                else:
                    return [thing, ]

            multi_poly = shapely.geometry.MultiPolygon([l for shp in shps for l in listify(shp)])

            patch = descartes.PolygonPatch(multi_poly, **kwargs)
            if projection is not None:
                patch.set_transform(projection)

            if type(ax) is Axes3D:
                res = ax.add_collection3d(patch)
            else:
                res = ax.add_patch(patch)

            if marker is not None:
                points = np.array([p for poly in multi_poly.geoms for p in poly.exterior.coords])
                if projection is None:
                    pnts_res = ax.scatter(points[:, 0], points[:, 1], c=color, **marker_kwargs)
                else:
                    pnts_res = ax.scatter(points[:, 0],
                                          points[:, 1],
                                          c=color,
                                          transform=projection,
                                          **marker_kwargs)

        elif type(color[0]) is tuple or type(color[0]) is np.ndarray or type(color[0]) is str:
            # list of colors
            res = []
            for shp in shps:
                patch = descartes.PolygonPatch(shp, **kwargs)
                res.append(patch)
            res = pltc.PatchCollection(res, **kwargs)

            if face_is_edge:
                res.set_facecolor(color)
            else:
                res.set_edgecolor(color)
            ax.add_collection(res)

            if marker is not None:
                points = np.array([p for poly in shps for p in poly.exterior.coords])
                pcolors = np.array(
                    [color[i] for (i, poly) in enumerate(shps) for p in poly.exterior.coords])
                if projection is None:
                    pnts_res = ax.scatter(points[:, 0], points[:, 1], c=pcolors, **marker_kwargs)
                else:
                    pnts_res = ax.scatter(points[:, 0],
                                          points[:, 1],
                                          c=pcolors,
                                          transform=projection,
                                          **marker_kwargs)

        else:
            # list of scalars that will be used with cmap to define a color
            if 'vmin' in kwargs:
                vmin = kwargs.pop('vmin')
            else:
                vmin = np.nanmin(color)

            if 'vmax' in kwargs:
                vmax = kwargs.pop('vmax')
            else:
                vmax = np.nanmax(color)
            clim = (vmin, vmax)

            res = []
            for shp in shps:
                patch = descartes.PolygonPatch(shp)

                # if projection is not None:
                #     patch.set_transform(projection)
                res.append(patch)
            res = pltc.PatchCollection(res, **kwargs)
            res.set_array(color)
            res.set_clim(clim)
            #if face_is_edge:
            print('kwargs = ', kwargs)
            print('setting face color = ', color)
            ax.add_collection(res)

            if marker is not None:
                points = np.array([p for poly in multi_poly.geoms for c in poly.exterior])
                if projection is None:
                    res = ax.scatter(points[:, 0], points[:, 1], c=color, **marker_kwargs)
                else:
                    res = ax.scatter(points[:, 0],
                                     points[:, 1],
                                     c=color,
                                     transform=projection,
                                     **marker_kwargs)

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
        fig, ax = get_ax(crs)

    if type(color) is str and color == 'elevation' and points.shape[1] != 3:
        color = 'gray'

    def get_color_extents(color):
        if 'vmin' not in kwargs:
            vmin = np.nanmin(color)
        else:
            vmin = kwargs.pop('vmin')
        if 'vmax' not in kwargs:
            vmax = np.nanmax(color)
        else:
            vmax = kwargs.pop('vmax')
        return vmin, vmax

    if type(ax) is Axes3D:
        if type(color) is str and color == 'elevation':
            col = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], tris, points[:, 2],
                                  **kwargs)
        elif type(color) != str:
            vmin, vmax = get_color_extents(color)
            col = ax.plot_trisurf(points[:, 0],
                                  points[:, 1],
                                  points[:, 2],
                                  tris,
                                  color,
                                  vmin=vmin,
                                  vmax=vmax,
                                  **kwargs)
        else:
            col = ax.plot_trisurf(points[:, 0],
                                  points[:, 1],
                                  points[:, 2],
                                  tris,
                                  color=color,
                                  **kwargs)
    else:
        if type(color) is str and color == 'elevation':
            col = ax.tripcolor(points[:, 0], points[:, 1], tris, points[:, 2], **kwargs)
        elif type(color) != str:
            vmin, vmax = get_color_extents(color)
            col = ax.tripcolor(points[:, 0],
                               points[:, 1],
                               tris,
                               color,
                               vmin=vmin,
                               vmax=vmax,
                               **kwargs)
        else:
            col = ax.triplot(points[:, 0], points[:, 1], tris, color=color, **kwargs)
    return col


def mesh(m2, crs, color='gray', ax=None, **kwargs):
    """Plots a watershed_workflow.mesh.Mesh2D object.

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
    shapes = [shapely.geometry.Polygon(m2.coords[c, :]) for c in m2.conn]
    return shplys(shapes, crs, color, ax, **kwargs)


def raster(profile, data, ax=None, vmin=None, vmax=None, mask=True, **kwargs):
    """Plots a raster.

    A wrapper for matplotlib imshow()

    Parameters
    ----------
    profile : rasterio profile
      Rasterio profile of the input raster.
    data : np.ndarray
      2D array of data.
    ax : matplotlib ax object
      Axes to plot on.  Calls get_ax() if not provided.
    vmin,vmax : float
      Min and max value to limit extent of color values.
    mask : bool
      If true (default), masks out values given as profile['nodata']
    kwargs : dict
      Dictionary of extra arguments passed to imshow().
    
    Returns
    -------
    im : matplotlib image object
      Return value of imshow()
    """
    if ax is None:
        fig, ax = get_ax(profile['crs'])

    assert (mask)
    assert ('nodata' in profile)
    if mask and 'nodata' in profile:
        nnd = len(np.where(data == profile['nodata'])[0])
        data = np.ma.array(data, mask=(data == profile['nodata']))

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    bounds = rasterio.transform.array_bounds(profile['height'], profile['width'],
                                             profile['transform'])
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    logging.info('BOUNDS: {}'.format(bounds))
    return ax.imshow(data, origin='upper', extent=extent, vmin=vmin, vmax=vmax, **kwargs)


def dem(profile, data, ax=None, vmin=None, vmax=None, **kwargs):
    """See raster documentation"""
    return raster(profile, data, ax, vmin, vmax, **kwargs)


def basemap(crs=None,
            ax=None,
            resolution='50m',
            land_kwargs=None,
            ocean_kwargs=None,
            state_kwargs=None,
            country_kwargs=None,
            coastline_kwargs=None):
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
    country_kwargs : dict
      Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
      political country boundary polygons.
    coastline_kwargs : dict
      Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
      natural coastline boundary polygons.
    """
    import cartopy.feature

    if ax is None:
        fig, ax = get_ax(crs)

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
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**state_kwargs)
        states = cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines',
                                                     resolution, **kwargs)
        # these seem a bit broken?
        if 'fix' in state_kwargs and state_kwargs.pop('fix'):
            states = watershed_workflow.utils.flatten(list(states.geometries()))
            shplys(states, watershed_workflow.crs.latlon_crs(), ax=ax, **state_kwargs)
        else:
            ax.add_feature(states)

    if country_kwargs is not None and country_kwargs is not False:
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**country_kwargs)
        country = cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land',
                                                      resolution, **kwargs)
        # these seem a bit broken?
        if 'fix' in country_kwargs and country_kwargs.pop('fix'):
            country = watershed_workflow.utils.flatten(list(country.geometries()))
            shplys(country, watershed_workflow.crs.latlon_crs(), ax=ax, **country_kwargs)
        else:
            ax.add_feature(country)

    if ocean_kwargs is not False:
        if ocean_kwargs is None:
            ocean_kwargs = dict()
        if 'edgecolor' not in ocean_kwargs:
            ocean_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in ocean_kwargs:
            ocean_kwargs['facecolor'] = cartopy.feature.COLORS['water']
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', resolution, **ocean_kwargs)
        ax.add_feature(ocean)

    if coastline_kwargs is not None and coastline_kwargs is not False:
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**coastline_kwargs)
        states = cartopy.feature.NaturalEarthFeature('physical', 'coastline', resolution, **kwargs)
        ax.add_feature(states)

    return


def feather_axis_limits(ax, delta=0.02):
    """Adds a small delta to the axis limits to provide a bit of buffer.

    Parameters
    ----------
    ax : matplotlib Axis object
      The axis to feather.
    delta : 2-tuple or double, default=0.02
      If a double, equivalent to (delta,delta).  Provides the fraction of 
      the current plot width,height to increase by.
    """
    try:
        assert (len(delta) == 2)
    except AssertionError:
        raise RuntimeError("feather_axis_limits expects delta argument of length 2 (dx,dy)")
    except ValueError:
        delta = (delta, delta)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = delta[0] * (xlim[1] - xlim[0])
    dy = delta[1] * (ylim[1] - ylim[0])
    ax.set_xlim((xlim[0] - dx, xlim[1] + dx))
    ax.set_ylim((ylim[0] - dy, ylim[1] + dy))
