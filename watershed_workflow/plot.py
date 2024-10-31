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
import geopandas

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


def shply(shply, **kwargs):
    """Plot a shapely object, in whatever CRS it is in.

    Parameters
    ----------
    shply : shapely.geometry object

    **kwargs
      passed to geopandas.GeoDataFrame.plot
    
    Returns
    -------
    ax : matplotlib axis
    """
    df = geopandas.GeoDataFrame(geometry=[shply,])
    return df.plot(**kwargs)


def hucs(hucs,
         outlet_kwargs=None,
         **kwargs):
    """Plot a SplitHUCs object.
    
    Parameters
    ----------
    hucs : watershed_workflow.split_hucs.SplitHucs object
      The collection of hucs to plot.
    outlet_kwargs : dict
      kwargs to pass to scatterplot of outlets
    kwargs : dict
      Extra arguments passed to geopandas.GeoDataFrame.plot

    Returns
    -------
    ax : matplotlib.Axes object
    """
    df = hucs.to_dataframe()
    ax = df.plot(**kwargs)
    df_out = geopandas.GeoDataFrame(crs=df.crs, geometry=df['outlets'])
    df_out.plot(ax=ax, **outlet_kwargs)
    return ax


def river(river, **kwargs):
    """Plot an itereable collection of reaches.

    A wrapper for plot.shply()

    Parameters
    ----------
    river : list(shapely.LineString)
      An iterable of shapely LineString reaches.
    kwargs : dict
      Extra arguments passed to geopandas.GeoDataFrame.plot

    Returns
    -------
    ax : matplotlib.Axes object
    """
    df = river.to_dataframe()
    return df.plot(**kwargs)


def rivers(rivers, color=None, ax=None, **kwargs):
    """Plot an itereable collection of river Tree objects.

    A wrapper for plot.shply()

    Parameters
    ----------
    rivers : list(river_tree.RiverTree)
      An iterable of river_tree.RiverTree objects.
    color : str, scalar, tuple, or iterable, optional
      If it is a tuple, this is assumed to be a single color
      (e.g. RGB, etc).  If it is a list or other iterable, this must
      be the same length as rivers, and each entry is used to color
      each river independently.
    ax : matplotib axes object, optional
      Axes to plot on.
    kwargs : dict
      Extra arguments passed to the plotting method, which is likely
      matplotlib.collections.LineCollection.

    Returns
    -------
    ax : matplotlib.Axes object

    """
    if color is None:
        color = watershed_workflow.colors.enumerated_colors(len(rivers))

    if type(color) is not str and len(color) == len(rivers):
        for r, c in zip(rivers, color):
            ax = river(r, color=c, **kwargs)
    else:
        for r in rivers:
            ax = river(r, color=color, **kwargs)
            

def triangulation(points, tris, ax=None, **kwargs):
    """Plots a triangulation.

    A wrapper for matplotlib's plot_trisurf() or tripcolor()

    Parameters
    ----------
    points : np.ndarray(npoints, 3)
      Array of point coordinates, x,y,z.
    tris : list, np.ndarray(ntris, 3)
      List of lists or ndarray of indices into the points array for defining
      the triangle topology.
    ax : matplotlib.Axes object
      Axes to plot on.  Default is None.
    kwargs : dict
      Extra arguments passed to plot_trisurf() (for 3D axes) or tripcolor()
      (for 2D).

    Returns
    -------
    ax : matplotlib.Axes object
    """
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

    if isinstance(ax, Axes3D):
        if type(color) is str and color == 'elevation':
            col = ax.plot_trisurf(points[:, 0],
                                  points[:, 1],
                                  points[:, 2],
                                  tris,
                                  points[:, 2],
                                  **kwargs)
        elif type(color) != str:
            vmin, vmax = get_color_extents(color)
            if 'vmin' not in kwargs:
                kwargs['vmin'] = vmin
            if 'vmax' not in kwargs:
                kwargs['vmax'] = vmax
            col = ax.plot_trisurf(points[:, 0],
                                  points[:, 1],
                                  points[:, 2],
                                  tris,
                                  color,
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
            col = ax.tripcolor(points[:, 0],
                               points[:, 1],
                               tris,
                               points[:, 2],
                               **kwargs)
        elif type(color) != str:
            vmin, vmax = get_color_extents(color)
            if 'vmin' not in kwargs:
                kwargs['vmin'] = vmin
            if 'vmax' not in kwargs:
                kwargs['vmax'] = vmax
            col = ax.tripcolor(points[:, 0],
                               points[:, 1],
                               tris,
                               color,
                               **kwargs)
        else:
            col = ax.triplot(points[:, 0], points[:, 1], tris, color=color, **kwargs)
    return col


def mesh(m2, **kwargs):
    """Plots a watershed_workflow.mesh.Mesh2D object.

    Parameters
    ----------
    m2 : Mesh2D
      The 2D mesh to plot.
    kwargs : dict
      Extra arguments passed to geopandas.GeoDataFrame.plot

    Returns
    -------
    ax : matplotlib.Axes object

    """
    df = geopandas.GeoDataFrame(geometry=[shapely.geometry.Polygon(m2.coords[c, :]) for c in m2.conn])
    return df.plot(**kwargs)


def basemap(crs=None,
            ax=None,
            resolution='50m',
            land_kwargs=None,
            ocean_kwargs=None,
            state_kwargs=None,
            country_kwargs=None,
            coastline_kwargs=None,
            lake_kwargs=None):
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

    Returns
    -------
    ax : matplotlib.Axes object
    """
    import cartopy.feature

    if ax is None:
        fig, ax = plt.subplots(1,1,1)

    if land_kwargs is not False:
        if land_kwargs is None:
            land_kwargs = dict()
        if 'edgecolor' not in land_kwargs:
            land_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in land_kwargs:
            land_kwargs['facecolor'] = cartopy.feature.COLORS['land']
        land = cartopy.feature.NaturalEarthFeature('physical', 'land', resolution, **land_kwargs)
        ax.add_feature(land)

    if ocean_kwargs is not False:
        if ocean_kwargs is None:
            ocean_kwargs = dict()
        if 'edgecolor' not in ocean_kwargs:
            ocean_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in ocean_kwargs:
            ocean_kwargs['facecolor'] = cartopy.feature.COLORS['water']
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', resolution, **ocean_kwargs)
        ax.add_feature(ocean)

    if lake_kwargs is not None and lake_kwargs is not False:
        if 'edgecolor' not in lake_kwargs:
            lake_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in lake_kwargs:
            lake_kwargs['facecolor'] = cartopy.feature.COLORS['water']
        lake = cartopy.feature.NaturalEarthFeature('physical', 'lakes', resolution, **lake_kwargs)
        ax.add_feature(lake)

    if coastline_kwargs is not None and coastline_kwargs is not False:
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**coastline_kwargs)
        states = cartopy.feature.NaturalEarthFeature('physical', 'coastline', resolution, **kwargs)
        ax.add_feature(states)

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
    return ax


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
