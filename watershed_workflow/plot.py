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
from mpl_toolkits.mplot3d import Axes3D
import geopandas
import itertools

import watershed_workflow.utils
import watershed_workflow.crs
import watershed_workflow.colors


def _is_iter(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def linestringsWithCoords(df, column=None, marker=None, **kwargs):
    """Plot linestrings, but also potentially scatter their coordinates."""
    if marker:
        marker_args = {'marker' : marker}
        if 'markersize' in kwargs:
            marker_args['s'] = kwargs.pop('markersize')

    # force cycled colors as default, not all blue as default
    if column is None and 'color' not in kwargs:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color = [c for (ind,c) in zip(df.index, itertools.cycle(color_cycle))]
        kwargs['color'] = color

    # call the default plotter, which, because this is all
    # LineStrings, will always add exactly one collection.
    ax = df.plot(**kwargs)

    if marker is not None:
        lc = ax.collections[-1]
        colors = lc.get_colors()

        # scatter the markers
        for i, seg in enumerate(geo for geo in df.geometry if not watershed_workflow.utils.isEmpty(geo)):
            if len(colors) == 1:
                color = colors[0]
            else:
                color = colors[i]
            ax.scatter(seg.xy[0], seg.xy[1], color=color, **marker_args)

    return ax
                          


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


