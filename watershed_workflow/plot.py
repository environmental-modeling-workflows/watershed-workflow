# mypy: ignore-errors
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

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterable
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as pltc
from matplotlib import cm as pcm
import shapely
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
import itertools

import watershed_workflow.utils
import watershed_workflow.crs
import watershed_workflow.colors


def _is_iter(obj: Any) -> bool:
    """Check if an object is iterable.

    Parameters
    ----------
    obj : Any
        Object to test for iterability.

    Returns
    -------
    bool
        True if object is iterable, False otherwise.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def linestringsWithCoords(df: gpd.GeoDataFrame | Iterable[shapely.geometry.LineString],
                        column: Optional[str] = None,
                        marker: Optional[str] = None,
                        **kwargs) -> plt.Axes:
    """Plot linestrings, but also potentially scatter their coordinates.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        GeoDataFrame containing LineString geometries to plot.
    column : str, optional
        Column name to use for coloring. If None, uses cycled colors.
    marker : str, optional
        Marker style to scatter at line coordinates. If None, no markers plotted.
    **kwargs : Any
        Additional keyword arguments passed to df.plot() and ax.scatter() calls.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(geometry=df)
        column = None
    
    if marker:
        marker_args = { 'marker': marker }
        if 'markersize' in kwargs:
            marker_args['s'] = kwargs.pop('markersize')

    # force cycled colors as default, not all blue as default
    if column is None and 'color' not in kwargs:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color = [c for (ind, c) in zip(df.index, itertools.cycle(color_cycle))]
        kwargs['color'] = color

    # call the default plotter, which, because this is all
    # LineStrings, will always add exactly one collection.
    ax = df.plot(**kwargs)

    if marker is not None:
        lc = ax.collections[-1]
        colors = lc.get_colors()

        # scatter the markers
        for i, seg in enumerate(geo for geo in df.geometry
                                if not watershed_workflow.utils.isEmpty(geo)):
            if len(colors) == 1:
                color = colors[0]
            else:
                color = colors[i]
            ax.scatter(seg.xy[0], seg.xy[1], color=color, **marker_args)

    return ax

def linestringWithCoords(ls, *args, **kwargs):
    return linestringsWithCoords(gpd.GeoDataFrame(geometry=[ls,]), 'geometry', *args, **kwargs)



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
    def __init__(self,
                 ax: 'plt.Axes',
                 items: Optional[List[Tuple[Any, Any, Union[Callable, str]]]] = None) -> None:
        self.ax = ax
        self.items : List[Any] = []
        if items is not None:
            for item in items:
                self.addItem(*item)

        self.ax.set_title("None")
        self.selected = None

    def addItem(self, data: List[Any],
                artist: pltc.Collection,
                formatter: Union[Callable[[Any], str], str]) -> None:
        """Adds an item to the list of things to label.

        Parameters
        ----------
        data : List[Any]
            A list of objects being labeled.  This is likely the
            underlying data, with properties, that was passed to
            a ww.plot function.
        artist : matplotlib.collections.Collection
            A matplotlib Collection, likely the return value of
            a ww.plot call or similar.
        formatter : Callable or str
            A function that accepts an entry in data and returns a
            string to label the item selected.  If this is a string,
            it is assumed to be a formattable string to which the
            item's properties dictionary is passed.
        """
        if isinstance(formatter, str):
            def format_this(item):
                return formatter.format(**dict(item)), list()
            formatter = format_this

        assert (len(artist) == len(data))
        self.items.append((artist, data, formatter))
        self._selected = []

    def deselect(self) -> None:
        """Clears anything plotted in the last click."""
        for artist in self._selected:
            artist.clear()
        self._selected = []

    def select(self, i: int, j: int, xy: Tuple[float, float]) -> None:
        """Selects item i, collection index j, with a click at xy.

        Parameters
        ----------
        i : int
            Index of the item in the items list.
        j : int
            Index within the collection.
        xy : Tuple[float, float]
            Click coordinates.
        """
        data, artist, formatter = self.items[i]

        if isinstance(data, list):
            dat = data[j]
            if isinstance(dat, shapely.geometry.base.BaseGeometry) and hasattr(dat, 'properties'):
                dat = dict(geometry=dat, **dat.properties)
            title = formatter(dat)
        elif isinstance(data, gpd.GeoDataFrame):
            title = formatter(data.iloc[j])
        self.ax.set_title(title)

        # redraw LineStrings with markers
        if isinstance(artist, pltc.LineCollection):
            line = artist.get_data()[i]
            color = artist.get_colors()[i]

            self._selected.append(self.ax.plot(line[:, 0], line[:, 1], '-x', color=color))

    def update(self, event: Any) -> None:
        """Acts on click.

        Parameters
        ----------
        event : matplotlib event
            The click event from matplotlib.
        """
        print('event loc:', event.mouseevent.x, event.mouseevent.y)
        print('event dict:', event.__dict__)

        i = next(i for (i, item) in enumerate(self.items) if item[0] is event.artist)
        self.select(i, 0, (event.mouseevent.x, event.mouseevent.y))
        self.ax.get_figure().canvas.draw_idle()


def triangulation(points: np.ndarray,
                  tris: Union[List, np.ndarray],
                  ax: plt.Axes,
                  **kwargs: Any) -> Any:
    """Plots a triangulation.

    A wrapper for matplotlib's plot_trisurf() or tripcolor()

    Parameters
    ----------
    points : np.ndarray
        Array of point coordinates, shape (npoints, 2) or (npoints, 3).
    tris : list or np.ndarray
        List of lists or ndarray of indices into the points array for defining
        the triangle topology, shape (ntris, 3).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs : Any
        Extra arguments passed to plot_trisurf() (for 3D axes) or tripcolor()
        (for 2D axes).

    Returns
    -------
    matplotlib collection
        The triangulation plot object.
    """
    color = kwargs.get('color', 'elevation')
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
            col = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], tris, points[:, 2],
                                  **kwargs)
        elif type(color) != str:
            vmin, vmax = get_color_extents(color)
            if 'vmin' not in kwargs:
                kwargs['vmin'] = vmin
            if 'vmax' not in kwargs:
                kwargs['vmax'] = vmax
            col = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], tris, color, **kwargs)
        else:
            col = ax.plot_trisurf(points[:, 0],
                                  points[:, 1],
                                  points[:, 2],
                                  tris,
                                  color=color,
                                  **kwargs)
    else:
        if isinstance(color, str) and color == 'elevation':
            col = ax.tripcolor(points[:, 0], points[:, 1], tris, points[:, 2], **kwargs)
        elif type(color) != str:
            vmin, vmax = get_color_extents(color)
            if 'vmin' not in kwargs:
                kwargs['vmin'] = vmin
            if 'vmax' not in kwargs:
                kwargs['vmax'] = vmax
            col = ax.tripcolor(points[:, 0], points[:, 1], tris, color, **kwargs)
        else:
            col = ax.triplot(points[:, 0], points[:, 1], tris, color=color, **kwargs)
    return col


def basemap(crs: Optional[Any] = None,
            ax: Optional[plt.Axes] = None,
            resolution: str = '50m',
            land_kwargs: Optional[Dict[str, Any] | bool] = None,
            ocean_kwargs: Optional[Dict[str, Any] | bool] = None,
            state_kwargs: Optional[Dict[str, Any] | bool] = None,
            country_kwargs: Optional[Dict[str, Any] | bool] = None,
            coastline_kwargs: Optional[Dict[str, Any] | bool] = None,
            lake_kwargs: Optional[Dict[str, Any] | bool] = None) -> 'plt.Axes':
    """Add a basemap to the axis.

    Uses cartopy to add political and natural boundaries and shapes to the axes
    image.

    Parameters
    ----------
    crs : CRS object, optional
        Coordinate system to plot. May be ignored if ax is provided.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If not provided, creates new subplot.
    resolution : str, optional
        Resolution of cartopy basemap. One of '10m', '50m', or '110m'.
        Default is '50m'.
    land_kwargs : dict or bool, optional
        Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
        land polygons. If False, land features are not added.
    ocean_kwargs : dict or bool, optional
        Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
        ocean polygons. If False, ocean features are not added.
    state_kwargs : dict or bool, optional
        Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
        political state boundary polygons. If False, state features are not added.
    country_kwargs : dict or bool, optional
        Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
        political country boundary polygons. If False, country features are not added.
    coastline_kwargs : dict or bool, optional
        Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
        natural coastline boundary polygons. If False, coastline features are not added.
    lake_kwargs : dict or bool, optional
        Extra arguments passed to cartopy.feature.NaturalEarthFeature call to get
        lake polygons. If False, lake features are not added.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with basemap features added.
    """
    import cartopy.feature

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    if land_kwargs is not False:
        if land_kwargs is None or land_kwargs is True:
            land_kwargs = dict()
        if 'edgecolor' not in land_kwargs:
            land_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in land_kwargs:
            land_kwargs['facecolor'] = cartopy.feature.COLORS['land']
        land = cartopy.feature.NaturalEarthFeature('physical', 'land', resolution, **land_kwargs)
        ax.add_feature(land)

    if ocean_kwargs is not False:
        if ocean_kwargs is None or ocean_kwargs is True:
            ocean_kwargs = dict()
        if 'edgecolor' not in ocean_kwargs:
            ocean_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in ocean_kwargs:
            ocean_kwargs['facecolor'] = cartopy.feature.COLORS['water']
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', resolution, **ocean_kwargs)
        ax.add_feature(ocean)

    if lake_kwargs is not None and lake_kwargs is not False:
        if lake_kwargs is True:
            lake_kwargs = dict()
        if 'edgecolor' not in lake_kwargs:
            lake_kwargs['edgecolor'] = 'face'
        if 'facecolor' not in lake_kwargs:
            lake_kwargs['facecolor'] = cartopy.feature.COLORS['water']
        lake = cartopy.feature.NaturalEarthFeature('physical', 'lakes', resolution, **lake_kwargs)
        ax.add_feature(lake)

    if coastline_kwargs is not None and coastline_kwargs is not False:
        if coastline_kwargs is True:
            coastline_kwargs = dict()
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**coastline_kwargs)
        states = cartopy.feature.NaturalEarthFeature('physical', 'coastline', resolution, **kwargs)
        ax.add_feature(states)

    if state_kwargs is not None and state_kwargs is not False:
        if state_kwargs is True:
            state_kwargs = dict()
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**state_kwargs)
        states = cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines',
                                                     resolution, **kwargs)
        ax.add_feature(states)

    if country_kwargs is not None and country_kwargs is not False:
        if country_kwargs is True:
            country_kwargs = dict()
        kwargs = { 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 0.5 }
        kwargs.update(**country_kwargs)
        country = cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land',
                                                      resolution, **kwargs)
        # these seem a bit broken?
        ax.add_feature(country)
    return ax
