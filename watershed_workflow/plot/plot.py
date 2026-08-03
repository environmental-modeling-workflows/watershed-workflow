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

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterable, TYPE_CHECKING
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as pltc
from matplotlib import cm as pcm
import shapely
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
import itertools
import xarray

import watershed_workflow.utils.geometry
import watershed_workflow.crs
import watershed_workflow.plot.colors

if TYPE_CHECKING:
    from watershed_workflow.mesh.mesh import Mesh2D, Edge

__all__ = ['findCell', 'plotCellContext']


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
                                if not watershed_workflow.utils.geometry.isEmpty(geo)):
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


def findCell(m2 : 'Mesh2D',
            cell : Optional[int] = None,
            edge : Optional[Any] = None,
            coordinate : Optional[Tuple[float, float]] = None,
            ) -> int:
    """Resolve a cell, edge, or coordinate to a single cell index.

    Exactly one of cell, edge, or coordinate must be provided.

    Parameters
    ----------
    m2 : Mesh2D
        The mesh to search.
    cell : int, optional
        A cell index -- returned as-is.
    edge : Edge or Tuple[int,int], optional
        An edge (pair of vertex indices) -- returns one of its
        neighboring cells (the first, if a boundary edge with only one).
    coordinate : Tuple[float, float], optional
        An (x,y) point in the mesh's CRS -- returns the cell whose
        centroid is nearest to the point.

    Returns
    -------
    int
        The resolved cell index.
    """
    from watershed_workflow.mesh.mesh import Edge

    provided = [v is not None for v in (cell, edge, coordinate)]
    if sum(provided) != 1:
        raise ValueError('findCell(): exactly one of cell, edge, or coordinate must be provided.')

    if cell is not None:
        return cell
    elif edge is not None:
        cells = m2.edge_cells[Edge(edge)]
        return cells[0]
    else:
        dists = np.linalg.norm(m2.centroids[:, 0:2] - np.asarray(coordinate), axis=1)
        return int(np.argmin(dists))


def plotCellContext(m2 : 'Mesh2D',
                    cell : Optional[int] = None,
                    edge : Optional[Any] = None,
                    coordinate : Optional[Tuple[float, float]] = None,
                    context_rings : int = 3,
                    dem : Optional[xarray.DataArray] = None,
                    dem_sm : Optional[xarray.DataArray] = None,
                    ax : Optional[Any] = None,
                    title : Optional[str] = None,
                    ) -> Tuple[Any, Any, int]:
    """Plot a cell zoomed in with its surrounding context, optionally against DEM(s).

    Useful for debugging mesh elevations/topology at a specific
    location. Exactly one of cell, edge, or coordinate must identify
    the location of interest. Shows, left to right:

    - Full mesh, colored by elevation, with a box showing where the
      zoomed-in panel sits and a marker at the target cell.
    - A zoomed-in view of the target cell highlighted against its
      neighbors' elevations, with per-vertex elevations annotated and
      the mesh outline drawn.
    - (if dem provided) The source DEM, cropped to the same zoom
      region, with the mesh outline overlaid in light gray.
    - (if dem_sm provided) A second DEM (e.g. smoothed), cropped to the
      same region, with the mesh outline overlaid.

    All elevation panels (mesh and DEM) share a single color scale, so
    DEM noise/features can be compared directly against mesh elevations.

    Parameters
    ----------
    m2 : Mesh2D
        The mesh, with current elevations in m2.coords[:,2].
    cell : int, optional
        Cell index to plot.
    edge : Edge or Tuple[int,int], optional
        An edge identifying the cell to plot (one of its neighbors).
    coordinate : Tuple[float, float], optional
        An (x,y) point (mesh CRS) -- the nearest cell is plotted.
    context_rings : int, optional
        Number of rings of neighbors (breadth-first, via
        cell_to_cells) to include around the target cell for context in
        the zoomed-in mesh panel. Default is 3.
    dem : xr.DataArray, optional
        A DEM (same CRS as m2). If provided, adds a panel showing it
        cropped to the local context.
    dem_sm : xr.DataArray, optional
        A second DEM, e.g. a smoothed version of dem. If provided, adds
        a panel showing it cropped to the same region.
    ax : array-like of matplotlib.Axes, optional
        Axes to plot on -- length 2, 3, or 4 depending on whether dem /
        dem_sm are provided. If None, creates a new figure.
    title : str, optional
        Title for the zoomed-in panel. Defaults to "cell {id}".

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : array-like of matplotlib.Axes
    c0 : int
        The resolved cell index that was plotted.
    """
    c0 = findCell(m2, cell=cell, edge=edge, coordinate=coordinate)

    # gather local context: BFS out from the target cell through cell_to_cells
    context_cells = {c0}
    frontier = {c0}
    for _ in range(context_rings):
        next_frontier = set()
        for c in frontier:
            next_frontier.update(m2.cell_to_cells[c])
        next_frontier -= context_cells
        context_cells.update(next_frontier)
        frontier = next_frontier

    context_cells = sorted(context_cells)
    context_conn = [m2.conn[c] for c in context_cells]
    context_verts = sorted(set(v for c in context_cells for v in m2.conn[c]))

    # compute the zoom region bounds, with a small margin
    xs = [m2.coords[v, 0] for v in context_verts]
    ys = [m2.coords[v, 1] for v in context_verts]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    margin = 0.05 * max(dx, dy, 1.0)
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)

    # crop the DEM(s) to the zoom region, if provided.  xarray's .sel()
    # with a slice requires bounds ordered to match the coordinate's own
    # ordering (many rasters store y descending, but this isn't
    # guaranteed) -- so order each slice using the coordinate's actual
    # first/last values rather than assuming.
    def _cropToZoom(da):
        y_ascending = da.y[0].item() < da.y[-1].item()
        y_slice = slice(ylim[0], ylim[1]) if y_ascending else slice(ylim[1], ylim[0])
        x_ascending = da.x[0].item() < da.x[-1].item()
        x_slice = slice(xlim[0], xlim[1]) if x_ascending else slice(xlim[1], xlim[0])
        cropped = da.sel(x=x_slice, y=y_slice)
        if cropped.size == 0:
            raise ValueError(
                f'plotCellContext(): cropping the DEM to the zoom region ({xlim}, {ylim}) '
                f'returned an empty array. Check that the DEM and mesh share the same CRS '
                f'and overlap spatially (DEM x range: {float(da.x.min())}-{float(da.x.max())}, '
                f'y range: {float(da.y.min())}-{float(da.y.max())}).')
        return cropped

    dem_crops = []
    if dem is not None:
        dem_crops.append(('DEM', _cropToZoom(dem)))
    if dem_sm is not None:
        dem_crops.append(('DEM (smoothed)', _cropToZoom(dem_sm)))

    n_panels = 2 + len(dem_crops)
    if ax is None:
        fig, ax = plt.subplots(1, n_panels, figsize=(8 * n_panels, 8))
    else:
        fig = ax[0].figure
    if n_panels == 1:
        ax = [ax]
    ax_full, ax_zoom = ax[0], ax[1]
    ax_dems = ax[2:]

    # shared elevation color scale across ALL elevation panels (mesh and DEM)
    context_centroids = m2.centroids[context_cells]
    context_vert_z = m2.coords[context_verts, 2]
    logging.info(f'plotCellContext: mesh local elevation range = '
                 f'[{np.nanmin(context_vert_z):.2f}, {np.nanmax(context_vert_z):.2f}] masl '
                 f'(vertices), [{np.nanmin(context_centroids[:,2]):.2f}, '
                 f'{np.nanmax(context_centroids[:,2]):.2f}] masl (cell centroids)')
    for label, crop in dem_crops:
        logging.info(f'plotCellContext: {label} local range = '
                     f'[{float(np.nanmin(crop.values)):.2f}, {float(np.nanmax(crop.values)):.2f}] masl')

    all_vals = [context_centroids[:, 2], context_vert_z]
    for _, crop in dem_crops:
        all_vals.append(crop.values.ravel())
    vmin = float(np.nanmin([np.nanmin(v) for v in all_vals]))
    vmax = float(np.nanmax([np.nanmax(v) for v in all_vals]))
    cmap = 'gist_earth'

    # -- panel 0: full mesh, with a box showing the zoom region --
    m2.plot(facecolors='elevation', ax=ax_full, colorbar=True, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_full.plot(*m2.centroids[c0, 0:2], marker='*', color='red', markersize=15,
                 markeredgecolor='k', zorder=6)
    rect = plt.Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                          facecolor='none', edgecolor='red', linewidth=2, zorder=5)
    ax_full.add_patch(rect)
    ax_full.set_aspect('equal', adjustable='box')
    ax_full.set_title('Full mesh (red box = zoom region)')

    # -- panel 1: zoomed-in local context, colored by elevation --
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    verts = [[m2.coords[i, 0:2] for i in conn] for conn in context_conn]
    facecolors = context_centroids[:, 2]
    gons = pltc.PolyCollection(verts, array=facecolors, cmap=cmap, norm=norm,
                               edgecolors='grey', linewidth=0.5)
    ax_zoom.add_collection(gons)
    plt.colorbar(gons, ax=ax_zoom, label='elevation [masl]')

    # highlight the target cell itself with a heavy outline
    cell_poly = pltc.PolyCollection([[m2.coords[i, 0:2] for i in m2.conn[c0]]],
                                    facecolor='none', edgecolors='red', linewidth=3)
    ax_zoom.add_collection(cell_poly)

    # mark the target cell's neighbors, to make the local topology explicit
    for n in m2.cell_to_cells[c0]:
        ax_zoom.plot(*m2.centroids[n, 0:2], marker='o', color='cyan', markersize=6, zorder=5)
        ax_zoom.plot([m2.centroids[c0, 0], m2.centroids[n, 0]],
                     [m2.centroids[c0, 1], m2.centroids[n, 1]],
                     color='cyan', linewidth=1, zorder=4)

    # annotate vertex elevations
    for v in context_verts:
        x, y, z = m2.coords[v]
        ax_zoom.annotate(f'{z:.2f}', (x, y), fontsize=7, color='k',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

    # mark the target cell's centroid
    ax_zoom.plot(*m2.centroids[c0, 0:2], marker='*', color='red', markersize=20,
                 markeredgecolor='k', zorder=6, label='target cell')

    ax_zoom.set_aspect('equal', adjustable='box')
    ax_zoom.legend(loc='upper right')
    ax_zoom.set_title(title if title is not None else f'cell {c0}')
    ax_zoom.set_xlim(*xlim)
    ax_zoom.set_ylim(*ylim)

    # -- remaining panels: cropped DEM(s), same color scale, mesh overlaid --
    cell_xy = m2.centroids[c0, 0:2]
    mesh_verts = [[m2.coords[i, 0:2] for i in conn] for conn in context_conn]
    for ax_dem, (label, crop) in zip(ax_dems, dem_crops):
        crop.plot.imshow(ax=ax_dem, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=True)
        mesh_outline = pltc.PolyCollection(mesh_verts, facecolor='none',
                                          edgecolors='lightgray', linewidth=0.5, zorder=4)
        ax_dem.add_collection(mesh_outline)
        ax_dem.plot(*cell_xy, marker='*', color='red', markersize=15,
                    markeredgecolor='k', zorder=6)
        ax_dem.set_title(label)
        ax_dem.set_aspect('equal', adjustable='box')
        ax_dem.set_xlim(*xlim)
        ax_dem.set_ylim(*ylim)

    fig.tight_layout()
    return fig, ax, c0
