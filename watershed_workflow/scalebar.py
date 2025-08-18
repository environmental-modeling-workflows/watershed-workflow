"""A module for drawing a scale bar in cartopy.

NOTE: this is based on a stack overflow example written by @mephistolotl, and
was staged at:

  https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot

"""

from typing import Tuple, Callable, Union, Optional, Dict, Any
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo


def _axes_to_lonlat(ax: plt.Axes, coords: Union[Tuple[float, float],
                                                  np.ndarray]) -> Tuple[float, float]:
    """Convert axes coordinates to (lon, lat).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        CartoPy axes object.
    coords : Tuple[float, float] or np.ndarray
        Coordinates in axes coordinate system.

    Returns
    -------
    Tuple[float, float]
        Longitude and latitude coordinates.
    """
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)
    return lonlat


def _upper_bound(start: np.ndarray, direction: np.ndarray, distance: float,
                 dist_func: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Find a point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Parameters
    ----------
    start : np.ndarray
        Starting point for the line.
    direction : np.ndarray
        Nonzero (2,)-shaped array, a direction vector.
    distance : float
        Positive distance to go past.
    dist_func : Callable
        A two-argument function which returns distance between points.

    Returns
    -------
    np.ndarray
        Coordinates of a point farther than distance from start.

    Raises
    ------
    ValueError
        If distance is not positive or direction vector is zero.
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length*direction
    while dist_func(start, end) < distance:
        length *= 2
        end = start + length*direction

    return end


def _distance_along_line(start: np.ndarray, end: np.ndarray, distance: float,
                         dist_func: Callable[[np.ndarray, np.ndarray],
                                             float], tol: float) -> np.ndarray:
    """Find point at a distance from start on the linestring from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Parameters
    ----------
    start : np.ndarray
        Starting point for the line.
    end : np.ndarray
        Outer bound on point's location.
    distance : float
        Positive distance to travel.
    dist_func : Callable
        Two-argument function which returns distance between points.
    tol : float
        Relative error in distance to allow.

    Returns
    -------
    np.ndarray
        Coordinates of a point at the specified distance from start.

    Raises
    ------
    ValueError
        If end is closer to start than given distance, or tolerance is not positive.
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left+right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax: plt.Axes,
                      start: np.ndarray,
                      distance: float,
                      angle: float = 0,
                      tol: float = 0.01) -> np.ndarray:
    """Find point at a given distance from start at a given angle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        CartoPy axes object.
    start : np.ndarray
        Starting point for the line in axes coordinates.
    distance : float
        Positive physical distance to travel (in meters).
    angle : float, optional
        Anti-clockwise angle for the bar, in radians. Default is 0.
    tol : float, optional
        Relative error in distance to allow. Default is 0.01.

    Returns
    -------
    np.ndarray
        Coordinates of a point at the specified distance and angle.
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys)[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scalebar(ax: plt.Axes,
             location: Union[Tuple[float, float], np.ndarray],
             length: float,
             metres_per_unit: float = 1000,
             unit_name: str = 'km',
             tol: float = 0.01,
             angle: float = 0,
             color: str = 'black',
             linewidth: float = 3,
             text_offset: float = 0.005,
             ha: str = 'center',
             va: str = 'bottom',
             plot_kwargs: Optional[Dict[str, Any]] = None,
             text_kwargs: Optional[Dict[str, Any]] = None,
             **kwargs: Any) -> None:
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        CartoPy axes object.
    location : Tuple[float, float] or np.ndarray
        Position of left-side of bar in axes coordinates.
    length : float
        Geodesic length of the scale bar in specified units.
    metres_per_unit : float, optional
        Number of metres in the given unit. Default is 1000 (km).
    unit_name : str, optional
        Name of the given unit. Default is 'km'.
    tol : float, optional
        Allowed relative error in length of bar. Default is 0.01.
    angle : float, optional
        Anti-clockwise rotation of the bar in degrees. Default is 0.
    color : str, optional
        Color of the bar and text. Default is 'black'.
    linewidth : float, optional
        Line width for the scale bar. Default is 3.
    text_offset : float, optional
        Perpendicular offset for text in axes coordinates. Default is 0.005.
    ha : str, optional
        Horizontal alignment of text. Default is 'center'.
    va : str, optional
        Vertical alignment of text. Default is 'bottom'.
    plot_kwargs : Dict[str, Any], optional
        Keyword arguments for the plot call. Overridden by **kwargs.
    text_kwargs : Dict[str, Any], optional
        Keyword arguments for the text call. Overridden by **kwargs.
    **kwargs : Any
        Additional keyword arguments passed to both plot and text calls.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = { 'linewidth': linewidth, 'color': color, **plot_kwargs, **kwargs }
    text_kwargs = { 'ha': ha, 'va': va, 'rotation': angle, 'color': color, **text_kwargs, **kwargs }

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad, tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location+end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location,
            f"{length} {unit_name}",
            rotation_mode='anchor',
            transform=ax.transAxes,
            **text_kwargs)
