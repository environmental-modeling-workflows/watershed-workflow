from typing import List, Dict, Tuple, Union, Optional, Callable, Any, Sequence, Literal, TypedDict, cast

Segment = Tuple[float, ...]
Channel = Literal["red", "green", "blue", "alpha"]
SegmentData = Dict[Channel, Sequence[Segment]]

try:
    from matplotlib.colors import Color  # type: ignore
except ImportError:
    Color = Union[str,  # named color, hex, grayscale str, shorthand
                  Tuple[float, float, float],  # RGB
                  Tuple[float, float, float, float],  # RGBA
                  ]

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import numpy as np
import collections
import random
import colorsys

#
# Lists of disparate color palettes
#
enumerated_palettes: Dict[int, List[Color]] = {
    1: [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf',
        '#999999'
    ],
    2: [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
        '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
    ],
    3: ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'],
    4: [
        "#399283", "#d2b48b", "#7f3a63", "#f3c5fa", "#e0079b", "#474747", "#c00018", "#2e21d0",
        "#5be13e", "#bce091", "#ed8220", "#769d31", "#d0de20", "#cd6ec6", "#547eec", "#8bd0eb",
        "#333a9e", "#94721a", "#d17778", "#f3c011", "#1eefc9", "#8e3703", "#02531d", "#d62df6"
    ],
}


def isNearlyGrey(color: Color, tolerance: float = 0.1) -> bool:
    """Determines whether a color is nearly grey.

    Parameters
    ----------
    color : Color
        Color to test for greyness.
    tolerance : float, optional
        Tolerance for RGB component differences. Default is 0.1.

    Returns
    -------
    bool
        True if color is nearly grey (RGB components within tolerance).
    """
    # Ensure the color hash is valid
    r, g, b, a = matplotlib.colors.to_rgba(color)

    # Check if the RGB values are within the tolerance range
    return abs(r - g) <= tolerance and abs(g - b) <= tolerance and abs(b - r) <= tolerance


def measureBoldness(color: Color) -> float:
    """Calculate a vibrancy and boldness score for a given color.

    Parameters
    ----------
    color : Color
        Color to measure boldness for.

    Returns
    -------
    float
        A score representing how vibrant and bold the color is (0 to 100).
    """
    r, g, b, a = matplotlib.colors.to_rgba(color)

    # Convert RGB to HSL for better vibrancy measurement
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Calculate vibrancy score (saturation and brightness impact vibrancy)
    vibrancy = s * (1 - abs(2*l - 1))

    # Calculate boldness score (intensity of RGB components)
    boldness = (r+g+b) / 3

    # Combine vibrancy and boldness into a single score (weighted average)
    score = (0.6*vibrancy + 0.4*boldness) * 100

    return round(score, 2)


# create a very big list of non-grey colors
_my_not_random = random.Random(7)  #2
xkcd_colors = [c for c in matplotlib.colors.XKCD_COLORS.values() if not isNearlyGrey(c)]
_my_not_random.shuffle(xkcd_colors)

_xkcd_by_bold = list(reversed(sorted(xkcd_colors, key=measureBoldness)))

xkcd_bolds = _xkcd_by_bold[0:len(_xkcd_by_bold) // 4]
_my_not_random.shuffle(xkcd_bolds)

xkcd_muted = _xkcd_by_bold[len(_xkcd_by_bold) // 2:3 * len(_xkcd_by_bold) // 4]
_my_not_random.shuffle(xkcd_muted)

#random.shuffle(xkcd_colors)
enumerated_palettes[5] = xkcd_colors

# create a bigish list of greyish colors

# this gives us way more unique colors to cycle through in plots
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.rcsetup.cycler(color=xkcd_bolds)


def enumerated_colors(count: int,
                      palette: Union[int, List[Color]] = 1,
                      chain: bool = True) -> List[Color]:
    """Gets an enumerated list of count independent colors.

    Parameters
    ----------
    count : int
        Number of colors to return.
    palette : int or List[Color], optional
        Palette identifier (int) or explicit list of colors. Default is 1.
    chain : bool, optional
        If True, cycle through palette colors when count exceeds palette size.
        Default is True.

    Returns
    -------
    List[Color]
        List of colors from the specified palette.

    Raises
    ------
    ValueError
        If no enumerated palette of sufficient length exists and chain is False.
    """
    if isinstance(palette, int):
        p = enumerated_palettes[palette]
    else:
        p = palette

    if count <= len(p):
        return p[0:count]

    elif chain:
        def chain_iter(p):
            while True:
                for c in p:
                    yield c

        return [c for (i, c) in zip(range(count), chain_iter(p))]

    else:
        raise ValueError("No enumerated palettes of length {}.".format(count))


# black-zero jet is jet, but with the 0-value set to black, with an immediate jump to blue
def blackzerojet_cmap(data: np.ndarray) -> 'matplotlib.colors.LinearSegmentedColormap':
    """Create a jet colormap with zero values set to black.

    Parameters
    ----------
    data : np.ndarray
        Data array used to determine color scaling.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap with zero values as black.
    """
    blackzerojet_dict : SegmentData = {
        'blue':  [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.1, 1.0, 1.0),   (0.34, 1.0, 1.0),  (0.65, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'green': [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.125, 0.0, 0.0), (0.375, 1.0, 1.0), (0.64, 1.0, 1.0), (0.9, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'red':   [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.35, 0.0, 0.0),  (0.66, 1.0, 1.0),  (0.89, 1.0, 1.0), (1.0, 0.5, 0.5)]
    }
    minval = data[np.where(data > 0.)[0]].min()
    maxval = data[np.where(data > 0.)[0]].max()
    oneminval = .9 * minval / maxval

    for color in ['blue', 'green', 'red']:
        c = cast(Channel, color)
        blackzerojet_dict[c] = [(e[0] * (1-oneminval) + oneminval, e[1], e[2]) for e in blackzerojet_dict[c]]

    return matplotlib.colors.LinearSegmentedColormap('blackzerojet', blackzerojet_dict)


# ice color map
def ice_cmap() -> 'matplotlib.colors.LinearSegmentedColormap':
    """Create an ice-themed colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Ice-themed colormap from white to blue.
    """
    x = np.linspace(0, 1, 7)
    b = np.array([1, 1, 1, 1, 1, 0.8, 0.6])
    g = np.array([1, 0.993, 0.973, 0.94, 0.893, 0.667, 0.48])
    r = np.array([1, 0.8, 0.6, 0.5, 0.2, 0., 0.])

    bb = np.array([x, b, b]).transpose()
    gg = np.array([x, g, g]).transpose()
    rr = np.array([x, r, r]).transpose()
    ice_dict : SegmentData = {
        'blue': [tuple(e) for e in bb],
        'green': [tuple(e) for e in gg],
        'red': [tuple(e) for e in rr],
    }
    return matplotlib.colors.LinearSegmentedColormap('ice', ice_dict)


# water color map
def water_cmap() -> 'matplotlib.colors.LinearSegmentedColormap':
    """Create a water-themed colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Water-themed colormap from white to dark blue.
    """
    x = np.linspace(0, 1, 8)
    b = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    g = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    r = np.array([1.0, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])

    bb = np.array([x, b, b]).transpose()
    gg = np.array([x, g, g]).transpose()
    rr = np.array([x, r, r]).transpose()
    water_dict : SegmentData = {
        'blue': [tuple(e) for e in bb],
        'green': [tuple(e) for e in gg],
        'red': [tuple(e) for e in rr],
    }
    return matplotlib.colors.LinearSegmentedColormap('water', water_dict)


# water color map
def gas_cmap() -> 'matplotlib.colors.LinearSegmentedColormap':
    """Create a gas-themed colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Gas-themed colormap from white to red.
    """
    x = np.linspace(0, 1, 8)
    r = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    #    g = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    b = np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    g = np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])

    bb = np.array([x, b, b]).transpose()
    gg = np.array([x, g, g]).transpose()
    rr = np.array([x, r, r]).transpose()
    gas_dict : SegmentData = {
        'blue': [tuple(e) for e in bb],
        'green': [tuple(e) for e in gg],
        'red': [tuple(e) for e in rr],
    }
    return matplotlib.colors.LinearSegmentedColormap('gas', gas_dict)


# jet-by-index
def cm_mapper(
    vmin: float = 0.,
    vmax: float = 1.,
    cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    norm: Optional[matplotlib.colors.Normalize] = None,
) -> Callable[[float], Tuple[float, float, float, float]]:
    """Provide a function that maps scalars to colors in a given colormap.

    Parameters
    ----------
    vmin, vmax : scalar
      Min and max scalars to be mapped.
    cmap : str or matplotlib.colors.Colormap instance
      The colormap to discretize.
    norm : optional, matplotlib Norm object
      A normalization.

    Returns
    -------
    Function, cmap(scalar) -> (r,g,b,a)

    Example
    -------
    .. code:: python
    
        # plot 4 lines
        x = np.arange(10)
        cm = cm_mapper(0,3,'jet')
        for i in range(4):
            plt.plot(x, x**i, color=cm(i))

    """
    if cmap is None:
        cmap = plt.get_cmap('jet')
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    sm = matplotlib.cm.ScalarMappable(norm, cmap)

    def mapper(value):
        return sm.to_rgba(value)
    return mapper


def cm_discrete(
    ncolors: int,
    cmap: Union[str, matplotlib.colors.Colormap] = plt.get_cmap('jet')
) -> 'matplotlib.colors.LinearSegmentedColormap':
    """Calculate a discrete colormap with N entries from the continuous colormap cmap.

    Parameters
    ----------
    ncolors : int
      Number of colors.
    cmap : str or matplotlib.colors.Colormap instance, optional
      The colormap to discretize.  Default is 'jet'.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap instance

    Example
    -------
    .. code:: python

        # plot 4 lines
        x = np.arange(10)
        colors = cmap_discretize('jet', 4)
        for i in range(4):
            plt.plot(x, x**i, color=colors[i])

    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., ncolors), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., ncolors + 1)

    cdict : SegmentData = {
        'blue': [(indices[i], colors_rgba[i - 1, 0], colors_rgba[i, 0]) for i in range(ncolors + 1)],
        'green': [(indices[i], colors_rgba[i - 1, 1], colors_rgba[i, 1]) for i in range(ncolors + 1)],
        'red': [(indices[i], colors_rgba[i - 1, 2], colors_rgba[i, 2]) for i in range(ncolors + 1)],
    }

    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%ncolors, cdict, 1024)


def desaturate(color: Color,
               amount: float = 0.4,
               is_hsv: bool = False) -> np.ndarray:
    """Desaturate a color by reducing its saturation.

    Parameters
    ----------
    color : Color or np.ndarray
        Color to desaturate. Can be in RGB or HSV format.
    amount : float, optional
        Amount of desaturation to apply (0-1). Default is 0.4.
    is_hsv : bool, optional
        If True, input color is in HSV format. Default is False (RGB).

    Returns
    -------
    np.ndarray
        Desaturated color in RGB format.
    """
    if not is_hsv:
        hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))
    else:
        hsv = color

    hsv[1] = max(0, hsv[1] - amount)
    return matplotlib.colors.hsv_to_rgb(hsv)


def darken(color: Color, fraction: float = 0.6) -> Tuple[float, float, float]:
    """Darken a color by reducing its brightness.

    Parameters
    ----------
    color : Color
        Color to darken.
    fraction : float, optional
        Fraction of brightness to remove (0-1). Default is 0.6.

    Returns
    -------
    Tuple[float, float, float]
        Darkened color in RGB format.
    """
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.maximum(rgb - fraction*rgb, 0))


def lighten(color: Color, fraction: float = 0.6) -> Tuple[float, float, float]:
    """Lighten a color by increasing its brightness.

    Parameters
    ----------
    color : Color
        Color to lighten.
    fraction : float, optional
        Fraction of brightness to add (0-1). Default is 0.6.

    Returns
    -------
    Tuple[float, float, float]
        Lightened color in RGB format.
    """
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.minimum(rgb + fraction * (1-rgb), 1))


def createIndexedColormap(
    indices: Union[List[int], np.ndarray],
    cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
) -> Tuple[List[int], 'matplotlib.colors.ListedColormap', 'matplotlib.colors.BoundaryNorm',
           List[int], List[str]]:
    """Generates an indexed colormap and labels for imaging, e.g. soil indices.
    Parameters
    ----------
    indices : iterable(int)
        Collection of indices that will be used in this colormap.
    cmap : optional, str
        Name of matplotlib colormap to sample.

    Returns
    -------
    indices_out : list(int)
        The unique, sorted list of indices found.
    cmap : cmap-type
        A linestringed map for use with plots.
    norm : BoundaryNorm
        A norm for use in `plot_trisurf()` or other plotting methods
        to ensure correct NLCD colors.
    ticks : list(int)
        A list of tick locations for the requested indices.  For use
        with `set_ticks()`.
    labels : list(str)
        A list of labels associated with the ticks.  For use with
        `set_{x,y}ticklabels()`.
    """
    indices = sorted(set(indices))

    cm_values = None
    if cmap is None:
        for i, palette in enumerated_palettes.items():
            if len(indices) < len(palette):
                cm_values = enumerated_colors(len(indices), palette=i)
        if cm_values is None:
            cmap = 'gist_rainbow'
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        
    if cm_values is None:
        cm = cm_mapper(0, len(indices) - 1, cmap)
        cm_values = [cm(i) for i in range(0, len(indices))]

    cmap = matplotlib.colors.ListedColormap(cm_values)
    ticks = indices + [indices[-1] + 1, ]
    norm = matplotlib.colors.BoundaryNorm(ticks, len(ticks) - 1)
    labels = [str(i) for i in indices]
    return indices, cmap, norm, ticks, labels

_doc_template = \
"""Generates a colormap and labels for imaging with the {label} colors.

Parameters
----------
indices : iterable(int), optional
    Collection of {label} indices that will be used in this colormap.
    If None (default), uses all {label} indices.

Returns
-------
indices_out : list(int)
    The unique, sorted list of indices found.
cmap : cmap-type
    A linestringed map for use with plots.
norm : BoundaryNorm
    A norm for use in `plot_trisurf()` or other plotting methods
    to ensure correct {label} colors.
ticks : list(int)
    A list of tick locations for the requested indices.  For use
    with `set_ticks()`.
labels : list(str)
    A list of labels associated with the ticks.  For use with
    `set_{{x,y}}ticklabels()`.
formatted: bool, 
    To make the labels formatted nicely (i.e. add newline in long label names)

Example
-------

Plot a triangulation given a set of {label} colors on those triangles.

Given a triangluation `mesh_points, mesh_tris` and {label} color
indices for each triangle (tri_{label_lower}):

.. code-block:: 

    indices, cmap, norm, ticks, labels = generate_{label_lower}_colormap(set(tri_{label_lower}))

    mp = ax.plot_trisurf(mesh_points[:,0], mesh_points[:,1], mesh_points[:,2], 
            triangles=mesh_tris, color=tri_{label_lower}, 
            cmap=cmap, norm=norm)
    cb = fig.colorbar(mp, orientation='horizontal')
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels(labels, rotation=45)

"""


def _createColormapCreator(
    label: str, all_colors: Dict[int, Tuple[str, Color]]
) -> Callable[[Optional[List[int]], bool],
              Tuple[List[int], 'matplotlib.colors.ListedColormap', 'matplotlib.colors.BoundaryNorm',
                    List[float], List[str]]]:
    """Create a colormap creator function for specific color sets.

    Parameters
    ----------
    label : str
        Label for the colormap type (e.g., 'NLCD', 'MODIS').
    all_colors : Dict[int, Tuple[str, Color]]
        Dictionary mapping indices to (name, color) tuples.

    Returns
    -------
    Callable
        Function that creates colormaps for the specified color set.
    """
    def _createColormap(
        indices: Optional[List[int]] = None,
        formatted: bool = False
    ) -> Tuple[List[int], 'matplotlib.colors.ListedColormap', 'matplotlib.colors.BoundaryNorm',
               List[float], List[str]]:
        """Create colormap for specified indices.

        Parameters
        ----------
        indices : List[int], optional
            List of color indices to include. If None, uses all available.
        formatted : bool, optional
            If True, format long labels with line breaks. Default is False.

        Returns
        -------
        Tuple[List[int], matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm, List[float], List[str]]
            Tuple containing indices, colormap, normalization, tick positions, and labels.
        """
        if indices is None:
            indices = list(all_colors.keys())

        indices = sorted(set(indices))

        print("making colormap with:", indices)
        values = [all_colors[k][1] for k in indices]
        print("making colormap with colors:", values)
        cmap = matplotlib.colors.ListedColormap(values)
        ticks = [i - 0.5 for i in indices] + [indices[-1] + 0.5, ]
        norm = matplotlib.colors.BoundaryNorm(ticks, len(ticks) - 1)
        labels = [all_colors[k][0] for k in indices]

        if formatted:
            nlcd_labels_fw = []
            for label in labels:
                label_fw = label
                if len(label) > 15:
                    if ' ' in label:
                        lsplit = label.split()
                        if len(lsplit) == 2:
                            label_fw = '\n'.join(lsplit)
                        elif len(lsplit) == 4:
                            label_fw = '\n'.join([' '.join(lsplit[0:2]), ' '.join(lsplit[2:])])
                        elif len(lsplit) == 3:
                            if len(lsplit[0]) > len(lsplit[-1]):
                                label_fw = '\n'.join([lsplit[0], ' '.join(lsplit[1:])])
                            else:
                                label_fw = '\n'.join([' '.join(lsplit[:-1]), lsplit[-1]])
                nlcd_labels_fw.append(label_fw)

            labels = nlcd_labels_fw

        return indices, cmap, norm, ticks, labels

    doc = _doc_template.format(label=label, label_lower=label.lower())
    _createColormap.__doc__ = doc
    return _createColormap


import watershed_workflow.sources.manager_nlcd

createNLCDColormap = _createColormapCreator('NLCD', watershed_workflow.sources.manager_nlcd.colors)

import watershed_workflow.sources.manager_modis_appeears

createMODISColormap = _createColormapCreator(
    'MODIS', watershed_workflow.sources.manager_modis_appeears.colors)


def createIndexedColorbar(ncolors: int,
                          cmap: matplotlib.colors.Colormap,
                          labels: Optional[List[str]] = None,
                          **kwargs: Any) -> 'matplotlib.colorbar.Colorbar':
    """Add an indexed colorbar based on a given colormap.

    This sets ticks in the middle of each color range, adds the
    colorbar, and sets the labels if provided.

    Parameters
    ----------
    ncolors : int
      Number of colors to display.
    cmap : matplotlib.colors.Colormap instance
      The colormap used in the image.
    labels : list
      Optional list of label strings that equal to the number of
      colors. If not provided, labels are set to range(ncolors).
    kwargs : dict
      Other arguments are passed on to the plt.colorbar() call, which
      can be used for things like fraction, pad, etc to control the
      location/spacing of the colorbar.

    Returns
    -------
    colorbar : the colorbar object

    """
    if labels is not None:
        assert (len(labels) == ncolors)

    cmap = cm_discrete(ncolors, cmap)
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)

    if 'fraction' not in kwargs: kwargs['fraction'] = 0.03
    if 'pad' not in kwargs: kwargs['pad'] = 0.04

    colorbar = plt.colorbar(mappable, **kwargs)
    ticks = np.linspace(0, ncolors, ncolors)
    colorbar.set_ticks(list(ticks))  # set tick locations

    # set tick labels
    if labels is not None:
        assert (len(labels) == len(ticks))
        colorbar.set_ticklabels(labels)
    else:
        colorbar.set_ticklabels([str(i) for i in range(ncolors)])
    return colorbar
