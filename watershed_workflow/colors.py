import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import numpy as np
import collections

#
# Lists of disparate color palettes
#
enumerated_palettes = {
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


def enumerated_colors(count, palette=1, chain=True):
    """Gets an enumerated list of count independent colors."""
    if isinstance(palette, int):
        p = enumerated_palettes[palette]
    else:
        p = palette

    if count <= len(p):
        return p[0:count]
    else:
        for p in enumerated_palettes.values():
            if count <= len(p):
                return p[0:count]

    if chain:
        # must chain...
        p = enumerated_palettes[palette]

        def chain_iter(p):
            while True:
                for c in p:
                    yield c

        return [c for (i, c) in zip(range(count), chain_iter(p))]

    else:
        raise ValueError("No enumerated palettes of length {}.".format(count))


# black-zero jet is jet, but with the 0-value set to black, with an immediate jump to blue
def blackzerojet_cmap(data):
    blackzerojet_dict = {
        'blue': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.11, 1, 1], [0.34000000000000002, 1, 1],
                 [0.65000000000000002, 0, 0], [1, 0, 0]],
        'green': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.125, 0, 0], [0.375, 1, 1],
                  [0.64000000000000001, 1, 1], [0.91000000000000003, 0, 0], [1, 0, 0]],
        'red': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.34999999999999998, 0, 0],
                [0.66000000000000003, 1, 1], [0.89000000000000001, 1, 1], [1, 0.5, 0.5]]
    }
    minval = data[np.where(data > 0.)[0]].min()
    print(minval)
    maxval = data[np.where(data > 0.)[0]].max()
    print(maxval)
    oneminval = .9 * minval / maxval
    for color in ['blue', 'green', 'red']:
        for i in range(1, len(blackzerojet_dict[color])):
            blackzerojet_dict[color][i][0] = blackzerojet_dict[color][i][0] * (
                1-oneminval) + oneminval

    return matplotlib.colors.LinearSegmentedColormap('blackzerojet', blackzerojet_dict)


# ice color map
def ice_cmap():
    x = np.linspace(0, 1, 7)
    b = np.array([1, 1, 1, 1, 1, 0.8, 0.6])
    g = np.array([1, 0.993, 0.973, 0.94, 0.893, 0.667, 0.48])
    r = np.array([1, 0.8, 0.6, 0.5, 0.2, 0., 0.])

    bb = np.array([x, b, b]).transpose()
    gg = np.array([x, g, g]).transpose()
    rr = np.array([x, r, r]).transpose()
    ice_dict = { 'blue': bb, 'green': gg, 'red': rr }
    return matplotlib.colors.LinearSegmentedColormap('ice', ice_dict)


# water color map
def water_cmap():
    x = np.linspace(0, 1, 8)
    b = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    g = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    r = np.array([1.0, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])

    bb = np.array([x, b, b]).transpose()
    gg = np.array([x, g, g]).transpose()
    rr = np.array([x, r, r]).transpose()
    water_dict = { 'blue': bb, 'green': gg, 'red': rr }
    return matplotlib.colors.LinearSegmentedColormap('water', water_dict)


# water color map
def gas_cmap():
    x = np.linspace(0, 1, 8)
    r = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    #    g = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    b = np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    g = np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])

    bb = np.array([x, b, b]).transpose()
    gg = np.array([x, g, g]).transpose()
    rr = np.array([x, r, r]).transpose()
    gas_dict = { 'blue': bb, 'green': gg, 'red': rr }
    return matplotlib.colors.LinearSegmentedColormap('gas', gas_dict)


# jet-by-index
def cm_mapper(vmin=0., vmax=1., cmap=None, norm=None, get_sm=False):
    """Provide a function that maps scalars to colors in a given colormap.

    Parameters
    ----------
    vmin, vmax : scalar
      Min and max scalars to be mapped.
    cmap : str or matplotlib.cmap instance
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
        cmap = matplotlib.cm.jet
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    sm = matplotlib.cm.ScalarMappable(norm, cmap)

    def mapper(value):
        return sm.to_rgba(value)

    if get_sm:
        return mapper, sm
    else:
        return mapper


def cm_discrete(ncolors, cmap=matplotlib.cm.jet):
    """Calculate a discrete colormap with N entries from the continuous colormap cmap.

    Parameters
    ----------
    ncolors : int
      Number of colors.
    cmap : str or matplotlib.cmap instance, optional
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
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., ncolors), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., ncolors + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                      for i in range(ncolors + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%ncolors, cdict, 1024)


def float_list_type(mystring):
    """Convert string-form list of doubles into list of doubles."""
    colors = []
    for f in mystring.strip("(").strip(")").strip("[").strip("]").split(","):
        try:
            colors.append(float(f))
        except:
            colors.append(f)
    return colors


def desaturate(color, amount=0.4, is_hsv=False):
    if not is_hsv:
        hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))
    else:
        hsv = color

    hsv[1] = max(0, hsv[1] - amount)
    return matplotlib.colors.hsv_to_rgb(hsv)


def darken(color, fraction=0.6):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.maximum(rgb - fraction*rgb, 0))


def lighten(color, fraction=0.6):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.minimum(rgb + fraction * (1-rgb), 1))


def generate_indexed_colormap(indices, cmap=None):
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
        A segmented map for use with plots.
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
    if cm_values is None:
        cm = cm_mapper(0, len(indices) - 1, cmap=matplotlib.cm.get_cmap(cmap))
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
    A segmented map for use with plots.
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


def _indexed_colormap(label, all_colors):
    def _generate_colormap(indices=None, formatted=False):
        if indices is None:
            indices = list(all_colors.keys())

        indices = sorted(set(indices))

        values = [all_colors[k][1] for k in indices]
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
    _generate_colormap.__doc__ = doc
    return _generate_colormap


import watershed_workflow.sources.manager_nlcd

generate_nlcd_colormap = _indexed_colormap('NLCD', watershed_workflow.sources.manager_nlcd.colors)

import watershed_workflow.sources.manager_modis_appeears

generate_modis_colormap = _indexed_colormap(
    'MODIS', watershed_workflow.sources.manager_modis_appeears.colors)


def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """Add an indexed colorbar based on a given colormap.

    This sets ticks in the middle of each color range, adds the
    colorbar, and sets the labels if provided.

    Parameters
    ----------
    ncolors : int
      Number of colors to display.
    cmap : matplotlib.cmap instance
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
    colorbar.set_ticks(ticks)  # set tick locations
    # set tick labels
    if labels is not None:
        assert (len(labels) == len(ticks))
        colorbar.set_ticklabels(labels)
    else:
        colorbar.set_ticklabels(range(ncolors))
    return colorbar
