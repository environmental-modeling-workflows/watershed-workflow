"""Shared plotting helpers for bin/ scripts."""
import logging
from matplotlib import pyplot as plt

import watershed_workflow.plot.plot
import watershed_workflow.utils.config


def plot_with_dem(args,
                  hucs,
                  reaches,
                  dem,
                  profile,
                  shape_color='k',
                  river_color='white',
                  cb=True,
                  cb_label='elevation [m]',
                  vmin=None,
                  vmax=None,
                  fig=None,
                  ax=None):
    logging.info('Plotting')
    logging.info('--------')

    if fig is None:
        fig = plt.figure(figsize=args.figsize)
    if ax is None:
        ax = watershed_workflow.plot.plot.get_ax(args.projection, fig=fig)

    if args.extent is None:
        args.extent = hucs.exterior().bounds

        if args.pad_fraction is not None:
            if len(args.pad_fraction) == 1:
                dxp = (args.extent[2] - args.extent[0]) * args.pad_fraction[0]
                dxm = dxp
                dym = dxp
                dyp = dxp
            elif len(args.pad_fraction) == 2:
                dxp = (args.extent[2] - args.extent[0]) * args.pad_fraction[0]
                dxm = dxp
                dyp = (args.extent[3] - args.extent[1]) * args.pad_fraction[1]
                dym = dyp
            elif len(args.pad_fraction) == 4:
                dxm = (args.extent[2] - args.extent[0]) * args.pad_fraction[0]
                dym = (args.extent[3] - args.extent[1]) * args.pad_fraction[1]
                dxp = (args.extent[2] - args.extent[0]) * args.pad_fraction[2]
                dyp = (args.extent[3] - args.extent[1]) * args.pad_fraction[3]
            else:
                raise ValueError('Option: --pad-fraction must be of length 1, 2, or 4')

            args.extent = [
                args.extent[0] - dxm, args.extent[1] - dym,
                args.extent[2] + dxp, args.extent[3] + dyp
            ]

    logging.info('plot extent: {}'.format(args.extent))

    if args.basemap:
        watershed_workflow.plot.plot.basemap(args.projection,
                                           ax=ax,
                                           resolution=args.basemap_resolution,
                                           land_kwargs={'zorder': 0},
                                           ocean_kwargs={'zorder': 2})

    if dem is not None:
        mappable = watershed_workflow.plot.plot.dem(profile, dem, ax, vmin, vmax)
        if args.basemap:
            mappable.set_zorder(1)
        if cb:
            cb = fig.colorbar(mappable, orientation="horizontal", pad=0)
            cb.set_label(cb_label)

    if reaches is not None:
        watershed_workflow.plot.plot.river(reaches, args.projection, river_color, ax,
                                         linewidth=0.5, zorder=3)

    if hucs is not None:
        watershed_workflow.plot.plot.hucs(hucs, args.projection, shape_color, ax,
                                        linewidth=.7, zorder=4)

    ax.set_xlim(args.extent[0], args.extent[2])
    ax.set_ylim(args.extent[1], args.extent[3])
    ax.set_aspect('equal', 'box')
    ax.set_title(args.title)
    return fig, ax
