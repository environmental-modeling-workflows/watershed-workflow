"""Interactive (ipywidgets-based) plotting helpers for Jupyter notebooks.

This module isolates the ipywidgets dependency from the rest of the
plot/ subpackage, which otherwise contains only static, one-shot
plotting functions. Use these when you want a plot that updates live
as the user changes an input (e.g. exploring a mesh cell-by-cell),
rather than a function that returns a finished figure.
"""
from typing import Optional, Any, Tuple, TYPE_CHECKING
import xarray

import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt

import watershed_workflow.plot.plot

if TYPE_CHECKING:
    from watershed_workflow.mesh.mesh import Mesh2D

__all__ = ['CellContextWidget']


class CellContextWidget:
    """Interactive version of plotCellContext(), updating live from widget input.

    Displays text boxes for cell, edge (as two vertex ids), and
    coordinate (x, y), plus an "update" button. Whichever box the user
    last typed into is used to resolve the plotted cell on the next
    update, via the same rules as watershed_workflow.plot.plot.findCell().
    (The cell box is also updated to reflect the resolved cell after
    each redraw, for visibility -- this does not count as "last
    edited" by the user.)

    Parameters
    ----------
    m2 : Mesh2D
        The mesh, with current elevations in m2.coords[:,2].
    cell : int, optional
        Initial cell to display. Defaults to 0 if no other input given.
    context_rings : int, optional
        Number of rings of neighbors for context. Default is 3.
    dem : xr.DataArray, optional
        A DEM (same CRS as m2) to show cropped to the local context.
    dem_sm : xr.DataArray, optional
        A second DEM (e.g. smoothed), shown in an additional panel.

    Usage
    -----
    In a Jupyter notebook::

        w = CellContextWidget(m2, dem=dem, dem_sm=dem_sm)
        w.display()

    """
    def __init__(self,
                m2 : 'Mesh2D',
                cell : Optional[int] = None,
                context_rings : int = 3,
                dem : Optional[xarray.DataArray] = None,
                dem_sm : Optional[xarray.DataArray] = None,
                ):
        self.m2 = m2
        self.context_rings = context_rings
        self.dem = dem
        self.dem_sm = dem_sm

        self.cell_input = widgets.Text(description='cell:',
                                       value='' if cell is None else str(cell))
        self.edge_input = widgets.Text(description='edge (v0,v1):', value='')
        self.coordinate_input = widgets.Text(description='coordinate (x,y):', value='')
        self.rings_input = widgets.IntText(description='context rings:', value=context_rings)
        self.update_button = widgets.Button(description='Update')
        self.output = widgets.Output()

        self.update_button.on_click(self._onUpdate)

        # track which field the user last typed into, so a stale value
        # reflected into cell_input after a redraw doesn't shadow a
        # freshly-entered edge/coordinate
        self._last_edited = 'cell'
        self._reflecting = False
        self.cell_input.observe(self._makeOnEdit('cell'), names='value')
        self.edge_input.observe(self._makeOnEdit('edge'), names='value')
        self.coordinate_input.observe(self._makeOnEdit('coordinate'), names='value')

        self.fig = None
        self.ax = None
        self.current_cell = cell if cell is not None else 0

    def _makeOnEdit(self, field : str):
        def _onEdit(change):
            if not self._reflecting:
                self._last_edited = field
        return _onEdit

    def _parseInputs(self) -> dict:
        """Determine which of cell/edge/coordinate to use, based on the
        field the user most recently typed into."""
        if self._last_edited == 'edge' and self.edge_input.value.strip():
            v0, v1 = (int(v) for v in self.edge_input.value.split(','))
            return { 'edge': (v0, v1) }
        elif self._last_edited == 'coordinate' and self.coordinate_input.value.strip():
            x, y = (float(v) for v in self.coordinate_input.value.split(','))
            return { 'coordinate': (x, y) }
        elif self.cell_input.value.strip():
            return { 'cell': int(self.cell_input.value) }
        else:
            return { 'cell': self.current_cell }

    def _onUpdate(self, _button) -> None:
        self.redraw()

    def redraw(self) -> None:
        """Re-resolve the cell from current widget input and redraw the plot."""
        kwargs = self._parseInputs()
        context_rings = self.rings_input.value

        with self.output:
            self.output.clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)
            self.fig, self.ax, self.current_cell = watershed_workflow.plot.plot.plotCellContext(
                self.m2, context_rings=context_rings, dem=self.dem, dem_sm=self.dem_sm, **kwargs)
            # reflect the resolved cell back into the cell box, so
            # edge/coordinate lookups are visible and re-editable; guard
            # with _reflecting so this programmatic update isn't mistaken
            # for the user editing the cell box themselves
            self._reflecting = True
            try:
                self.cell_input.value = str(self.current_cell)
            finally:
                self._reflecting = False
            plt.show()

    def display(self) -> None:
        """Display the widget controls and the initial plot."""
        controls = widgets.VBox([
            widgets.HBox([self.cell_input, self.edge_input, self.coordinate_input]),
            widgets.HBox([self.rings_input, self.update_button]),
        ])
        display(controls, self.output)
        self.redraw()
