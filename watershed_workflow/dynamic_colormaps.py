"""
Dynamic colormap scaling for matplotlib images with shared axes.

Provides functionality to automatically adjust colormap limits (vmin/vmax) based on
the currently visible region of data, useful for multi-panel plots with shared axes.

Supports multiple data types through a wrapper abstraction:
- xarray DataArray and Dataset
- GeoPandas GeoDataFrame
- matplotlib Collections (PolyCollection, LineCollection, PathCollection)

Type Definitions
----------------
WrapperArgs : Union[Collection, Tuple[Any], List[Any], Dict[str, Any]]
    Arguments to pass to DataWrapperFactory.wrap() to create a DataWrapper instance.
    Can be:
    - A Collection object (self-contained with geometry and values)
    - A tuple/list of positional arguments (data, image, ...) for xarray/GeoDataFrame
    - A dict of keyword arguments (data=..., image=..., ...)

WrapperGroup : Tuple[List[WrapperArgs], bool]
    A group of WrapperArgs that share the same colormap range, plus a boolean
    flag indicating whether limits should be symmetric around 0.

Extending with Custom Types
----------------------------
To add support for a custom data type, create a DataWrapper subclass and register it:

    from dynamic_colormaps import DataWrapper, DataWrapperFactory

    class DataWrapperMyType(DataWrapper):
        def __init__(self, my_data, my_image):
            self.data = my_data
            self.image = my_image

        def subset(self, xlim, ylim):
            # Subset logic for your type
            subsetted = self.data[...]
            return DataWrapperMyType(subsetted, self.image)

        def getLimits(self):
            return float(self.data.min()), float(self.data.max())

        def isEmpty(self):
            return len(self.data) == 0

        def setLimits(self, vmin, vmax):
            self.image.set_clim(vmin, vmax)

    # Register it
    DataWrapperFactory.register(MyDataType, DataWrapperMyType)

    # Now use it normally with createDynamicScaler
    wrapper_groups = [
        ([(my_data, my_image), (my_data2, my_image2)], False),
    ]
    scaler = createDynamicScaler(wrapper_groups, ax)

Colorbar Updates
----------------
Colorbars created with plt.colorbar(img, ax=ax) will update automatically when the
scaler calls img.set_clim(). The colorbar is connected to the image's colormap
normalization, so changes to vmin/vmax propagate automatically.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Iterable, Optional, Any, List, Dict
import numpy as np
import xarray as xr
import geopandas as gpd
from matplotlib.collections import Collection, PolyCollection, LineCollection, PathCollection
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from shapely.geometry import box, Polygon, LineString, Point

# Type definitions
#
# Arguments to a DataWrapper constructor, as a single argument, a
# Tuple or List of args, or a Dict of kwargs.  The first argument (or
# the data argument for kwargs) is used to determine the type.
WrapperArgs = Union[Collection, Tuple[Any], List[Any], Dict[str, Any]]

# Arguments for a group of wrappers, plus the symmetric flag.
WrapperGroup = Tuple[List[WrapperArgs], bool]



class DataWrapper(ABC):
    """
    Abstract base class for data wrappers.

    Wraps different data types to provide a uniform interface for subsetting
    and computing min/max values.
    """

    @abstractmethod
    def subset(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> 'DataWrapper':
        """
        Return a new wrapper containing only data within the specified bounds.

        Parameters
        ----------
        xlim : Tuple[float, float]
            (xmin, xmax) bounds
        ylim : Tuple[float, float]
            (ymin, ymax) bounds

        Returns
        -------
        DataWrapper
            New DataWrapper instance with subsetted data
        """
        pass

    @abstractmethod
    def getLimits(self) -> Tuple[float, float]:
        """
        Compute min and max values of the data.

        Returns
        -------
        Tuple[float, float]
            (vmin, vmax)
        """
        pass

    @abstractmethod
    def isEmpty(self) -> bool:
        """
        Check if the wrapper contains any data.

        Returns
        -------
        bool
            True if empty, False otherwise
        """
        pass

    @abstractmethod
    def setLimits(self, vmin : float, vmax : float):
        """
        Sets the color limits of the objects or image being displayed.
        """
        pass



class DataWrapperXarray(DataWrapper):
    """
    DataWrapper for xarray DataArray and Dataset objects.

    Provides subsetting via coordinate-based selection and min/max computation.
    """

    def __init__(self, data: Union[xr.DataArray, xr.Dataset], image: AxesImage, 
                 var_name: Optional[str] = None,
                 x_name: str = 'x', y_name: str = 'y'):
        """
        Parameters
        ----------
        data : Union[xr.DataArray, xr.Dataset]
            DataArray or Dataset with coordinates
        image : AxesImage
            The return value of data.plot.imshow()
        var_name : str, optional
            For Dataset, the variable name to extract. If None and data is a Dataset,
            defaults to the first data variable.
        x_name : str, optional
            Name of the x coordinate. Default is 'x'.
        y_name : str, optional
            Name of the y coordinate. Default is 'y'.
        """
        if isinstance(data, xr.Dataset):
            if var_name is None:
                raise ValueError('When providing a Dataset, must also provide var_name option specifying what variable is plotted.')
            self.data = data[var_name]
        elif isinstance(data, xr.DataArray):
            self.data = data
        else:
            raise TypeError(f"Expected xarray.DataArray or Dataset, got {type(data)}")
        
        self.x_name = x_name
        self.y_name = y_name
        self.image = image

    def subset(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> DataWrapper:
        """
        Subset data to visible region using coordinate-based selection.

        Parameters
        ----------
        xlim : Tuple[float, float]
            (xmin, xmax) bounds
        ylim : Tuple[float, float]
            (ymin, ymax) bounds

        Returns
        -------
        DataWrapper
            New DataWrapperXarray with subsetted data

        Raises
        ------
        KeyError
            If coordinates are not present
        """
        try:
            subsetted = self.data.sel(
                {self.x_name: slice(xlim[0], xlim[1]),
                 self.y_name: slice(ylim[0], ylim[1])}
            )
            wrapper = DataWrapperXarray(subsetted, self.image, x_name=self.x_name, y_name=self.y_name)
            return wrapper
        except KeyError as e:
            raise KeyError(f"Could not subset xarray with coordinates: {e}")

    def getLimits(self) -> Tuple[float, float]:
        """
        Returns
        -------
        Tuple[float, float]
            (vmin, vmax)
        """
        vmin = float(self.data.min())
        vmax = float(self.data.max())
        return vmin, vmax

    def isEmpty(self) -> bool:
        """
        Returns
        -------
        bool
            True if size is 0
        """
        return self.data.size == 0

    def setLimits(self, vmin : float, vmax : float) -> None:
        """Sets the color limits."""
        self.image.set_clim(vmin, vmax)


class DataWrapperGeoDataFrame(DataWrapper):
    """
    Wrapper for GeoPandas GeoDataFrame objects.

    Provides subsetting based on intersection with bounding box and min/max computation
    from a value column.
    """

    def __init__(self, gdf: gpd.GeoDataFrame, collection: Collection,
                 value_column: str = 'value'):
        """
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with geometry column
        collection : Collection
            The drawn shapes, return value of gdf.plot(...)
        value_column : str, optional
            Name of column containing values for min/max computation. Default is 'value'.
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(f"Expected GeoDataFrame, got {type(gdf)}")

        if value_column not in gdf.columns:
            raise ValueError(f"Column '{value_column}' not found in GeoDataFrame")

        self.gdf = gdf
        self.value_column = value_column
        self.collection = collection

    def subset(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> DataWrapper:
        """
        Subset by geometry intersection with bounding box.

        Parameters
        ----------
        xlim : Tuple[float, float]
            (xmin, xmax) bounds
        ylim : Tuple[float, float]
            (ymin, ymax) bounds

        Returns
        -------
        DataWrapper
            New DataWrapperGeoDataFrame with subsetted data
        """
        bbox = box(xlim[0], ylim[0], xlim[1], ylim[1])
        mask = self.gdf.geometry.intersects(bbox)
        subsetted = self.gdf[mask]
        return DataWrapperGeoDataFrame(subsetted, self.collection, self.value_column)

    def getLimits(self) -> Tuple[float, float]:
        """
        Returns
        -------
        Tuple[float, float]
            (vmin, vmax) from value_column
        """
        if self.isEmpty():
            raise ValueError("Cannot compute limits from empty GeoDataFrame")
        vmin = float(self.gdf[self.value_column].min())
        vmax = float(self.gdf[self.value_column].max())
        return vmin, vmax

    def setLimits(self, vmin : float, vmax : float) -> None:
        """Sets the color limits."""
        self.collection.set_clim(vmin, vmax)

    def isEmpty(self) -> bool:
        """
        Returns
        -------
        bool
            True if no rows
        """
        return len(self.gdf) == 0



class DataWrapperCollection(DataWrapperGeoDataFrame):
    """
    DataWrapper for matplotlib Collection objects (PolyCollection, LineCollection, PathCollection).

    Converts collection to GeoDataFrame internally for efficient spatial subsetting.
    Inherits all subsetting and limit computation from DataWrapperGeoDataFrame.
    """

    def __init__(self, collection: Collection, values: Optional[Iterable[float]] = None):
        """
        Parameters
        ----------
        collection : matplotlib.collections.Collection
            A PolyCollection, LineCollection, or PathCollection
        values : Iterable[float], optional
            Values corresponding to each element in the collection.
            If None, attempts to extract from collection.get_array().
            Length must match number of paths in collection.
        """
        if not isinstance(collection, Collection):
            raise TypeError(f"Expected Collection, got {type(collection)}")

        # Try to get values from collection if not provided
        if values is None:
            values = collection.get_array()
            if values is None:
                raise ValueError(
                    "No values provided and collection.get_array() returned None. "
                    "Please provide values explicitly."
                )

        values_array = np.asarray(values)

        # Convert paths to geometries
        geometries = []
        if isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                vertices = path.vertices

                # Close the polygon if not already closed
                if not np.allclose(vertices[0], vertices[-1]):
                    vertices = np.vstack([vertices, vertices[0]])
                geometries.append(Polygon(vertices))
                
        elif isinstance(collection, LineCollection):
            for path in collection.get_paths():
                vertices = path.vertices
                geometries.append(LineString(vertices))

        elif isinstance(collection, PathCollection):
            # PathCollection stores points; create point geometries
            offsets = collection.get_offsets().data
            geometries = [Point(v) for v in offsets]
        else:
            raise TypeError(f"Unsupported collection type: {type(collection)}")

        if len(values_array) != len(geometries):
            raise ValueError(
                f"Number of values ({len(values_array)}) must match "
                f"number of geometries ({len(geometries)})"
            )

        # Create GeoDataFrame and initialize parent
        gdf = gpd.GeoDataFrame(
            {'values': values_array},
            geometry=geometries
        )
        super().__init__(gdf, collection, 'values')


class DataWrapperFactory:
    """
    Factory for creating appropriate wrapper instances.
    """

    _wrappers = {
        xr.DataArray: DataWrapperXarray,
        xr.Dataset: DataWrapperXarray,
        gpd.GeoDataFrame: DataWrapperGeoDataFrame,
        Collection: DataWrapperCollection
    }

    @classmethod
    def wrap(cls, data, *args, **kwargs) -> DataWrapper:
        """
        Create a wrapper for the given data type.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset | Collection | GeoDataFrame
            The data to wrap.
        *args, **kwargs:
            All other args are passed to the DataWrapper constructor.

        Returns
        -------
        DataWrapper
            Appropriate DataWrapper subclass instance

        Raises
        ------
        TypeError
            If data type is not supported
        """
        # Handle registered types
        for dtype in cls._wrappers.keys():
            if isinstance(data, dtype):
                return cls._wrappers[dtype](data, *args, **kwargs)

        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Supported types: xarray.DataArray, xarray.Dataset, "
            "geopandas.GeoDataFrame, matplotlib.collection.Collection"
        )

    @classmethod
    def register(cls, data_type: type, wrapper_class: type) -> None:
        """
        Register a new data type and its wrapper.

        Parameters
        ----------
        data_type : type
            The data type to support
        wrapper_class : type
            The DataWrapper subclass to use

        Raises
        ------
        TypeError
            If wrapper_class is not a DataWrapper subclass
        """
        if not issubclass(wrapper_class, DataWrapper):
            raise TypeError(
                f"wrapper_class must be a DataWrapper subclass, "
                f"got {wrapper_class}"
            )
        cls._wrappers[data_type] = wrapper_class


class DynamicColormapScaler:
    """
    Manages dynamic vmin/vmax scaling for groups of matplotlib images.

    This class enables automatic colormap adjustment when zooming/panning shared axes.
    Images are organized into groups, where each group shares a common colormap range.

    Supports multiple data types through DataWrapper abstraction:
    - xarray DataArray and Dataset
    - GeoPandas GeoDataFrame
    - matplotlib Collections (PolyCollection, LineCollection, PathCollection)
    """

    def __init__(self, output: Optional[Any] = None):
        """
        Parameters
        ----------
        output : optional
            Optional jupyter widgets.Output object for debug messages.
            If None, prints to stdout.
        """
        self.img_groups : List[Tuple[List[DataWrapper], bool]] = []
        self.output = output

    def addGroup(
        self,
        wrapper_args_group: List[WrapperArgs],
        symmetric: bool = False
    ) -> None:
        """
        Add a group of wrapper arguments that share a common colormap range.

        Parameters
        ----------
        wrapper_args_group : List[WrapperArgs]
            List of arguments to create DataWrappers. Each element can be:
            - A Collection object (self-contained)
            - A tuple/list of (data, image, ...) for xarray/GeoDataFrame
            - A dict of keyword arguments
        symmetric : bool, optional
            If True, vmin and vmax will be symmetric around 0. Default is False.
            Useful for difference maps or anomalies.

        Raises
        ------
        ValueError
            If wrapper_args_group is empty
        TypeError
            If data types are not supported
        """
        if not wrapper_args_group:
            raise ValueError("wrapper_args_group cannot be empty")

        # Wrap each data item
        self._printDebug(f'Adding {"symmetric" if symmetric else "non-symmetric"} group of size {len(wrapper_args_group)}')
        wrappers = []
        for args in wrapper_args_group:
            try:
                if isinstance(args, dict):
                    self._printDebug(f'... adding a wrapper of type {type(args["data"])}')
                    wrapper = DataWrapperFactory.wrap(**args)
                elif isinstance(args, list) or isinstance(args, tuple):
                    self._printDebug(f'... adding a wrapper of type {type(args[0])}')
                    wrapper = DataWrapperFactory.wrap(*args)
                else:
                    self._printDebug(f'... adding a wrapper of type {type(args)}')
                    wrapper = DataWrapperFactory.wrap(args)
            except (TypeError, ValueError) as e:
                self._printDebug(f'... could not wrap data of type {type(args)}')
                raise TypeError(f"Could not wrap data: {e}")
            else:
                wrappers.append(wrapper)

        self._printDebug(f'Added a group with {len(wrappers)} wrappers')
        self.img_groups.append((wrappers, symmetric))
        self._printDebug(f'Now have {len(self.img_groups)} groups')

    def _computeLimits(
        self,
        data_wrappers: List[DataWrapper],
        symmetric: bool = False
    ) -> Tuple[float, float]:
        """
        Compute vmin and vmax from a list of data wrappers.

        Parameters
        ----------
        data_wrappers : List[DataWrapper]
            List of DataWrapper objects to compute limits from
        symmetric : bool, optional
            If True, return symmetric limits around 0. Default is False.

        Returns
        -------
        Tuple[float, float]
            (vmin, vmax)
        """
        vmin_all = min(wrapper.getLimits()[0] for wrapper in data_wrappers)
        vmax_all = max(wrapper.getLimits()[1] for wrapper in data_wrappers)

        if symmetric:
            abs_max = max(abs(vmin_all), abs(vmax_all))
            vmin = -abs_max
            vmax = abs_max
        else:
            vmin, vmax = vmin_all, vmax_all

        return vmin, vmax

    def _printDebug(self, message: str) -> None:
        """
        Print debug message to output widget or stdout.

        Parameters
        ----------
        message : str
            Message to print
        """
        if self.output is not None:
            with self.output:
                print(message)
        else:
            print(message)

    def _onLimitsChanged(self, event: Any) -> None:
        """
        Callback function for axis limit changes.

        Connected to 'ylim_changed' which fires once after both xlim and ylim
        are updated (after zoom/pan completes).

        Parameters
        ----------
        event : matplotlib event object
            Event containing axes information
        """
        self._printDebug(f'onLimitsChanged...')
        xlim = event.axes.get_xlim()
        ylim = event.axes.get_ylim()
        self._printDebug(f'Axis limits changed: xlim={xlim}, ylim={ylim}')

        # Update all image groups
        self._printDebug(f'Updating: {len(self.img_groups)} groups')
        for wrappers, symmetric in self.img_groups:
            self._printDebug(f'Updating: {wrappers, symmetric}')
            self._updateGroup(wrappers, xlim, ylim, symmetric)

    def _updateGroup(
        self,
        data_wrappers : List[DataWrapper],
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        symmetric: bool
    ) -> None:
        """
        Update colormap limits for a group of images based on visible data.

        Parameters
        ----------
        data_wrappers : List[DataWrapper]
            List of wrappers to update.
        xlim : Tuple[float, float]
            Visible x-axis limits
        ylim : Tuple[float, float]
            Visible y-axis limits
        symmetric : bool
            Whether limits should be symmetric around 0
        """
        try:
            # Subset all data to visible region
            visible_wrappers = [
                data_wrapper.subset(xlim, ylim)
                for data_wrapper in data_wrappers
            ]
        except (KeyError, ValueError) as e:
            self._printDebug(f'Could not subset data: {e}')
            return
        else:
            self._printDebug(f'Got {len(visible_wrappers)} visible wrappers.')

        # Check if any visible data exists
        if all(wrapper.isEmpty() for wrapper in visible_wrappers):
            self._printDebug('Visible data is empty, skipping update')
            return

        # Compute new limits from visible data
        vmin, vmax = self._computeLimits(visible_wrappers, symmetric=symmetric)
        self._printDebug(f'Updating group: vmin={vmin:.4f}, vmax={vmax:.4f}')
        self._printDebug(f'Is symmetric? {symmetric}')

        # Apply limits to all images in the group
        for wrapper in data_wrappers:
            wrapper.setLimits(vmin=vmin, vmax=vmax)

    def connect(self, ax: Axes) -> None:
        """
        Connect the callback to an axes object.

        Should be called on one of the shared axes. The callback will be
        triggered whenever the axis limits change (via zoom/pan).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to monitor. Should be one of the shared axes
            that all images are plotted on.
        """
        ax.callbacks.connect('ylim_changed', self._onLimitsChanged)
        self._printDebug(f'Connected callback to axes: {ax}')


def createDynamicScaler(
        data_wrapper_args_list : List[WrapperGroup],
        ax: Axes,
        output: Optional[Any] = None
) -> DynamicColormapScaler:
    """
    Convenience function to create and connect a DynamicColormapScaler.

    This is the primary user-facing function for setting up dynamic colormap scaling.

    Parameters
    ----------
    data_wrapper_args_list : List[WrapperGroup]
        List of WrapperGroups. Each group is a (wrapper_args_list, symmetric) tuple where:
        - wrapper_args_list is a List[WrapperArgs] containing arguments to create DataWrappers
        - symmetric is a bool indicating symmetric scaling around 0
    ax : matplotlib.axes.Axes
        Axes to monitor for zoom/pan events. Should be one of the shared axes.
    output : optional
        Optional jupyter widgets.Output for debug messages.

    Returns
    -------
    DynamicColormapScaler
        Configured scaler instance

    Notes
    -----
    Each WrapperArgs can be:
    - A Collection object (self-contained with geometry and values)
    - A tuple/list like (data, image, ...) for xarray DataArray/Dataset or GeoDataFrame
    - A dict of keyword arguments like {'data': ..., 'image': ..., ...}

    Examples
    --------
    Create a scaler with xarray and GeoDataFrame data:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import xarray as xr
        import geopandas as gpd
        import numpy as np
        from dynamic_colormaps import createDynamicScaler

        # Create figure and axes
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))

        # Plot xarray DataArray on first axes
        dem = xr.open_dataarray('dem.nc')
        extent = [dem['x'].min().item(), dem['x'].max().item(),
                  dem['y'].min().item(), dem['y'].max().item()]
        img1 = axes[0].imshow(dem, extent=extent, cmap='gist_earth', origin='lower')
        axes[0].set_title('DEM')
        cbar1 = plt.colorbar(img1, ax=axes[0])

        # Plot GeoDataFrame on second axes
        gdf = gpd.read_file('regions.shp')
        gdf['value'] = np.random.rand(len(gdf)) * 100
        collection = gdf.plot(ax=axes[1], column='value', cmap='viridis', legend=True)
        axes[1].set_title('Regions')

        # Create wrapper argument groups
        # Both datasets share linear scaling in the same group
        wrapper_groups = [
            ([
                (dem, img1),                    # xarray: (data, image)
                (gdf, collection, 'value')      # GeoDataFrame: (gdf, collection, value_column)
            ], False),
        ]

        # Setup dynamic scaling
        scaler = createDynamicScaler(wrapper_groups, axes[0])
        plt.show()

    Example with PolyCollection, LineCollection, and scatter:

    .. code-block:: python

        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection, LineCollection
        import numpy as np
        from dynamic_colormaps import createDynamicScaler

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))

        # PolyCollection example
        polygons = [[[0, 0], [1, 0], [1, 1], [0, 1]],
                    [[1, 1], [2, 1], [2, 2], [1, 2]]]
        poly_values = [10, 20]
        poly_coll = PolyCollection(polygons, edgecolors='black')
        axes[0].add_collection(poly_coll)
        poly_coll.set_array(np.array(poly_values))
        axes[0].set_title('PolyCollection')

        # LineCollection example
        lines = [[[0, 0], [1, 1]], [[1, 0], [2, 1]]]
        line_values = [15, 25]
        line_coll = LineCollection(lines)
        axes[1].add_collection(line_coll)
        line_coll.set_array(np.array(line_values))
        axes[1].set_title('LineCollection')

        # Scatter plot example (returns PathCollection)
        scatter_x = np.random.rand(50)
        scatter_y = np.random.rand(50)
        scatter_values = np.random.rand(50) * 100
        scatter_coll = axes[2].scatter(scatter_x, scatter_y, c=scatter_values, cmap='viridis')
        axes[2].set_title('Scatter (PathCollection)')

        # Setup with all three collection types in one group with linear scaling
        # Collections are self-contained and automatically extract values via get_array()
        wrapper_groups = [
            (
                [
                    poly_coll,      # Collection objects are self-contained
                    line_coll,
                    scatter_coll,
                ],
                False
            ),
        ]

        scaler = createDynamicScaler(wrapper_groups, axes[0])
        plt.show()

    Example with symmetric scaling for differences:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import xarray as xr
        from dynamic_colormaps import createDynamicScaler

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))

        # Load data
        dem = xr.open_dataarray('dem.nc')
        dem_smooth = dem.rolling(x=5, y=5, center=True).mean()
        diff = dem_smooth - dem

        extent = [dem['x'].min().item(), dem['x'].max().item(),
                  dem['y'].min().item(), dem['y'].max().item()]

        # Plot
        img1 = axes[0].imshow(dem, extent=extent, cmap='gist_earth', origin='lower')
        img2 = axes[1].imshow(dem_smooth, extent=extent, cmap='gist_earth', origin='lower')
        img3 = axes[2].imshow(diff, extent=extent, cmap='RdBu_r', origin='lower')

        axes[0].set_title('Original DEM')
        axes[1].set_title('Smoothed DEM')
        axes[2].set_title('Difference')

        # Create wrapper groups: first two share linear scaling, third has symmetric
        wrapper_groups = [
            ([(dem, img1), (dem_smooth, img2)], False),  # Linear
            ([(diff, img3)], True),                       # Symmetric
        ]

        scaler = createDynamicScaler(wrapper_groups, axes[0])
        plt.show()
    """
    scaler = DynamicColormapScaler(output=output)

    for wrapper_args_list, symmetric in data_wrapper_args_list:
        scaler.addGroup(wrapper_args_list, symmetric=symmetric)

    scaler.connect(ax)
    return scaler
