"""Shared abstract base class for all data source managers.

Both ManagerDataset (raster/xarray) and ManagerShapes (vector/GeoDataFrame)
inherit from Manager, which exposes a ManagerAttributes plain-data object
through convenience properties.
"""

import dataclasses
import logging


@dataclasses.dataclass
class ManagerAttributes:
    """A plain-data class holding all metadata for a data source manager.

    Parameters
    ----------
    category : str
        High-level data category, e.g. ``'meteorology'``, ``'elevation'``,
        ``'soil_structure'``.
    product : str
        Human-readable product name, e.g. ``'DayMet 1km'``.
    source : str
        Human-readable source name or URL, e.g. ``'ORNL DAAC THREDDS'``.
    description : str
        Short paragraph describing the dataset.
    product_short : str or None, optional
        Filesystem-safe product slug used as the second-level cache directory
        (e.g. ``'daymet'``).  ``None`` for managers that do not use the
        standard cache system.
    source_short : str or None, optional
        Filesystem-safe source slug used as the third-level cache directory
        and directory-name prefix (e.g. ``'ornl_daac_thredds'``).  ``None``
        for managers that do not use the standard cache system.
    url : str or None, optional
        URL for the data source or portal.
    license : str or None, optional
        License under which the data is distributed.
    citation : str or None, optional
        Suggested citation for the data.
    native_crs_in : object, optional
        CRS expected for input geometry.  May be ``None`` when not yet
        known (e.g. ``ManagerShapefile`` discovers it at first open).
    native_crs_out : object, optional
        CRS of the data returned by the manager.
    native_resolution : float or None, optional
        Characteristic resolution of the data in ``native_crs_in`` units.
        Used for buffering and cache-directory snapping.
    native_start : object, optional
        Earliest start date of the data (cftime.datetime), or ``None`` for
        non-temporal datasets.
    native_end : object, optional
        Latest end date of the data (cftime.datetime), or ``None`` for
        non-temporal datasets.
    native_id_field : str or None, optional
        Name of the primary ID field in the native data.  ``None`` for
        raster managers.
    valid_variables : list or None, optional
        Valid variable names, or ``None`` for single-variable datasets.
    default_variables : list or None, optional
        Default variables to retrieve, or ``None`` for single-variable
        datasets.
    is_temporal : bool, optional
        ``True`` when the cache directory name encodes a year range.
        Default ``False``.
    is_resampled : bool, optional
        ``True`` when the cache directory name encodes a temporal resampling
        token (e.g. AORC ``'1D'``).  Default ``False``.
    """

    # Required fields (categorisation / provenance)
    category: str
    product: str
    source: str
    description: str

    # Optional cache-path slugs (None for non-caching managers)
    product_short: str | None = None
    source_short: str | None = None

    # Optional metadata
    url: str | None = None
    license: str | None = None
    citation: str | None = None

    # Native data properties
    native_crs_in: object = None
    native_crs_out: object = None
    native_resolution: float | None = None
    native_start: object = None     # cftime.datetime | None
    native_end: object = None       # cftime.datetime | None
    native_id_field: str | None = None

    # Variable selection
    valid_variables: list | None = None
    default_variables: list | None = None

    # Temporal / resampling flags (used by cache_info functions)
    is_temporal: bool = False
    is_resampled: bool = False


class Manager:
    """Base class for all data source managers.

    Accepts a :class:`ManagerAttributes` instance and exposes its fields
    as convenience properties.

    Parameters
    ----------
    attrs : ManagerAttributes
        Plain-data object holding all metadata for this manager.
    """

    def __init__(self, attrs: ManagerAttributes):
        self.attrs = attrs

    # ------------------------------------------------------------------
    # Convenience properties — delegate to self.attrs
    # ------------------------------------------------------------------

    @property
    def category(self):
        """str: High-level data category."""
        return self.attrs.category

    @property
    def product(self):
        """str: Human-readable product name."""
        return self.attrs.product

    @property
    def product_short(self):
        """str or None: Filesystem-safe product slug."""
        return self.attrs.product_short

    @property
    def source(self):
        """str: Human-readable source name."""
        return self.attrs.source

    @property
    def source_short(self):
        """str or None: Filesystem-safe source slug."""
        return self.attrs.source_short

    @property
    def url(self):
        """str or None: URL for the data source."""
        return self.attrs.url

    @property
    def license(self):
        """str or None: License under which the data is distributed."""
        return self.attrs.license

    @property
    def citation(self):
        """str or None: Suggested citation for the data."""
        return self.attrs.citation

    @property
    def description(self):
        """str: Short paragraph describing the dataset."""
        return self.attrs.description

    @property
    def native_crs_in(self):
        """CRS or None: Expected CRS of input geometry."""
        return self.attrs.native_crs_in

    @property
    def native_crs_out(self):
        """CRS or None: CRS of data returned by the manager."""
        return self.attrs.native_crs_out

    @property
    def native_resolution(self):
        """float or None: Characteristic resolution in native_crs_in units."""
        return self.attrs.native_resolution

    @property
    def native_start(self):
        """cftime.datetime or None: Earliest start date of the data."""
        return self.attrs.native_start

    @property
    def native_end(self):
        """cftime.datetime or None: Latest end date of the data."""
        return self.attrs.native_end

    @property
    def native_id_field(self):
        """str or None: Name of the primary ID field in the native data."""
        return self.attrs.native_id_field

    @property
    def valid_variables(self):
        """list or None: Valid variable names."""
        return self.attrs.valid_variables

    @property
    def default_variables(self):
        """list or None: Default variables to retrieve."""
        return self.attrs.default_variables

    @property
    def is_temporal(self):
        """bool: True when the cache directory encodes a year range."""
        return self.attrs.is_temporal

    @property
    def is_resampled(self):
        """bool: True when the cache directory encodes a temporal resampling token."""
        return self.attrs.is_resampled

    @property
    def name(self):
        """str: Class name, used in log messages."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Hook called before any data access
    # ------------------------------------------------------------------

    def _prerequestDataset(self) -> None:
        """Called before processing any request.

        Override to perform one-time setup that cannot happen at
        construction time (e.g. downloading an index file, reading
        metadata from a file that may not yet exist).
        """
        pass
