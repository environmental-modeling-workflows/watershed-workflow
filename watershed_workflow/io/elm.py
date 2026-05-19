"""Write ELM domain and surface data files from a Mesh2D.

Domain files
------------
NetCDF files with a 1D (unstructured) layout: dimensions nj=1,
ni=num_cells, nv=0.  All cells are active (mask=1, frac=1.0).
Vertex arrays (xv, yv) are omitted because nv=0 causes ELM to skip
reading them.

Surface data files
------------------
Built with SurfdataBuilder, which clones structure from a template
surfdata file and replaces spatial variables.  The template supplies
non-spatial scalars (urban block, biogeochem constants) interpolated
to the new mesh via nearest-neighbor lookup on template LATIXY/LONGXY.

The following key must be present in m2.cell_data:
  'pft index'   [int]    ELM natural PFT index 0-16  (-> PCT_NAT_PFT)

Geometry (TOPO, SLOPE, ASPECT_DEG, SINSL_COSAS, SINSL_SINAS) is
computed directly from the 3D coordinates of m2.

TODO: SOIL_COLOR -- currently nearest-neighbor from template; no
      standard geospatial source exists.
TODO: SLP_P10 -- only read when use_erosion=.true. (default .false.);
      placeholder tan(slope) values are safe for ATS-ELM runs.
TODO: SKY_VIEW / TERRAIN_CONFIG -- currently set to ELM flat-surface
      defaults (svf=1, tc=0); assess whether computing actual sky view
      factors from the mesh geometry makes physical sense at watershed
      scale given ELM's subgrid radiation assumptions.
TODO: MONTHLY_LAI/SAI/HEIGHT_TOP/HEIGHT_BOT -- only active in Satellite
      Phenology mode (use_cn=.false.); full BGC runs (use_cn=.true.,
      the intended ATS-ELM configuration) predict LAI internally and
      ignore these.  Use SurfdataBuilder.setPhenology().
"""

import numpy as np
import pandas as pd
import scipy.spatial
import xarray as xr

from watershed_workflow.mesh.mesh import Mesh2D
import watershed_workflow.crs
import watershed_workflow.utils.data
import watershed_workflow.utils.warp


# Earth radius used by ELM (elm_varcon.F90: re = SHR_CONST_REARTH * 0.001)
_R_ELM_KM = 6371.22

# Variables pulled from the template via nearest-neighbor at construction time.
# Biogeochem/hydrology scalars and the full urban block.
_TEMPLATE_NN_VARS = [
    'EF1_BTR', 'EF1_FET', 'EF1_FDT', 'EF1_SHR', 'EF1_GRS', 'EF1_CRP',
    'peatf', 'abm', 'gdp',
    'APATITE_P', 'LABILE_P', 'OCCLUDED_P', 'SECONDARY_P',
    'F0', 'P3', 'ZWT0', 'binfl', 'Ws', 'Ds', 'Dsmax',
    'parEro_c1', 'parEro_c2', 'parEro_c3', 'Tillage', 'Litho',
    'SOIL_COLOR', 'SOIL_ORDER',
    'CANYON_HWR', 'EM_IMPROAD', 'EM_PERROAD', 'EM_ROOF', 'EM_WALL',
    'HT_ROOF', 'THICK_ROOF', 'THICK_WALL',
    'T_BUILDING_MAX', 'T_BUILDING_MIN', 'WIND_HGT_CANYON',
    'WTLUNIT_ROOF', 'WTROAD_PERV',
    'ALB_IMPROAD_DIR', 'ALB_IMPROAD_DIF', 'ALB_PERROAD_DIR', 'ALB_PERROAD_DIF',
    'ALB_ROOF_DIR', 'ALB_ROOF_DIF', 'ALB_WALL_DIR', 'ALB_WALL_DIF',
    'TK_ROOF', 'TK_WALL', 'TK_IMPROAD',
    'CV_ROOF', 'CV_WALL', 'CV_IMPROAD',
    'NLEV_IMPROAD',
]


def elm_default_dzsoi(nlevgrnd=15):
    """Layer thicknesses [m] from ELM's default exponential vertical grid.

    Reproduces the formula in initVerticalMod.F90 (the ``more_vertlayers=False``
    branch) using the parameter values from elm_varpar.F90:

        zsoi(j) = scalez * (exp(zecoeff * (j - 0.5)) - 1)   j = 1 .. nlevgrnd
        dzsoi(1) = 0.5 * (zsoi(1) + zsoi(2))
        dzsoi(j) = 0.5 * (zsoi(j+1) - zsoi(j-1))            j = 2 .. nlevgrnd-1
        dzsoi(nlevgrnd) = zsoi(nlevgrnd) - zsoi(nlevgrnd-1)

    Parameters
    ----------
    nlevgrnd : int
        Number of ground layers (default 15, matching ELM's default).

    Returns
    -------
    dzsoi : np.ndarray, shape (nlevgrnd,)
        Layer thicknesses in metres, shallowest first.
    """
    scalez  = 0.025
    zecoeff = 0.50
    j = np.arange(1, nlevgrnd + 1, dtype=float)
    zsoi = scalez * (np.exp(zecoeff * (j - 0.5)) - 1.0)
    dzsoi = np.empty(nlevgrnd)
    dzsoi[0] = 0.5 * (zsoi[0] + zsoi[1])
    dzsoi[1:-1] = 0.5 * (zsoi[2:] - zsoi[:-2])
    dzsoi[-1] = zsoi[-1] - zsoi[-2]
    return dzsoi


def writeDomain(m2, filename):
    """Write an ELM domain file from a Mesh2D.

    Parameters
    ----------
    m2 : Mesh2D
        Surface mesh.  Must have a CRS set.  Coordinates are assumed to
        be in the projected CRS (meters).
    filename : str
        Path to the output NetCDF file.
    """
    assert m2.crs is not None, "Mesh2D must have a CRS set"

    ncells = m2.num_cells

    centroids = m2.centroids[:, 0:2]
    lon, lat = watershed_workflow.utils.warp.warpXY(
        centroids[:, 0], centroids[:, 1], m2.crs, watershed_workflow.crs.latlon_crs)

    area_km2 = m2.cell_areas * 1e-6

    # ELM reads 'area' in steradians then multiplies by re^2 (km) to get km^2.
    area_sr = area_km2 / (_R_ELM_KM ** 2)

    ds = xr.Dataset(
        {
            'xc': xr.DataArray(
                lon[np.newaxis, :],
                dims=('nj', 'ni'),
                attrs={
                    'long_name': 'longitude of land gridcell center',
                    'units': 'degrees_east',
                },
            ),
            'yc': xr.DataArray(
                lat[np.newaxis, :],
                dims=('nj', 'ni'),
                attrs={
                    'long_name': 'latitude of land gridcell center',
                    'units': 'degrees_north',
                },
            ),
            'area': xr.DataArray(
                area_sr[np.newaxis, :],
                dims=('nj', 'ni'),
                attrs={
                    'long_name': 'area of grid cell in radians squared',
                    'coordinates': 'xc yc',
                    'units': 'radian2',
                },
            ),
            'area_km2': xr.DataArray(
                area_km2[np.newaxis, :],
                dims=('nj', 'ni'),
                attrs={
                    'long_name': 'area of grid cell in kilometers squared',
                    'coordinates': 'xc yc',
                    'units': 'km^2',
                },
            ),
            'mask': xr.DataArray(
                np.ones((1, ncells), dtype=np.int32),
                dims=('nj', 'ni'),
                attrs={
                    'long_name': 'domain mask',
                    'coordinates': 'xc yc',
                    'comment': '0 value indicates cell is not active',
                },
            ),
            'frac': xr.DataArray(
                np.ones((1, ncells), dtype=np.float64),
                dims=('nj', 'ni'),
                attrs={
                    'long_name': 'fraction of grid cell that is active',
                    'coordinates': 'xc yc',
                },
            ),
        },
        attrs={
            'title': '1D (unstructured) domain file for running ELM',
            'Conventions': 'CF-1.0',
        },
    )

    # nv=0 dimension: ELM skips xv/yv when nv < 3.  Assign an empty
    # coordinate to force the dimension into the file.
    ds = ds.assign_coords(nv=xr.DataArray(np.empty(0, dtype=np.int32), dims=('nv',)))
    ds.to_netcdf(filename)


class SurfdataBuilder:
    """Assembles an ELM surface data file from a Mesh2D and a template.

    Construction opens the template and immediately sets geometry, land
    cover, and all template-derived scalars (biogeochem, urban block,
    SOIL_COLOR, SOIL_ORDER) via nearest-neighbor interpolation.  The
    caller then supplies soil texture (setSoilProperties) and optionally
    monthly phenology (setPhenology) before writing.

    Individual variables can be overridden at any point via update().

    Parameters
    ----------
    m2 : Mesh2D
        Surface mesh.  Must have a CRS set, 3D coordinates, and
        cell_data column 'pft index' (ELM natural PFT index 0-16).
    template_file : str
        Path to an existing ELM surfdata NetCDF file.
    use_cn : bool, optional
        If True (default), BGC mode is assumed: monthly phenology variables
        (MONTHLY_LAI/SAI/HEIGHT_TOP/HEIGHT_BOT) are omitted since ELM predicts
        LAI internally.  A minimal ``lsmpft`` coordinate is still written to
        satisfy ELM's unconditional ``check_dim('lsmpft')`` call.
        If False, monthly phenology variables are written from the template via
        nearest-neighbor interpolation; call setPhenology() afterward to
        override with real data.
    """

    def __init__(self, m2, template_file, use_cn=True):
        assert m2.crs is not None, "Mesh2D must have a CRS set"
        assert m2.coords.shape[1] == 3, "Mesh2D must have 3D coordinates"
        assert 'pft index' in m2.cell_data.columns, \
            "m2.cell_data missing required column 'pft index'"

        self._m2 = m2
        self._ncells = m2.num_cells
        self._pft_index = m2.cell_data['pft index'].values.astype(int)

        centroids_proj = m2.centroids[:, 0:2]
        self._lon, self._lat = watershed_workflow.utils.warp.warpXY(
            centroids_proj[:, 0], centroids_proj[:, 1],
            m2.crs, watershed_workflow.crs.latlon_crs)

        self._use_cn = use_cn
        self._tmpl = xr.open_dataset(template_file)
        self._nlevsoi = int(self._tmpl.dims['nlevsoi'])

        tmpl_lat = self._tmpl['LATIXY'].values.ravel()
        tmpl_lon = self._tmpl['LONGXY'].values.ravel()
        tree = scipy.spatial.cKDTree(np.column_stack([tmpl_lat, tmpl_lon]))
        _, self._nn_idx = tree.query(np.column_stack([self._lat, self._lon]))

        # Initialise dataset with non-spatial template scalars.
        self._ds = xr.Dataset(attrs=dict(self._tmpl.attrs))
        for v in ['mxsoil_color', 'mxsoil_order', 'natpft', 'time']:
            if v in self._tmpl:
                self._ds[v] = self._tmpl[v]

        # lsmpft: ELM's check_dim('lsmpft') runs unconditionally, so the
        # dimension must be present.  In BGC mode (use_cn=True) we write a
        # minimal 17-element coordinate (~136 bytes) rather than the full
        # monthly phenology arrays.  In SP mode (use_cn=False) the phenology
        # variables are written by _setTemplateScalars below, which naturally
        # creates the lsmpft dimension.
        if use_cn:
            n_lsmpft = int(self._tmpl.dims['lsmpft'])
            self._ds['lsmpft'] = xr.DataArray(
                np.arange(n_lsmpft, dtype=np.int32), dims=('lsmpft',))

        self._setGeometry()
        self._setLandCover()
        self._setTemplateScalars()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, name, value, dims=None):
        """DWIM setter for a single surfdata variable.

        Parameters
        ----------
        name : str
            ELM variable name (e.g. 'SKY_VIEW', 'PCT_SAND').
        value : bool, int, float, np.ndarray, or xr.DataArray
            False       -- remove the variable from the dataset.
            True        -- nearest-neighbor from template (template must
                           contain the variable).
            int/float   -- broadcast scalar uniformly; dims inferred from
                           template or supplied via `dims`.
            np.ndarray  -- use directly; dims inferred from template or
                           supplied via `dims`.
            xr.DataArray with x/y or lon/lat coords
                        -- interpolate raster to mesh centroids; dims
                           inferred from template or supplied via `dims`.
            xr.DataArray with gridcell dim
                        -- assign directly.
            pd.DataFrame -- not supported; use setPhenology() instead.
        dims : tuple of str, optional
            Explicit dimension names for the output DataArray.  Required
            when `name` is absent from the template and `value` is not
            an xr.DataArray with named dims.
        """
        if isinstance(value, pd.DataFrame):
            raise TypeError(
                "DataFrame values are not supported by update(); "
                "use setPhenology() for monthly LAI/SAI/height tables.")

        if value is False:
            if name in self._ds:
                del self._ds[name]
            return

        out_dims = dims if dims is not None else self._inferDims(name)
        attrs = dict(self._tmpl[name].attrs) if name in self._tmpl else {}

        if value is True:
            if name not in self._tmpl:
                raise KeyError(
                    f"'{name}' not found in template; cannot use update(name, True).")
            data = self._nn(name)
            self._ds[name] = xr.DataArray(data, dims=out_dims, attrs=attrs)
            return

        if isinstance(value, (int, float)):
            shape = self._dimsToShape(out_dims)
            data = np.full(shape, value)
            self._ds[name] = xr.DataArray(data, dims=out_dims, attrs=attrs)
            return

        if isinstance(value, np.ndarray):
            if out_dims is None:
                raise ValueError(
                    f"Cannot infer dims for '{name}' (not in template); "
                    f"pass dims= explicitly.")
            self._ds[name] = xr.DataArray(value, dims=out_dims, attrs=attrs)
            return

        if isinstance(value, xr.DataArray):
            # Raster with spatial coords: interpolate to mesh centroids.
            if {'x', 'y'}.issubset(value.coords) or {'lon', 'lat'}.issubset(value.coords):
                interp = watershed_workflow.utils.data.interpolateValues(
                    np.column_stack([self._lon, self._lat]),
                    watershed_workflow.crs.latlon_crs,
                    value)
                if out_dims is None:
                    raise ValueError(
                        f"Cannot infer dims for '{name}' (not in template); "
                        f"pass dims= explicitly.")
                self._ds[name] = xr.DataArray(interp, dims=out_dims, attrs=attrs)
            else:
                # Already on mesh (or has explicit named dims); assign directly,
                # substituting any trailing lsmlat/lsmlon with gridcell.
                new_dims = self._substituteGridcellDims(value.dims)
                self._ds[name] = value.assign_coords({}).rename(
                    dict(zip(value.dims, new_dims)))
            return

        raise TypeError(
            f"update() value must be bool, int, float, ndarray, or DataArray; "
            f"got {type(value).__name__}")

    def setSoilProperties(self, soil_type, soil_properties):
        """Set soil texture and organic matter from a material-ID lookup table.

        Parameters
        ----------
        soil_type : np.ndarray of shape (nlevsoi, ncells), int
            Material ID per layer per surface cell.  Same convention as
            mat_ids in Mesh3D.extruded_Mesh2D (layers leading, cells trailing).
        soil_properties : pd.DataFrame
            Indexed by material ID integer.  Required columns:

              'total sand pct [%]'      percent sand (0-100)
              'total clay pct [%]'      percent clay (0-100)

            Optional columns (zero-filled if absent):

              'total gravel pct [%]'    percent gravel (0-100)
              'organic matter [kg/m^3]' organic matter density
        """
        soil_type = np.asarray(soil_type, dtype=int)
        assert soil_type.shape == (self._nlevsoi, self._ncells), (
            f"soil_type shape {soil_type.shape} does not match "
            f"(nlevsoi={self._nlevsoi}, ncells={self._ncells})")

        flat = soil_type.ravel()
        pct_sand = (soil_properties.loc[flat, 'total sand pct [%]']
                    .values.reshape(self._nlevsoi, self._ncells))
        pct_clay = (soil_properties.loc[flat, 'total clay pct [%]']
                    .values.reshape(self._nlevsoi, self._ncells))

        if 'total gravel pct [%]' in soil_properties.columns:
            pct_grvl = (soil_properties.loc[flat, 'total gravel pct [%]']
                        .values.reshape(self._nlevsoi, self._ncells))
        else:
            pct_grvl = np.zeros((self._nlevsoi, self._ncells))

        if 'organic matter [kg/m^3]' in soil_properties.columns:
            organic = (soil_properties.loc[flat, 'organic matter [kg/m^3]']
                       .values.reshape(self._nlevsoi, self._ncells))
        else:
            organic = np.zeros((self._nlevsoi, self._ncells))

        def _a(v): return dict(self._tmpl[v].attrs) if v in self._tmpl else {}
        self._ds['PCT_SAND'] = xr.DataArray(pct_sand, dims=('nlevsoi', 'gridcell'), attrs=_a('PCT_SAND'))
        self._ds['PCT_CLAY'] = xr.DataArray(pct_clay, dims=('nlevsoi', 'gridcell'), attrs=_a('PCT_CLAY'))
        self._ds['PCT_GRVL'] = xr.DataArray(pct_grvl, dims=('nlevsoi', 'gridcell'), attrs=_a('PCT_GRVL'))
        self._ds['ORGANIC']  = xr.DataArray(organic,  dims=('nlevsoi', 'gridcell'), attrs=_a('ORGANIC'))

    def setPhenology(self, lai=False, sai=False, height_top=False, height_bot=False):
        """Set monthly canopy phenology tables.

        Only needed in Satellite Phenology mode (use_cn=False).  BGC
        runs (use_cn=True, the intended ATS-ELM configuration) predict
        LAI internally and ignore these variables.

        Parameters
        ----------
        lai, sai, height_top, height_bot : False, True, DataFrame, or DataArray
            False (default) -- remove variable from dataset.
            True            -- nearest-neighbor from template.
            pd.DataFrame    -- index months 1-12, columns PFT indices 0-16;
                               broadcast uniformly across all cells.
            xr.DataArray    -- shape (time=12, y, x) with spatial coords;
                               interpolated to mesh centroids and slotted
                               into the dominant PFT per cell.
        """
        pairs = [
            ('MONTHLY_LAI',        lai),
            ('MONTHLY_SAI',        sai),
            ('MONTHLY_HEIGHT_TOP', height_top),
            ('MONTHLY_HEIGHT_BOT', height_bot),
        ]
        for varname, param in pairs:
            table = self._computePhenologyTable(varname, param)
            if table is None:
                if varname in self._ds:
                    del self._ds[varname]
            else:
                attrs = dict(self._tmpl[varname].attrs) if varname in self._tmpl else {}
                self._ds[varname] = xr.DataArray(
                    table, dims=('time', 'lsmpft', 'gridcell'), attrs=attrs)

    def build(self):
        """Return the assembled xr.Dataset (copy, safe to inspect)."""
        return self._ds.copy()

    def write(self, filename):
        """Write the assembled dataset to a NetCDF file.

        Parameters
        ----------
        filename : str
            Output path.
        """
        self._tmpl.close()
        self._ds.to_netcdf(filename)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nn(self, varname):
        """Return template variable reindexed to nearest-neighbor mesh cells.

        The template may be a 1-column file with shape (..., 1, 1) or (..., 1, N).
        Squeeze out size-1 trailing spatial dims before fancy-indexing so that the
        result shape matches the dims returned by _inferDims (which collapses all
        trailing lsmlat/lsmlon dims into a single 'gridcell' axis).
        """
        v = self._tmpl[varname].values
        # squeeze trailing size-1 spatial axes until the last axis is the one to index
        while v.ndim >= 2 and v.shape[-2] == 1:
            v = v.squeeze(axis=-2)
        return v[..., self._nn_idx]

    def _inferDims(self, name):
        """Infer output dims for name by inspecting the template.

        Trailing lsmlat/lsmlon (2D template) or gridcell (1D template)
        are replaced with 'gridcell'.  Returns None if name is absent.
        """
        if name not in self._tmpl:
            return None
        raw = list(self._tmpl[name].dims)
        return tuple(self._substituteGridcellDims(raw))

    def _substituteGridcellDims(self, dims):
        """Replace trailing spatial dims with 'gridcell'."""
        dims = list(dims)
        if len(dims) >= 2 and dims[-2] == 'lsmlat' and dims[-1] == 'lsmlon':
            return tuple(dims[:-2]) + ('gridcell',)
        if dims and dims[-1] in ('lsmlat', 'lsmlon', 'gridcell'):
            return tuple(dims[:-1]) + ('gridcell',)
        return tuple(dims)

    def _dimsToShape(self, dims):
        """Return the expected array shape for the given output dims."""
        size = {
            'gridcell': self._ncells,
            'nlevsoi':  self._nlevsoi,
            'natpft':   17,
            'lsmpft':   17,
            'numurbl':  3,
            'nlevslp':  11,
        }
        # For dims not in the table, pull from template or dataset.
        shape = []
        for d in dims:
            if d in size:
                shape.append(size[d])
            elif d in self._tmpl.dims:
                shape.append(int(self._tmpl.dims[d]))
            elif d in self._ds.dims:
                shape.append(int(self._ds.dims[d]))
            else:
                raise ValueError(f"Cannot determine size of dimension '{d}'")
        return tuple(shape)

    def _setGeometry(self):
        """Compute and assign all geometry variables from m2."""
        m2 = self._m2
        ncells = self._ncells

        area_km2 = m2.cell_areas * 1e-6

        topo      = m2.centroids[:, 2]
        slope_deg = m2.slopes
        aspect_deg = m2.aspects
        slope_rad = np.deg2rad(slope_deg)
        nz = m2.normals[:, 2]

        # ELM stores sinsl_cosas = tan(slope)*cos(aspect) = ny/nz and
        # sinsl_sinas = tan(slope)*sin(aspect) = nx/nz (domainMod.F90).
        # Derived from the unit normal to avoid degree/radian ambiguity.
        # NOTE: the fmyuan Coweeta file has a bug where slope/aspect in
        # degrees were passed raw to NumPy trig (expecting radians).
        sinsl_cosas = m2.normals[:, 1] / nz
        sinsl_sinas = m2.normals[:, 0] / nz

        # SLP_P10: 11 quantiles collapsed to uniform tan(slope) per cell.
        # TODO: verify whether ELM reads SLP_P10 outside use_erosion=T.
        slp_p10 = np.tile(np.tan(slope_rad), (11, 1))

        # ELM stores sky_view = svf/cos(slope), terrain_config = tc/cos(slope).
        # Flat-surface defaults: svf=1, tc=0 → sky_view=1/cos(0)=1, tc=0.
        # TODO: assess computing actual SVF from mesh geometry.
        sky_view       = 1.0 / np.cos(slope_rad)
        terrain_config = np.zeros(ncells)

        def _a(v): return dict(self._tmpl[v].attrs) if v in self._tmpl else {}
        def _da(data, dims, v):
            return xr.DataArray(data, dims=dims, attrs=_a(v))

        self._ds['LONGXY']        = _da(self._lon,    ('gridcell',),          'LONGXY')
        self._ds['LATIXY']        = _da(self._lat,    ('gridcell',),          'LATIXY')
        self._ds['AREA']          = _da(area_km2,     ('gridcell',),          'AREA')
        self._ds['TOPO']          = _da(topo,         ('gridcell',),          'TOPO')
        self._ds['SLOPE']         = _da(slope_deg,    ('gridcell',),          'SLOPE')
        self._ds['ASPECT_DEG']    = _da(aspect_deg,   ('gridcell',),          'ASPECT_DEG')
        self._ds['SINSL_COSAS']   = _da(sinsl_cosas,  ('gridcell',),          'SINSL_COSAS')
        self._ds['SINSL_SINAS']   = _da(sinsl_sinas,  ('gridcell',),          'SINSL_SINAS')
        self._ds['STD_ELEV']      = _da(np.zeros(ncells), ('gridcell',),      'STD_ELEV')
        self._ds['SLP_P10']       = _da(slp_p10,      ('nlevslp', 'gridcell'), 'SLP_P10')
        self._ds['SKY_VIEW']      = _da(sky_view,     ('gridcell',),          'SKY_VIEW')
        self._ds['TERRAIN_CONFIG']= _da(terrain_config, ('gridcell',),        'TERRAIN_CONFIG')

    def _setLandCover(self):
        """Assign ATS-ELM land cover constraints from m2.cell_data['pft index']."""
        ncells = self._ncells
        pct_nat_pft = np.zeros((17, ncells), dtype=np.float64)
        for c, p in enumerate(self._pft_index):
            pct_nat_pft[p, c] = 100.0

        def _a(v): return dict(self._tmpl[v].attrs) if v in self._tmpl else {}
        def _da(data, dims, v):
            return xr.DataArray(data, dims=dims, attrs=_a(v))

        self._ds['LANDFRAC_PFT']  = _da(np.ones(ncells),                    ('gridcell',),          'LANDFRAC_PFT')
        self._ds['PFTDATA_MASK']  = _da(np.ones(ncells, dtype=np.int32),    ('gridcell',),          'PFTDATA_MASK')
        self._ds['PCT_NATVEG']    = _da(np.full(ncells, 100.0),             ('gridcell',),          'PCT_NATVEG')
        self._ds['PCT_CROP']      = _da(np.zeros(ncells),                   ('gridcell',),          'PCT_CROP')
        self._ds['PCT_LAKE']      = _da(np.zeros(ncells),                   ('gridcell',),          'PCT_LAKE')
        self._ds['PCT_WETLAND']   = _da(np.zeros(ncells),                   ('gridcell',),          'PCT_WETLAND')
        self._ds['PCT_GLACIER']   = _da(np.zeros(ncells),                   ('gridcell',),          'PCT_GLACIER')
        self._ds['PCT_URBAN']     = _da(np.zeros((3, ncells)),              ('numurbl', 'gridcell'), 'PCT_URBAN')
        self._ds['URBAN_REGION_ID'] = _da(np.zeros(ncells, dtype=np.int32), ('gridcell',),          'URBAN_REGION_ID')
        self._ds['PCT_NAT_PFT']   = _da(pct_nat_pft,                       ('natpft', 'gridcell'),  'PCT_NAT_PFT')
        self._ds['LAKEDEPTH']     = _da(np.zeros(ncells),                   ('gridcell',),          'LAKEDEPTH')
        # FMAX=1: ATS controls inundated fraction dynamically.
        self._ds['FMAX']          = _da(np.ones(ncells),                    ('gridcell',),          'FMAX')

        # Placeholder soil texture until setSoilProperties() is called.
        self._ds['PCT_SAND'] = xr.DataArray(
            np.full((self._nlevsoi, ncells), 40.0), dims=('nlevsoi', 'gridcell'))
        self._ds['PCT_CLAY'] = xr.DataArray(
            np.full((self._nlevsoi, ncells), 20.0), dims=('nlevsoi', 'gridcell'))
        self._ds['PCT_GRVL'] = xr.DataArray(
            np.zeros((self._nlevsoi, ncells)), dims=('nlevsoi', 'gridcell'))
        self._ds['ORGANIC']  = xr.DataArray(
            np.zeros((self._nlevsoi, ncells)), dims=('nlevsoi', 'gridcell'))

    def _setTemplateScalars(self):
        """NN-interpolate all known template scalars to the mesh."""
        _PHENOLOGY_VARS = ('MONTHLY_LAI', 'MONTHLY_SAI',
                           'MONTHLY_HEIGHT_TOP', 'MONTHLY_HEIGHT_BOT')
        vars_to_copy = _TEMPLATE_NN_VARS
        if not self._use_cn:
            vars_to_copy = list(_TEMPLATE_NN_VARS) + list(_PHENOLOGY_VARS)
        for name in vars_to_copy:
            if name not in self._tmpl:
                continue
            dims = self._inferDims(name)
            data = self._nn(name)
            attrs = dict(self._tmpl[name].attrs)
            self._ds[name] = xr.DataArray(data, dims=dims, attrs=attrs)

    def _computePhenologyTable(self, varname, param):
        """Return a (12, 17, ncells) array for one monthly phenology variable.

        Returns None when param is False (variable should be omitted).
        """
        n_time, n_pft = 12, 17
        ncells = self._ncells

        if param is False:
            return None

        if param is True:
            if varname not in self._tmpl:
                raise KeyError(
                    f"'{varname}' not found in template; cannot use True.")
            return self._nn(varname)

        if isinstance(param, pd.DataFrame):
            table = np.zeros((n_time, n_pft))
            month_0based = min(int(i) for i in param.index) == 0
            for month_idx in param.index:
                m = int(month_idx) if month_0based else int(month_idx) - 1
                for pft_col in param.columns:
                    table[m, int(pft_col)] = param.loc[month_idx, pft_col]
            return np.broadcast_to(
                table[:, :, np.newaxis], (n_time, n_pft, ncells)).copy()

        if isinstance(param, xr.DataArray):
            assert param.sizes.get('time', param.shape[0]) == n_time, (
                f"DataArray leading dimension must be 12 (months), "
                f"got shape {param.shape}")
            interp = watershed_workflow.utils.data.interpolateValues(
                np.column_stack([self._lon, self._lat]),
                watershed_workflow.crs.latlon_crs,
                param)  # (12, ncells)
            table = np.zeros((n_time, n_pft, ncells))
            for c in range(ncells):
                table[:, self._pft_index[c], c] = interp[:, c]
            return table

        raise TypeError(
            f"phenology param must be False, True, DataFrame, or DataArray; "
            f"got {type(param).__name__}")
