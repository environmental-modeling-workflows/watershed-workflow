#!/usr/bin/env python
# coding: utf-8

# # Coweeta ELM+ATS Coupled Workflow
# 
# This workflow generates all input files needed for a coupled ELM+ATS simulation at the
# Coweeta watershed. ELM drives all meteorological forcing and provides porosity (watsat)
# to ATS at runtime via the ELM-ATS coupling layer.
#
# Datasets used:
# * `NHD Plus HR` for hydrography
# * `3DEP` for elevation
# * `NLCD` for land cover (PFT assignment for ELM)
# * `NRCS/SSURGO` for soil texture (sand/clay → ATS WRM via Rosetta) and porosity
#   (NRCS/Rosetta `porosity [-]`, written to fsurdat as `WATSAT` and read directly by
#   ELM via `watsat_type = 'from_file'`, rather than computed from sand/clay via ELM's
#   Cosby et al. 1984 pedotransfer function) -- so ELM's watsat and ATS's base_porosity
#   agree exactly
# 
# Outputs written to `elm_output_data/`:
# * `coweeta_np<NTASKS>.exo` — 3D mesh (15 layers, ~42 m deep, Oak Harbor dz
#   structure), partitioned into NTASKS pieces; the filename is tagged with
#   NTASKS since the mesh partition is fixed at build time and must match the
#   MPI rank count used to build and run the ELM+ATS cases
# * `coweeta_np<NTASKS>.h5` — lat/lon cell data for ELM column matching
# * `coweeta_domain.nc` — ELM domain file
# * `coweeta_surfdata.nc` — ELM surface data (PFT, texture, geometry)
# * `coweeta_stage1a.xml` — ATS-only steady-state spinup input (stage 1a of
#   `examples/coweeta/build_example.sh`); self-contained, no ELM coupling
# * `coweeta_stage2.xml`, `coweeta_stage3.xml`, `coweeta_stage3b.xml` — ATS input
#   files for the coupled ELM+ATS runs (stages 2, 3, and 3b), based on
#   `elm_ats_template.xml`; identical except for which previous stage's checkpoint
#   each restarts ATS pressure from (stage 3b restarts from stage 1a, not stage 2)
# * `coweeta_subsurface_properties.csv` — soil properties lookup table
# * `user_nl_elm_stage1c`, `user_nl_elm_stage2`, `user_nl_elm_stage3`,
#   `user_nl_elm_stage3b` — ELM's own input files for each ELM-running stage,
#   written alongside the ATS XMLs since they are just as much domain-specific
#   input generation
# * `coweeta_1c_domain.nc`, `coweeta_1c_surfdata.nc` — stage 1c's synthetic
#   one-flat-column-per-PFT domain/surfdata
# * `coweeta_1c_pft_crosswalk.json` — maps stage 1c's columns to PFT type, consumed
#   by `examples/coweeta/distribute_pft_finidat.py` to build stage 2's `finidat`
# * `coweeta_stage3b_surfdata.nc` — SP-mode (`use_cn=False`) ELM surface data with
#   MONTHLY_LAI/etc. phenology tables derived from MODIS LAI, for stage 3b
#
# After running this script, copy the outputs into a campaign directory's shared
# `inputdata/` (e.g. `Coweeta_Campaign0/inputdata/`) as described in the completion cell.


## FIX ME -- why is this broken without importing netcdf first?
import netCDF4

import watershed_workflow.io
watershed_workflow.io.setupLogging(1)

import exodus3

import os, sys
import json
import logging
import numpy as np
from matplotlib import pyplot as plt
import shapely
import shapely.geometry
import pandas as pd
import geopandas as gpd
pd.options.display.max_columns = None

# ats_input_spec is not pip-installable; add its source tree to the path
sys.path.insert(0, os.path.expanduser('~/code/ats/ats_input_spec/repos/master'))

import watershed_workflow
import watershed_workflow.utils
import watershed_workflow.utils.warp
import watershed_workflow.sources
import watershed_workflow.plot
import watershed_workflow.mesh
import watershed_workflow.crs
import watershed_workflow.properties.land_cover
import watershed_workflow.hydro
import watershed_workflow.io
import watershed_workflow.io.elm
import watershed_workflow.sources.standard_names as names
import watershed_workflow.sources.manager_nlcd
import watershed_workflow.sources.manager_modis_earthdata

import ats_input_spec
import ats_input_spec.public
import ats_input_spec.io

import amanzi_xml.utils.io as aio
import amanzi_xml.utils.search as asearch
import amanzi_xml.utils.errors as aerrors

import h5py

plt.rcParams['figure.figsize'] = (8, 6)


# ## Parameters

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
def splitPathFull(path):
    parts = []
    while True:
        head, tail = os.path.split(path)
        if head == path:
            parts.insert(0, head); break
        elif tail == path:
            parts.insert(0, tail); break
        else:
            parts.insert(0, tail); path = head
    return parts

cwd = splitPathFull(os.getcwd())
if cwd[-1] == 'Coweeta':
    pass
elif cwd[-1] == 'examples':
    cwd.append('Coweeta')
else:
    cwd.extend(['examples', 'Coweeta'])

data_dir = os.path.join(*(cwd + ['input_data']))
def toInput(f): return os.path.join(data_dir, f)

output_dir = os.path.join(*(cwd + ['elm_output_data']))
os.makedirs(output_dir, exist_ok=True)
def toOutput(f): return os.path.join(output_dir, f)

# Shared figure layout for the four slide-panel maps (elevation/mesh, PFT,
# soil type, met forcing) -- fixed axes rect regardless of whether a
# colorbar is present, so the maps themselves are identically sized and
# positioned when assembled into a slide. No ticks, tick labels, axis
# labels, or spines; colorbar (when present) sits at a fixed rect on the
# right.
# Sized for a 2x2 grid on a standard 13.33x7.5in widescreen slide (each
# panel occupies roughly a 6x4.5in quadrant) so that fonts saved at their
# nominal point size (14pt titles, 12pt colorbar labels) read correctly
# at full size in the deck.
_PANEL_FIGSIZE = (6, 4.5)
_PANEL_AX_RECT = [0.04, 0.06, 0.68, 0.84]     # [left, bottom, width, height]
_PANEL_CBAX_RECT = [0.76, 0.15, 0.03, 0.62]
_PANEL_TITLE_FONTSIZE = 14
_PANEL_CBAR_FONTSIZE = 12

def newPanelFigure():
    """Create a figure + map axes at the standard slide-panel rect."""
    fig = plt.figure(figsize=_PANEL_FIGSIZE)
    ax = fig.add_axes(_PANEL_AX_RECT)
    return fig, ax

def finishPanel(fig, ax, title, mappable=None, cbar_kwargs=None):
    """Strip ticks/labels/spines, add the standardized title, and (if a
    mappable is given) a right-side colorbar at the fixed panel rect."""
    ax.set_title(title, fontsize=_PANEL_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    if mappable is not None:
        cbax = fig.add_axes(_PANEL_CBAX_RECT)
        kwargs = dict(cbar_kwargs) if cbar_kwargs else {}
        kwargs.setdefault('cax', cbax)
        cb = fig.colorbar(mappable, **kwargs)
        cb.ax.tick_params(labelsize=_PANEL_CBAR_FONTSIZE)
    return fig

work_dir = os.path.join(*cwd)
def toWorkingDir(f): return os.path.join(work_dir, f)

# Point at local data cache so this script works offline / without large downloads
watershed_workflow.utils.setDataDirectory(data_dir)

# -----------------------------------------------------------------------
# Mesh / watershed parameters
# -----------------------------------------------------------------------
name = 'coweeta'
coweeta_shapefile = toInput('coweeta_basin.shp')

simplify          = 60
ignore_small_rivers = 2
prune_by_area_fraction = 0.01

refine_L0 = 125;  refine_L1 = 300
refine_d0 = 200;  refine_d1 = 600
refine_A0 = refine_L0**2 / 2
refine_A1 = refine_L1**2 / 2

# -----------------------------------------------------------------------
# ELM / ATS template paths
# -----------------------------------------------------------------------
# Oak Harbor 1-column surfdata used as template for SurfdataBuilder
template_surfdata = toInput('elm_ats_surfdata_template.nc')

# ATS XML template (site-independent ELM-ATS skeleton with MESH_FILENAME placeholders)
ats_xml_template = toInput('elm_ats_template.xml')


print(f'template_surfdata: {template_surfdata}')
print(f'ats_xml_template : {ats_xml_template}')
print(f'output_dir       : {output_dir}')

output_filenames = {}

crs = watershed_workflow.crs.default_crs


# ## Section 1: Basin Geometry

coweeta_source = watershed_workflow.sources.ManagerShapefile(coweeta_shapefile)
coweeta = coweeta_source.getShapes(out_crs=crs)
coweeta.rename(columns={'AREA': names.AREA}, inplace=True)
watershed = watershed_workflow.Watershed(coweeta)
watershed_ax = watershed.plot()
watershed_ax.get_figure().savefig(toOutput('coweeta_watershed_boundary.png'), dpi=150)
plt.close(watershed_ax.get_figure())

sources = watershed_workflow.sources.getDefaultSources()
sources['hydrography'] = watershed_workflow.sources.sources['geometry']['NHDPlusHR-pynhd_nhdplushr']
watershed_workflow.sources.logSources(sources)

reaches = sources['hydrography'].getShapesByGeometry(watershed.exterior, crs, out_crs=crs)
rivers  = watershed_workflow.hydro.createRivers(reaches, method='hydroseq')
watershed_orig, rivers_orig = watershed, rivers

def createCopy(ws, rivs):
    return ws.deepcopy(), [r.deepcopy() for r in rivs]

watershed, rivers = createCopy(watershed_orig, rivers_orig)

watershed_workflow.simplify(watershed, rivers, refine_L0, refine_L1, refine_d0, refine_d1)
for river in rivers:
    river.resetDataFrame()
for river in rivers:
    watershed_workflow.hydro.findOutletsByCrossings(watershed, river)

min_angle = 32

def widths(reach):
    mapping = {1: 8, 2: 12, 3: 16}
    return mapping.get(reach.properties['stream_order'], 8)

m2, areas, dists = watershed_workflow.tessalateRiverAligned(
    watershed, rivers,
    river_width=widths,
    refine_min_angle=min_angle,
    refine_distance=[refine_d0, refine_A0, refine_d1, refine_A1],
    diagnostics=True)

# elevate to a dem
dem = sources['DEM'].getDataset(watershed.exterior.buffer(100), watershed.crs)['dem']
watershed_workflow.elevate(m2, dem)

watershed_workflow.mesh.addWatershedAndOutletRegions(m2, watershed, outlet_width=250, exterior_outlet=True)
watershed_workflow.mesh.addRiverCorridorRegions(m2, rivers)
watershed_workflow.mesh.addStreamOrderRegions(m2, rivers)

for ls in m2.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

# now deal with the river...
# adding elevations to the river tree for stream bed conditioning
watershed_workflow.mesh.setProfileByDEM(rivers, dem)

# now condition the river to fix places where the DEM does not intersect the river
def computeBurnInDepthFromData(reach):
    return reach['bankfull_depth'] 

def computeBurnInDepth(da_sq_miles):
    """burn-in depth as a function of drainage area"""
    depth_in_feet = 1.22 * da_sq_miles**0.317
    return 0.3048 * depth_in_feet # ft --> meters

def computeBurnInDepthFromDA(reach):
    depth = computeBurnInDepth(reach['drainage_area_sqkm'] * 0.386102)
    logging.debug(f"reach of DA {reach['drainage_area_sqkm']} has depth {depth}")
    return depth

watershed_workflow.mesh.conditionRiverMeshes(m2,
                       rivers,
                       network_burn_in_depth=computeBurnInDepthFromDA)

# hydrologically condition the mesh, removing pits
outlet_edge = watershed_workflow.mesh.Edge(rivers[0]['elems'][-1][0],rivers[0]['elems'][-1][-1])
print(outlet_edge)
river_cells = [i for (i,elem) in enumerate(m2.conn) if len(elem) > 3]    
m2, res = watershed_workflow.mesh.conditionMesh(m2, preserved_pits=river_cells, forced_outlet_edges=[outlet_edge,])

print(m2.cell_areas[0])

print(m2.cell_areas[0])
print(m2.centroids[0])

fig, ax = newPanelFigure()
mp = m2.plot(facecolors='elevation', edgecolors=None, ax=ax, colorbar=False)
ax.set_aspect('equal', 'datalim')
finishPanel(fig, ax, 'surface elevation [m]', mappable=mp)
fig.savefig(toOutput('coweeta_mesh_elevation.png'), dpi=150, transparent=True)
plt.close(fig)

# Save the mesh's own outer boundary (not the original watershed
# delineation polygon -- mesh conditioning/stream burn-in grows the domain
# slightly beyond it) as a shapefile, so other scripts (e.g. the met
# forcing slide panel) can clip/scale to the *exact* mesh footprint rather
# than an approximation, keeping the watershed outline pixel-aligned
# across all slide panels.
boundary_vertex_order = []
for e in m2.boundary_edges:
    boundary_vertex_order.append(e[0])
boundary_coords = m2.coords[boundary_vertex_order, 0:2]
boundary_poly = shapely.geometry.Polygon(boundary_coords)
boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_poly], crs=m2.crs)
boundary_gdf.to_file(toOutput('coweeta_mesh_boundary.shp'))

print(m2.num_cells)

# pre-partition
#
# Note this MUST be done after addStreamOrderRegions() and after all DEM conditioning.
#
# NTASKS must match the MPI rank count build_example.sh will use to build and
# run the ELM+ATS cases -- ATS's mesh decomposition (and any checkpoint
# restart across spinup stages) is fixed at this partition count. The mesh
# filename itself is tagged with NTASKS (mesh_name, used below for
# coweeta_np<NTASKS>.exo / coweeta_np<NTASKS>.h5) so a stale mesh built for a
# different task count is obviously
# named wrong rather than silently mismatched -- build_example.sh's own
# NTASKS-vs-partition-count check is a second, independent guard on the same
# invariant.
try:
    ntasks = int(os.environ['NTASKS'])
except KeyError:
    ntasks = 4 # default for Coweeta
mesh_name = f'{name}_np{ntasks}'
m2 = m2.partition(ntasks, True)


# ## Section 2: NLCD Land Cover and ELM PFT Assignment

nlcd = sources['land cover'].getDataset(watershed.exterior.buffer(100), watershed.crs)['cover']
logging.info('Found land cover types: {}'.format(set(list(nlcd.values.ravel()))))

nlcd_indices, nlcd_cmap, nlcd_norm, nlcd_ticks, nlcd_labels = \
    watershed_workflow.plot.createNLCDColormap(np.unique(nlcd))

fig, ax = plt.subplots()
nlcd.plot.imshow(ax=ax, cmap=nlcd_cmap, norm=nlcd_norm, add_colorbar=False)
watershed_workflow.plot.createIndexedColorbar(
    ncolors=len(nlcd_indices), cmap=nlcd_cmap, labels=nlcd_labels, ax=ax)
ax.set_title('NLCD Land Cover')
fig.savefig(toOutput('coweeta_nlcd_land_cover.png'), dpi=150)
plt.close(fig)

m2_nlcd = watershed_workflow.getDatasetOnMesh(m2, nlcd, method='nearest')
m2.cell_data['land_cover'] = m2_nlcd

assert 127 not in m2_nlcd, 'nodata NLCD value found on mesh'

nlcd_indices, nlcd_cmap, nlcd_norm, nlcd_ticks, nlcd_labels = \
    watershed_workflow.plot.createNLCDColormap(np.unique(m2_nlcd))

# Map NLCD codes to ELM natural PFT indices (0-16)
pft_index = watershed_workflow.properties.land_cover.mapNLCDToPFT(
    m2_nlcd, climate_zone='temperate')
m2.cell_data['pft index'] = pft_index

print('PFT indices on mesh:', np.unique(pft_index))

fig, ax = newPanelFigure()
pft_index_list, pft_cmap, pft_norm, pft_ticks, pft_labels = \
    watershed_workflow.plot.createPFTColormap(np.unique(pft_index))

mp = m2.plot('pft index', cmap=pft_cmap, norm=pft_norm,
             edgecolors=None, colorbar=False, ax=ax)
ax.set_aspect('equal', 'datalim')
finishPanel(fig, ax, 'ELM plant functional type')
# wrap onto two lines (break at the last space) so long PFT names fit
# within the panel's fixed colorbar-label margin without shrinking fonts
pft_labels_wrapped = [lbl.rsplit(' ', 1)[0] + '\n' + lbl.rsplit(' ', 1)[1]
                       if ' ' in lbl else lbl for lbl in pft_labels]
pft_cb = watershed_workflow.plot.createIndexedColorbar(
    ncolors=len(pft_index_list), cmap=pft_cmap, labels=pft_labels_wrapped,
    cax=fig.add_axes(_PANEL_CBAX_RECT))
pft_cb.ax.tick_params(labelsize=_PANEL_CBAR_FONTSIZE)

fig.savefig(toOutput('coweeta_pft.png'), dpi=150, transparent=True)
plt.close(fig)

nlcd_labels_dict = dict(zip(nlcd_indices, nlcd_labels))
watershed_workflow.mesh.addSurfaceRegions(m2, names=nlcd_labels_dict)

for ls in m2.labeled_sets:
    print(f'{ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')


# ## Section 3: NRCS Soil Texture and ATS WRM Parameters
#
# NRCS SSURGO provides:
# - sand/clay texture, used for ATS's Rosetta van Genuchten parameters (alpha, n,
#   residual saturation) and permeability, and (for stages 1c/2/3, see Sections 5/6)
#   for ELM's `bsw`/`sucsat`/`hksat`, still computed via its Clapp-Hornberger
#   pedotransfer function from sand/clay regardless of watsat_type
# - porosity (NRCS/Rosetta `porosity [-]`), used directly for ATS's `base_porosity`
#   (stage 1a only, see Section 8) and written to fsurdat as `WATSAT` for ELM's
#   `watsat_type = 'from_file'` (stages 1c/2/3) -- so ELM's watsat and ATS's porosity
#   never diverge, instead of ELM independently deriving watsat from sand/clay via its
#   Cosby et al. 1984 pedotransfer function (`watsat = 0.489 - 0.00126*sand`)
#
# Each column is vertically homogeneous — the same NRCS MUKEY texture applies to all 15
# layers.  ELM internally duplicates layer-10 texture for layers 11-15 (SoilStateType.F90
# lines 708-713), consistent with this approach.

nrcs = sources['soil structure'].getShapesByGeometry(
    watershed.exterior, watershed.crs, out_crs=crs)

print(watershed_workflow.properties.soil.__file__)

def replaceColumnNans(df, col_nan, col_replacement):
    mask = df[col_nan].isna()
    df.loc[mask, col_nan] = df.loc[mask, col_replacement]

# Fill any missing Rosetta-derived values with Rosetta fallback columns
replaceColumnNans(nrcs, 'porosity [-]', 'Rosetta porosity [-]')
replaceColumnNans(nrcs, 'permeability [m^2]', 'Rosetta permeability [m^2]')

# Drop Rosetta backup columns (we keep the originals, now gap-filled)
for col in ['Rosetta porosity [-]', 'Rosetta permeability [m^2]',
            'bulk density [g/cm^3]']:
    if col in nrcs.columns:
        nrcs.pop(col)

# Drop any rows still containing NaN
nan_mask = nrcs.isna().any(axis=1)
dropped = nrcs.index[nan_mask]
if len(dropped):
    logging.warning(f'Dropping {len(dropped)} NRCS rows with NaN after gap-fill: {list(dropped)}')
nrcs = nrcs[~nan_mask]


# Map NRCS MUKEY onto the 2D mesh cells
soil_color_mukey = watershed_workflow.getShapePropertiesOnMesh(
    m2, nrcs, 'mukey', resolution=50, nodata=-999)

# Fail if any mesh cell has no NRCS coverage
assert -999 not in soil_color_mukey, \
    'Some mesh cells have no NRCS coverage (nodata=-999). Check watershed extent vs SSURGO.'

nrcs.set_index('mukey', drop=False, inplace=True)

unique_mukeys = list(np.unique(soil_color_mukey))
nrcs = nrcs.loc[unique_mukeys]  # keep only MUKEYs that appear on the mesh

# Assign sequential ATS IDs starting at 1000
nrcs['ATS ID'] = range(1000, 1000 + len(unique_mukeys))
nrcs.set_index('ATS ID', drop=True, inplace=True)

# Build the ATS-ID indexed soil color array
soil_color = -np.ones_like(soil_color_mukey)
for ats_id, mukey in zip(nrcs.index, nrcs['mukey']):
    soil_color[soil_color_mukey == mukey] = ats_id

m2.cell_data['soil_color'] = soil_color
print(f'{len(nrcs)} unique NRCS soil types on mesh, ATS IDs {nrcs.index.min()}–{nrcs.index.max()}')

sc_indices, sc_cmap, sc_norm, sc_ticks, sc_labels = \
    watershed_workflow.plot.createIndexedColormap(nrcs.index)

fig, ax = newPanelFigure()
mp = m2.plot(facecolors=m2.cell_data['soil_color'], cmap=sc_cmap, norm=sc_norm,
             edgecolors=None, colorbar=False, ax=ax)
ax.set_aspect('equal', 'datalim')
# No colorbar -- with this many soil units, a legend of raw ATS IDs
# conveys no useful information on a slide; the map is spatial pattern only.
finishPanel(fig, ax, 'NRCS soil type')
fig.savefig(toOutput('coweeta_nrcs_soil_type.png'), dpi=150, transparent=True)
plt.close(fig)

# Save subsurface properties CSV
output_filenames['subsurface_properties'] = toOutput(f'{name}_subsurface_properties.csv')
nrcs.to_csv(output_filenames['subsurface_properties'])
print('Saved:', output_filenames['subsurface_properties'])


# ## Section 4: Extrude 2D Mesh to 3D
# 
# Uses ELM's default exponential 15-layer vertical grid (`elm_default_dzsoi`, ~42 m total depth).
# All 15 layers get the same NRCS soil type as the surface cell (vertically homogeneous).
# Mat IDs 1000–(1000+N) for ELM nlevsoi layers; same IDs continue for layers 11-15 since
# ELM duplicates layer-10 texture internally.

# ELM vertical grid -- uses ELM's default exponential formula (initVerticalMod.F90)
dzs = watershed_workflow.io.elm.elm_default_dzsoi(nlevgrnd=15)

print(f'Number of layers: {len(dzs)}')
print(f'Layer thicknesses: {np.round(dzs, 4)}')
print(f'Total depth: {dzs.sum():.2f} m')


ncells = m2.num_cells
soil_color_arr = m2.cell_data['soil_color'].values  # shape (ncells,), ATS IDs

# All 15 layers get the surface cell's NRCS ATS ID (vertically homogeneous)
layer_mat_ids = [soil_color_arr.copy() for _ in range(15)]

watershed_workflow.Mesh3D.summarizeExtrusion(
    ['constant'] * 15, dzs.tolist(), [1] * 15, layer_mat_ids)

m3 = watershed_workflow.Mesh3D.extruded_Mesh2D(
    m2, 'constant', dzs.tolist(), [1] * 15, layer_mat_ids)

print('2D labeled sets')
for ls in m2.labeled_sets:
    print(f'  {ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print('\n3D labeled sets')
for ls in m3.labeled_sets:
    print(f'  {ls.setid} : {ls.entity} : {len(ls.ent_ids)} : "{ls.name}"')

print('\n3D side sets')
for ss in m3.side_sets:
    print(f'  {ss.setid} : FACE : {len(ss.cell_list)} : "{ss.name}"')

# Partition the 2D mesh and write lat/lon cell data for ELM column matching
centroids = m2.centroids[:, 0:2]
lon, lat = watershed_workflow.utils.warp.warpXY(
    centroids[:, 0], centroids[:, 1], m2.crs, watershed_workflow.crs.latlon_crs)

m2.cell_data['longitude'] = lon
m2.cell_data['latitude']  = lat

output_filenames['mesh_h5'] = toOutput(f'{mesh_name}.h5')

# should put this capability somewhere in the library, but its specific to both ATS and HDF?
with h5py.File(output_filenames['mesh_h5'], 'w') as fid:
    grp1 = fid.create_group('longitude.cell.0')
    grp1.create_dataset('0', data=lon)

    grp2 = fid.create_group('latitude.cell.0')
    grp2.create_dataset('0', data=lat)

print('Wrote:', output_filenames['mesh_h5'])

# Write 3D mesh exodus file
output_filenames['mesh'] = toOutput(f'{mesh_name}.exo')
try:
    os.remove(output_filenames['mesh'])
except FileNotFoundError:
    pass
m3.writeExodus(output_filenames['mesh'], 'one block')
print('Wrote:', output_filenames['mesh'])


# ## Section 5: ELM Domain and Surface Data Files
# 
# - `writeDomain()` writes the ELM domain NetCDF (lat/lon/area/mask/frac per cell)
# - `SurfdataBuilder` clones the Oak Harbor template surfdata, replaces geometry and land
#   cover, then fills soil texture and porosity from NRCS sand/clay/porosity for the 10
#   ELM `nlevsoi` layers (`PCT_SAND`/`PCT_CLAY`/`PCT_GRVL`/`ORGANIC`/`WATSAT`). ELM
#   internally duplicates layer-10 values for layers 11-15 (SoilStateType.F90 720-731).
# 
# These files are shared across every full-mesh ELM run in the spinup workflow (stages 2
# and 3 in `examples/coweeta/build_example.sh`) -- they are built once here, from the
# same mesh and NRCS data used throughout this script, so they always match the ATS
# mesh exactly. `distribute_pft_finidat.py` also reads `coweeta_surfdata.nc` directly
# (its `PCT_NAT_PFT` array) to determine each full-mesh cell's PFT type when broadcasting
# stage 1c's restart onto stage 2's finidat. Stage 1c instead runs on its own synthetic,
# much smaller domain/surfdata pair (`coweeta_1c_domain.nc` / `coweeta_1c_surfdata.nc`,
# built further below).

output_filenames['elm_domain'] = toOutput(f'{name}_domain.nc')
watershed_workflow.io.elm.writeDomain(m2, output_filenames['elm_domain'])
print('Wrote ELM domain:', output_filenames['elm_domain'])

# Build the SurfdataBuilder (requires 'pft index' in m2.cell_data)
builder = watershed_workflow.io.elm.SurfdataBuilder(m2, template_surfdata)
nlevsoi = builder._nlevsoi  # should be 10
print(f'Template nlevsoi: {nlevsoi}')

# Build soil_type array: shape (nlevsoi=10, ncells)
# All 10 layers get the surface cell's NRCS ATS ID
soil_type = np.tile(soil_color_arr, (nlevsoi, 1))  # (10, ncells)

# soil_properties DataFrame indexed by ATS ID
# setSoilProperties needs 'total sand pct [%]' and 'total clay pct [%]'.
# 'porosity [-]' is included too, written out as WATSAT -- the same NRCS/
# Rosetta porosity used for ATS's own base_porosity in stage 1a's
# add_soil_type() call, so ELM's from-file watsat matches ATS's porosity
# exactly (see watsat_type='from_file' in the user_nl_elm writers below).
soil_props_for_elm = nrcs[['total sand pct [%]', 'total clay pct [%]', 'porosity [-]']].copy()
if 'total gravel pct [%]' in nrcs.columns:
    soil_props_for_elm['total gravel pct [%]'] = nrcs['total gravel pct [%]']

builder.setSoilProperties(soil_type, soil_props_for_elm)
print('Set soil properties for', nlevsoi, 'layers x', ncells, 'cells')

output_filenames['elm_surfdata'] = toOutput(f'{name}_surfdata.nc')
builder.write(output_filenames['elm_surfdata'])
print('Wrote ELM surfdata:', output_filenames['elm_surfdata'])


# ## Stage 3b: MODIS LAI -> Satellite Phenology
#
# Stage 3b (see `examples/coweeta/build_example.sh`) is a fully coupled ELM+ATS
# transient run like stage 3, but runs ELM in Satellite Phenology (SP) mode
# (`use_cn=False`, compset `ICBELMBC`) instead of prognostic CN -- LAI/SAI are
# prescribed from a MODIS-derived monthly climatology per PFT rather than
# predicted from carbon allocation. This sidesteps stages 1c/2's CN spinup
# entirely: stage 3b restarts ATS pressure directly from stage 1a's checkpoint
# and starts ELM cold (no finidat -- SP mode carries no carbon/nitrogen state).
#
# The cells below build that MODIS-derived climatology:
# 1. Fetch MODIS LAI + land cover (LULC) over the watershed via
#    `ManagerMODISEarthdata` (see `watershed_workflow/examples/get_MODIS_LAI.ipynb`).
# 2. Reduce MODIS LULC's own time dimension to a single dominant class per pixel
#    (`computeMode`), then compute a MODIS-class-keyed LAI time series
#    (`computeTimeSeries`).
# 3. Correlate MODIS LULC against the NLCD raster already fetched in Section 2
#    (`computeCrosswalk`), and use that correlation to translate the MODIS-class
#    LAI time series into an NLCD-class-keyed one (`applyCrosswalk`).
# 4. Average across years and collapse to a repeating 12-month climatology
#    (`filterLeapDay` + `computeAverageYear`, then a monthly groupby).
# 5. Reduce NLCD classes to ELM PFT indices using the same `mapNLCDToPFT` table
#    Section 2 uses for `pft_index`, weighting by how many mesh cells carry each
#    NLCD code (`np.unique(m2_nlcd, return_counts=True)`), producing
#    `lai_month_pft_df`: index 1-12 (month), columns = PFT indices present on
#    the mesh -- exactly the shape `SurfdataBuilder.setPhenology(lai=...)` expects.
#
# `builder_3b` then clones the same mesh/template/soil setup as the main `builder`
# above, but with `use_cn=False` and `setPhenology(lai=lai_month_pft_df, ...)`,
# writing a separate `coweeta_stage3b_surfdata.nc` (the BGC-mode `builder` above
# has no phenology tables and is unused by stage 3b).

import watershed_workflow.utils.data

# MODIS query window -- 2010-2014, matching stage 3b's actual simulated
# period (build_example.sh's STAGE3_STARTDATE/STAGE3_STOP_N, ~2010-10-01 to
# 2013-10-01, comfortably inside this range). Although atm_forcing_transient's
# real AORC coverage extends to 2022 (see coweeta_aorc_elm.ipynb), stage 3b's
# own run length is bounded to 2010-2014 to match the largest continuous
# MODIS LAI/LULC download already cached for Coweeta at the time this was
# written (a fresh 2010-2022 MODIS pull is a large, slow first-time download).
modis_start_year = 2010
modis_end_year = 2014
modis_startdate = f'{modis_start_year}-1-1'
modis_enddate = f'{modis_end_year}-12-31'

modis_source = watershed_workflow.sources.ManagerMODISEarthdata()
modis_task = modis_source.requestDataset(
    watershed.exterior, watershed.crs, modis_startdate, modis_enddate, out_crs=crs)
modis_data = modis_source.fetchDataset(modis_task)
print('Fetched MODIS variables:', list(modis_data.data_vars))

# MODIS LULC is itself time-varying (annual land-cover product); collapse to a
# single dominant class per pixel via mode, matching get_MODIS_LAI.ipynb.
modis_data['LULC'] = watershed_workflow.utils.computeMode(modis_data['LULC'], 'time_LULC')
modis_data = modis_data.rename({'time_LAI': 'time'})

# Dynamic per-MODIS-class LAI time series, clipped to the Coweeta watershed.
modis_lai_df = watershed_workflow.properties.land_cover.computeTimeSeries(
    modis_data['LAI'], modis_data['LULC'],
    polygon=watershed.exterior, polygon_crs=watershed.crs)
print('MODIS-class LAI columns:', list(modis_lai_df.columns))

# Correlate MODIS LULC against the NLCD raster already fetched in Section 2
# (`nlcd`), then translate MODIS-class LAI into NLCD-class-keyed LAI. Method
# 'maximal area' picks each NLCD class's single best-correlated MODIS class,
# matching the crosswalk get_MODIS_LAI.ipynb demonstrates.
modis_nlcd_crosswalk = watershed_workflow.properties.land_cover.computeCrosswalk(
    modis_data['LULC'], nlcd, method='maximal area')

nlcd_codes_on_mesh, nlcd_counts_on_mesh = np.unique(m2_nlcd, return_counts=True)

nlcd_lai_df = watershed_workflow.properties.land_cover.applyCrosswalk(
    modis_nlcd_crosswalk, modis_lai_df, unique_nlcd=list(nlcd_codes_on_mesh))
watershed_workflow.properties.land_cover.removeNullLAI(nlcd_lai_df)
print('NLCD-class LAI columns:', list(nlcd_lai_df.columns))

# Average across all MODIS years to a single repeating annual climatology
# (noleap calendar, matching ELM's), then collapse to monthly means -- SP mode
# only supports a 12-month cycle, not year-by-year LAI.
nlcd_lai_noleap = watershed_workflow.utils.filterLeapDay(nlcd_lai_df)
nlcd_lai_typical = watershed_workflow.utils.data.computeAverageYear(
    nlcd_lai_noleap, '2000-01-01', output_nyears=1)

nlcd_lai_typical = nlcd_lai_typical.copy()
# nlcd_lai_typical['time'] holds cftime.DatetimeNoLeap objects (noleap
# calendar), which pd.DatetimeIndex cannot ingest directly -- pull .month
# off each cftime object instead of routing through pandas' datetime64.
nlcd_lai_typical['month'] = [t.month for t in nlcd_lai_typical['time']]
nlcd_lai_columns = [c for c in nlcd_lai_typical.columns if c not in ('time', 'month')]
nlcd_lai_monthly = nlcd_lai_typical.groupby('month')[nlcd_lai_columns].mean()
print('NLCD-class monthly LAI climatology, shape:', nlcd_lai_monthly.shape)

# Reduce NLCD-class-keyed monthly LAI to PFT-keyed monthly LAI via the same
# mapNLCDToPFT table Section 2 uses for pft_index, weighting by how many mesh
# cells carry each NLCD code (so a PFT dominated by a large-area NLCD class
# isn't diluted by a tiny sliver of another NLCD class mapping to the same PFT).
nlcd_code_to_name = {code: watershed_workflow.sources.manager_nlcd.colors[int(code)][0]
                      for code in nlcd_codes_on_mesh}
nlcd_code_to_count = dict(zip(nlcd_codes_on_mesh.tolist(), nlcd_counts_on_mesh.tolist()))
nlcd_code_to_pft = {int(code): int(watershed_workflow.properties.land_cover.mapNLCDToPFT(
    int(code), climate_zone='temperate')) for code in nlcd_codes_on_mesh}

pft_present = sorted(set(nlcd_code_to_pft.values()))
lai_month_pft = np.zeros((12, len(pft_present)))
pft_col_index = {pft: i for i, pft in enumerate(pft_present)}

for pft in pft_present:
    codes_for_pft = [code for code, p in nlcd_code_to_pft.items() if p == pft]
    weights = np.array([nlcd_code_to_count[code] for code in codes_for_pft], dtype=float)
    weights /= weights.sum()
    col_names = [f'{nlcd_code_to_name[code]} LAI [-]' for code in codes_for_pft]
    values = nlcd_lai_monthly[col_names].values  # (12, len(codes_for_pft))
    lai_month_pft[:, pft_col_index[pft]] = values @ weights

lai_month_pft_df = pd.DataFrame(lai_month_pft, index=range(1, 13), columns=pft_present)
print('MODIS-derived monthly LAI climatology by PFT:')
print(lai_month_pft_df)

# SP-mode surfdata: same mesh/template/soil setup as the main (BGC-mode)
# builder above, but use_cn=False and with the MODIS-derived monthly LAI
# climatology set via setPhenology(). MODIS's standard LAI product has no
# companion SAI/canopy-height band, so those fall back to setPhenology()'s
# template nearest-neighbor default (sai=True/height_top=True/height_bot=True)
# rather than fabricating data.
builder_3b = watershed_workflow.io.elm.SurfdataBuilder(m2, template_surfdata, use_cn=False)
builder_3b.setSoilProperties(soil_type, soil_props_for_elm)
builder_3b.setPhenology(lai=lai_month_pft_df, sai=True, height_top=True, height_bot=True)

output_filenames['elm_surfdata_stage3b'] = toOutput(f'{name}_stage3b_surfdata.nc')
builder_3b.write(output_filenames['elm_surfdata_stage3b'])
print('Wrote SP-mode ELM surfdata:', output_filenames['elm_surfdata_stage3b'])


# ## Stage 1c: Spatially-Reduced Per-PFT ELM Carbon Spinup
#
# A 200-year accelerated-decomposition carbon spinup on a small mesh: one flat,
# single-PFT gridcell per PFT actually present in the Coweeta domain (from Section 2's
# `pft_index` crosswalk output), each assumed 100% covered by that PFT, driven by
# cyclic forcing (see `writeUserNlElmStage1c()` below). Running this on N << 2701
# gridcells instead of the full mesh is what makes a 200-year accelerated-decomposition
# spinup affordable at all.
#
# This stage's restart is stage 2's ELM initial condition: each of these per-PFT
# columns' full state (carbon/nitrogen/phosphorus pools, vegetation biomass) is
# broadcast onto every full-mesh cell that shares its PFT type by
# `examples/coweeta/distribute_pft_finidat.py`, run after this stage's `case.submit`
# (see `examples/coweeta/build_example.sh`'s "Deriving stage 2 finidat" block). The
# `coweeta_1c_pft_crosswalk.json` file written below records the PFT type of each of
# this stage's columns for that script to consume.
#
# **No ATS involvement at all**: no ATS XML, no exodus mesh, no
# `domain_decomp_type='ats'`. This is a pure `ELM_USRDAT` run, structurally close to
# the hand-authored Oak Harbor column examples (`examples/oakharbor_column/`), but
# built here via `watershed_workflow` so it inherits Coweeta's actual soil texture
# (from the mesh cell nearest the domain centroid) rather than a generic template
# soil.
#
# All N synthetic cells share the *same* centroid position (the real mesh cell nearest
# the domain's mean centroid) -- `ELM`'s domain/surfdata consistency checks
# (`surfrdMod.F90`) compare `lat`/`lon` between the two files at matching *gridcell
# index*, not for distinctness between different gridcells, so there is no need to
# separate the cells spatially; only their PFT assignment differs. (Note this also
# means the 1c restart's own lon/lat is the shared synthetic centroid, not the real
# full-mesh cells' positions -- `distribute_pft_finidat.py` overwrites lon/lat with
# each target cell's real coordinates when broadcasting, rather than copying 1c's
# placeholder position.)
#

# One flat gridcell per PFT actually present in Coweeta (from Section 2's crosswalk),
# each 100% covered by that PFT. Position: the real mesh cell nearest the domain's
# mean centroid (in projected coords), reused for every synthetic cell and for its
# representative soil material ID -- not an arbitrary/offset point.
stage1c_pft_indices = np.unique(pft_index)
n_1c = len(stage1c_pft_indices)
print(f'Stage 1c: {n_1c} PFTs present in Coweeta: {stage1c_pft_indices}')

domain_centroid = m2.centroids[:, 0:2].mean(axis=0)
nearest_cell = np.argmin(np.linalg.norm(m2.centroids[:, 0:2] - domain_centroid, axis=1))
cell_center = m2.centroids[nearest_cell, 0:2]
cell_material_id = soil_color_arr[nearest_cell]
print(f'Representative cell: index {nearest_cell}, center {cell_center}, '
      f'material ID {cell_material_id}')

# Build N small flat square cells, all centered at cell_center (no offset -- see
# markdown above), each ~10m x 10m -- area is otherwise arbitrary for an ELM-only
# run (no lateral coupling).
half_side = 5.0  # m
coords_1c = []
conn_1c = []
for i in range(n_1c):
    base = len(coords_1c)
    cx, cy = cell_center
    coords_1c.extend([
        (cx - half_side, cy - half_side, 0.0),
        (cx + half_side, cy - half_side, 0.0),
        (cx + half_side, cy + half_side, 0.0),
        (cx - half_side, cy + half_side, 0.0),
    ])
    conn_1c.append([base, base + 1, base + 2, base + 3])
coords_1c = np.array(coords_1c)

m2_1c = watershed_workflow.Mesh2D(
    coords_1c, conn_1c, crs=m2.crs,
    cell_data=pd.DataFrame({'pft index': stage1c_pft_indices}))

print('Stage 1c mesh:', m2_1c.num_cells, 'cells')

# Crosswalk consumed by examples/coweeta/distribute_pft_finidat.py: maps each stage 1c
# column (1-based, matching cols1d_gridcell_index/pfts1d_column_index in 1c's own
# restart file) to the PFT type it represents. This is fully redundant with 1c's own
# restart file (cols1d_ityp + pfts1d_itypveg/wtxy already identify each column's active
# PFT), but is written here explicitly so the distribute script has a single
# authoritative source for "which 1c column represents PFT N" without needing to
# re-derive it from restart-file internals, and so this mapping is visible/diffable
# independent of any particular restart file's timestamp.
output_filenames['pft_crosswalk'] = toOutput('coweeta_1c_pft_crosswalk.json')
crosswalk = {
    'description': ('Maps 1-based stage-1c column index to the ELM natural PFT type '
                     '(0-16) it represents. Column i (1-based) in the 1c restart file '
                     'corresponds to PFT stage1c_pft_indices[i-1].'),
    'pft_indices': [int(p) for p in stage1c_pft_indices],
}
with open(output_filenames['pft_crosswalk'], 'w') as f:
    json.dump(crosswalk, f, indent=2)
print('Wrote stage 1c PFT crosswalk:', output_filenames['pft_crosswalk'])

output_filenames['elm_1c_domain'] = toOutput(f'{name}_1c_domain.nc')
watershed_workflow.io.elm.writeDomain(m2_1c, output_filenames['elm_1c_domain'])
print('Wrote stage 1c ELM domain:', output_filenames['elm_1c_domain'])

builder_1c = watershed_workflow.io.elm.SurfdataBuilder(m2_1c, template_surfdata)
nlevsoi_1c = builder_1c._nlevsoi

# Same representative material ID (cell_material_id, from the real mesh cell nearest
# the domain centroid) for all layers and all N synthetic cells.
soil_type_1c = np.full((nlevsoi_1c, n_1c), cell_material_id)
builder_1c.setSoilProperties(soil_type_1c, soil_props_for_elm)

output_filenames['elm_1c_surfdata'] = toOutput(f'{name}_1c_surfdata.nc')
builder_1c.write(output_filenames['elm_1c_surfdata'])
print('Wrote stage 1c ELM surfdata:', output_filenames['elm_1c_surfdata'])


# ## Stage 1a: ATS-only Steady-State Spinup
# 
# This writes the ATS input file for **stage 1a** of the multi-stage spinup workflow
# (see `examples/coweeta/build_example.sh`): a pure ATS run, uncoupled from ELM, that
# drives the mesh built above with a spatially uniform, constant-in-time precipitation
# rate until subsurface + surface flow reach steady state.
# 
# Unlike Stage 2's coupled XML (spliced from `elm_ats_template.xml`), this one is built
# from `steadystate-template.xml` -- the same ATS-only template
# `coweeta_ats.ipynb`'s `write_spinup_steadystate()` uses -- via `ats_input_spec`'s
# `get_main()`/`populate_basic_properties()` pattern, rather than by pruning ELM-only
# pieces out of the coupled template. An earlier version of this script built stage
# 1a's XML by loading `elm_ats_template.xml` and popping ELM-only evaluators
# (`surface-evaporation`, transpiration, etc.) out of `state/evaluators`; that missed
# the fact that `elm_ats_template.xml`'s `observations` block *also* references those
# same evaluators (e.g. an `evaporation [m d^-1]` observable pointing at
# `surface-evaporation`), so a pruned coupled template still asked State to create
# evaluators that no longer existed
# (`Evaluator "surface-evaporation" @ "" cannot be created in State`). Building from
# `steadystate-template.xml` avoids this whole class of bug, since that template never
# had ELM-only evaluators or observations to begin with.
# 
# Because this run is a precursor to the ELM-coupled stage 2 (which restarts ATS
# pressure from this run's `checkpoint_final.h5`), the mesh partitioning here must match
# ELM's domain decomposition -- so `partitioner = 'from exodus file'` is set explicitly
# below. This is *not* set in `steadystate-template.xml` itself, since ATS-only uses of
# that template (e.g. `coweeta_ats.ipynb`) don't need to match any external
# decomposition and should keep using ATS's own default partitioner (Zoltan RCB).
# 
# - porosity comes from NRCS directly (`add_soil_type`'s `porosity` argument), since
#   there is no ELM to provide `base_porosity` at runtime here
# - forcing is a uniform, constant-in-time `surface-precipitation`. This notebook does
#   not download meteorology itself (that's `coweeta_aorc_elm.ipynb`'s job, so AORC is
#   only fetched once); instead the precip value is left as a placeholder token,
#   `STEADYSTATE_PRECIP_MPS`, that `build_example.sh` substitutes at build time from
#   `elm_output_data/coweeta_mean_precip_rain_mps.txt` (written by
#   `coweeta_aorc_elm.ipynb`)
# - no land cover / canopy physics (steadystate-template.xml has none -- this is a pure
#   subsurface + overland flow problem, no ET)
# 
# Output: `coweeta_stage1a.xml`, to be copied into `examples/coweeta/1a_ats_spinup/`.
# 

# Mesh paths, written as a DIN_LOC_CAMPAIGN/... placeholder rather than a
# notebook-relative path -- build_example.sh sed's DIN_LOC_CAMPAIGN to the
# campaign's actual shared inputdata/ directory when it copies this XML into
# each stage's case dir, so the placeholder must appear literally (as its own
# path component) rather than as a relative climb that depends on case-dir
# nesting depth, which differs per stage.
rel_exo_1a = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.exo')
rel_h5_1a  = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.h5')

steadystate_template = toInput('steadystate-template.xml')

print('Mesh exo (DIN_LOC_CAMPAIGN placeholder):', rel_exo_1a)
print('Mesh h5  (DIN_LOC_CAMPAIGN placeholder):', rel_h5_1a)


# ### Shared mesh/region/observation builder (all ELM+ATS stages)
# 
# `add_domains_elm_ats()`, `build_main_elm_ats()`, and `populate_basic_properties_elm_ats()`
# below are used identically by stages 1a, 2, and 3 -- mirroring `coweeta_ats.ipynb`'s
# `add_domains()`/`get_main()`/`populate_basic_properties()` pattern (its `README`/cells
# 97-101), but specialized for ELM+ATS coupling: no snow/canopy domains (ELM owns those
# natively in every ELM+ATS stage, unlike `coweeta_ats.ipynb`'s own ATS-only runs), and
# `porosity_from_ats` toggles whether ATS or ELM supplies `base_porosity` at runtime.
# 
# Building `mesh`/`regions`/`observations` from scratch and fully replacing (never
# inheriting piecemeal from a template) is required, not just tidier: a `partitioner`
# parameter set anywhere other than inside `mesh->domain` is silently ignored by
# `ats_mesh_factory.cc`'s `createMesh()`/`Mesh_MSTK::read_plist_()` (which only reads
# `partitioner` from the specific named mesh sublist passed to the mesh factory, not the
# outer `mesh` list) -- `elm_ats_template.xml`'s own top-level `mesh->partitioner` was
# exactly this bug, silently falling back to ATS's default `metis` partitioner instead of
# `from exodus file`, producing an ELM/ATS domain-decomposition mismatch. Building
# `mesh->domain->partitioner` explicitly via `add_domains_elm_ats()` for every stage, and
# fully replacing (not merging) `mesh` when splicing into any template, avoids this
# silently reappearing.

def add_domains_elm_ats(main_list, mesh_file):
    """Add subsurface + surface domains for an ELM+ATS coupled or ATS-only-IC
    stage. No snow/canopy domains -- ELM owns those natively in every ELM+ATS
    stage, unlike coweeta_ats.ipynb's own ATS-only runs (contrast its
    add_domains(snow=True, canopy=True) defaults).

    Also sets mesh->domain->partitioner explicitly here, not left to whatever
    template this main_list eventually gets spliced into: the ATS mesh factory
    only reads "partitioner" from the specific named mesh sublist (mesh->domain),
    not the outer mesh list, so it must live at this exact nesting level to be
    read at all -- see the markdown cell above for why this matters.
    """
    ats_input_spec.public.add_domain(
        main_list, domain_name='domain', dimension=3,
        mesh_type='read mesh file', mesh_args={'file': mesh_file})
    main_list['mesh']['domain']['build columns from set'] = 'surface'
    main_list['mesh']['domain']['partitioner'] = 'from exodus file'
    ats_input_spec.public.add_domain(
        main_list, domain_name='surface', dimension=2,
        mesh_type='surface', mesh_args={'surface sideset name': 'surface'})


def build_main_elm_ats(mesh_file, porosity_from_ats):
    """Build an ATS "main" input spec from scratch: domains, labeled-set
    regions from the extruded 3D mesh, per-soil-type WRM/permeability/porosity,
    and a single river-discharge observation. Shared by stages 1a, 2, and 3 --
    the same get_main()/add_domains()/add_soil_type() pattern coweeta_ats.ipynb's
    get_main() uses (its cell defining get_main()), reimplemented here since
    this script doesn't import that notebook's helpers directly.

    porosity_from_ats: True for stage 1a (no ELM to provide base_porosity at
    runtime, so ATS's own soil porosity is used); False for stages 2/3 (ELM
    provides base_porosity via ExternalModelATS.F90 at runtime).
    """
    main_list = ats_input_spec.public.get_main()
    add_domains_elm_ats(main_list, mesh_file)

    # Labeled sets from the extruded 3D mesh (river corridor, stream order,
    # outlet, etc. -- everything add_domain's own 'computational domain' /
    # 'surface domain' / boundary regions don't already cover).
    for ls in m3.labeled_sets:
        ats_input_spec.public.add_region_labeled_set(
            main_list, ls.name, ls.setid, mesh_file, ls.entity)
    for ss in m3.side_sets:
        ats_input_spec.public.add_region_labeled_set(
            main_list, ss.name, ss.setid, mesh_file, 'FACE')

    # Soil properties per NRCS/Rosetta soil type. add_soil_type's own region
    # is named "material ID {label}" and is what the WRM/permeability/porosity
    # evaluators built alongside it reference -- independent of any region
    # names a template being spliced into might otherwise define.
    for ats_id in nrcs.index:
        props = nrcs.loc[ats_id]
        region_name = f'soil type {ats_id}'
        smoothing = 0.01 if props['van Genuchten n [-]'] < 1.5 else 0.0

        ats_input_spec.public.add_soil_type(
            main_list, region_name, label=ats_id, filename=mesh_file,
            porosity=(float(props['porosity [-]']) if porosity_from_ats else None),
            permeability=float(props['permeability [m^2]']),
            van_genuchten_alpha=float(props['van Genuchten alpha [Pa^-1]']),
            van_genuchten_n=float(props['van Genuchten n [-]']),
            residual_sat=float(props['residual saturation [-]']),
            smoothing_interval=float(smoothing),
        )

    # Single flux-out-of-domain observation at the real outlet region (built
    # by addWatershedAndOutletRegions earlier in this script) -- not the
    # full add_observations_water_balance() suite, which adds several
    # atmospheric-forcing-driven observables (e.g. surface-precipitation_rain)
    # that only exist as evaluators in a standalone ATS run driven by its own
    # met forcing, not in the ELM-coupled evaluator graph stages 2/3 use
    # (where ELM supplies water fluxes directly via the coupling interface).
    obs = ats_input_spec.public.add_observation(
        main_list, 'river_discharge', 'river_discharge.csv', time_units='d',
        obs_args={'times start period stop': [0., 1., -1.], 'times start period stop units': 'd'})
    ats_input_spec.public.add_observable(
        obs, 'river discharge [mol d^-1]', 'surface-water_flux', 'surface domain outlet',
        'extensive integral', 'face', time_integrated=True)

    return main_list


def populate_basic_properties_elm_ats(xml, main_xml, include_observations=True):
    """Splice a from-scratch-built main_xml (see build_main_elm_ats above) into
    a template-loaded xml: full replace for mesh/regions/observations (never
    inherited piecemeal from the template -- see markdown cell above for why),
    merge-by-name for model parameters/evaluators (the template's own physics
    config for these, e.g. WRM defaults, PK-specific evaluators, is kept where
    main_xml doesn't define an equivalently-named entry).

    include_observations=False skips the observations replace, leaving
    whatever xml's own template already has there untouched -- for a
    hypothetical stage that builds main_xml via the same build_main_elm_ats()
    call as every other stage (so it always includes water-balance
    observations) but never actually runs ATS timestepping, so those
    observations would never be written to. No current stage uses this.

    Direct port of coweeta_ats.ipynb's populate_basic_properties(), minus the
    land cover types replace block -- no ELM+ATS stage calls add_land_cover
    (no canopy; see add_domains_elm_ats above).
    """
    xml.replace('mesh', asearch.child_by_name(main_xml, 'mesh'))
    xml.replace('regions', asearch.child_by_name(main_xml, 'regions'))

    if include_observations:
        obs_idx = next(i for (i, el) in enumerate(xml) if el.get('name') == 'observations')
        xml[obs_idx] = asearch.child_by_name(main_xml, 'observations')

    xml_mp = asearch.find_path(xml, ['state', 'model parameters'], no_skip=True)
    for parlist in asearch.find_path(main_xml, ['state', 'model parameters'], no_skip=True):
        try:
            xml_mp.replace(parlist.getName(), parlist)
        except aerrors.MissingXMLError:
            xml_mp.append(parlist)

    xml_ev = asearch.find_path(xml, ['state', 'evaluators'], no_skip=True)
    for elist in asearch.find_path(main_xml, ['state', 'evaluators'], no_skip=True):
        try:
            xml_ev.replace(elist.getName(), elist)
        except aerrors.MissingXMLError:
            xml_ev.append(elist)


print('Defined add_domains_elm_ats(), build_main_elm_ats(), and populate_basic_properties_elm_ats().')

# Build stage 1a's ATS "main" input spec via the shared helper above.
# porosity_from_ats=True since there's no ELM in this stage to provide
# base_porosity at runtime.
main_props_1a = build_main_elm_ats(rel_exo_1a, porosity_from_ats=True)

# Uniform, constant-in-time surface-precipitation over the whole surface
# domain. A dummy numeric value is used here -- this script does not
# download meteorology itself (that's coweeta_aorc_elm.ipynb's job, so AORC
# is only fetched once). The dummy is overwritten with a placeholder token
# directly in the XML tree below (see the "write" cell), which
# build_example.sh later substitutes with 0.6x the domain/time-mean AORC rain
# rate written by coweeta_aorc_elm.ipynb to
# Coweeta_Campaign0/inputdata/coweeta_mean_precip_rain_mps.txt (the 0.6x
# factor roughly accounts for ET, the same rule of thumb
# coweeta_ats.ipynb's write_spinup_steadystate() uses).
precip_ev_1a = main_props_1a['state']['evaluators'].append_empty('surface-precipitation')
precip_ev_1a.set_type('independent variable constant',
                      ats_input_spec.public.known_specs['evaluator-independent-variable-constant-spec'])
precip_ev_1a['value'] = 0.0   # dummy; overwritten with a placeholder token before writing

main_xml_1a = ats_input_spec.io.to_xml(main_props_1a)

print(f'Built domains + region + WRM + permeability + porosity entries for '
      f'{len(nrcs)} soil types, plus a uniform surface-precipitation evaluator '
      f'and water-balance observations.')

# Load steadystate-template.xml and splice in the mesh/regions/soil/precip/
# observations built above via the shared helper.
xml_1a = aio.fromFile(steadystate_template)
populate_basic_properties_elm_ats(xml_1a, main_xml_1a)

print('Spliced mesh, regions, observations, WRM parameters, permeability, '
      'base_porosity, and surface-precipitation into steadystate-template.xml.')

# Overwrite the dummy surface-precipitation value with a placeholder token
# (bypassing type-checked setValue, since 'double' Parameters normally must
# hold a numeric value) that build_example.sh substitutes at build time.
precip_placeholder = 'STEADYSTATE_PRECIP_MPS'

precip_value_el = asearch.find_path(
    xml_1a, ['state', 'evaluators', 'surface-precipitation', 'value'], no_skip=True)
precip_value_el.set('value', precip_placeholder)

# Write the stage 1a ATS XML
output_filenames['ats_xml_stage1a'] = toOutput(f'{name}_stage1a.xml')
aio.toFile(xml_1a, output_filenames['ats_xml_stage1a'])
print('Wrote stage 1a ATS XML:', output_filenames['ats_xml_stage1a'])


# ## Writing `user_nl_elm`
#
# `user_nl_elm` is ELM's own input file, exactly like the ATS XML written above and
# below -- so it belongs here, in this script, rather than as a heredoc in the general
# CIME-orchestration script `build_example.sh`. The cells below define a small helper,
# `elmRestartPath()`, and three functions, `writeUserNlElmStage1c/2/3()`, modeled on
# `coweeta_ats.ipynb`'s `write_transient()` / `write_spinup_steadystate()` pattern:
# plain functions parameterized by case directory name (and, for stage 3, the
# *previous* stage's case directory name and run length), computing every path as a
# relative offset from the case directory -- confirmed safe, since the C++ ATS reader
# does no `chdir` before resolving `ats_inputdir`/`ats_inputfile`.
#
# Case directory names (`1a_ats_spinup`, `1c_pft_columns`, `2_cyclic_steadystate`,
# `3_transient`) and each ELM-running stage's `RUN_STARTDATE`/`STOP_N` (1c: start year 1,
# 200 years; 2: start year 1, 10 years) are therefore duplicated here and in
# `build_example.sh`'s own `STAGE1C_START_YEAR`/`STAGE1C_STOP_N`/etc. -- both sides need
# to agree on the same convention, but there is no runtime dependency between them:
# `build_example.sh` never needs to have actually run a prior stage to build the next
# one, since `finidat`'s restart-file name is fully determined by the *simulated*
# model calendar (`RUN_STARTDATE year + STOP_N`), not by wall-clock/submission time --
# except for stage 2, whose `finidat` is instead a fixed `DIN_LOC_CAMPAIGN`-relative
# path to the PFT-broadcast file `examples/coweeta/distribute_pft_finidat.py` produces
# from stage 1c's *actual* restart output (see that stage's section below), since that
# file's content genuinely does depend on stage 1c having actually been run.
#
# `writeUserNlElmStage1c()` writes `user_nl_elm_stage1c` (cyclic forcing,
# `nyears_ad_carbon_only = 200`, `spinup_mortality_factor = 10`, no
# `use_ats`/`ats_inputfile`, no `finidat` -- cold start).
#

def elmRestartPath(case_dir_name, start_year, stop_n):
    """Relative path (from a sibling case directory) to an ELM restart file.

    ELM restart files are named <case>.elm.r.<YYYY>-01-01-00000.nc, where YYYY
    is the *simulated* model calendar date at write time (RUN_STARTDATE year +
    STOP_N) -- not a wallclock or submission timestamp. This mirrors
    build_example.sh's own elmRestartFile() bash function; the two must agree
    on case directory names and RUN_STARTDATE/STOP_N per stage.
    """
    restart_year = start_year + stop_n
    return os.path.join('..', '..', case_dir_name, 'run',
                         f'{case_dir_name}.elm.r.{restart_year:04d}-01-01-00000.nc')


def writeUserNlElmStage1c(surfdata_filename):
    """Write user_nl_elm for stage 1c: spatially-reduced, one-flat-column-per-PFT
    ELM carbon spinup, cyclic forcing, accelerated decomposition, cold start
    (no finidat). No ATS at all -- no domain_decomp_type/ats_inputdir/
    ats_inputfile -- and fsurdat points at the synthetic per-PFT surfdata file
    instead of the full Coweeta domain."""
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! Porosity (watsat) read directly from fsurdat's WATSAT variable (NRCS/
 ! Rosetta porosity), rather than computed via Cosby et al. 1984 pedotransfer
 ! from PCT_SAND/PCT_CLAY -- matches the porosity ATS itself uses (see
 ! build_main_elm_ats()'s add_soil_type() calls in this script).
 watsat_type = 'from_file'

 ! Cyclic spinup forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_spinup'
 const_climate_hist = .true.
 ! atm_forcing_spinup is real hourly AORC data (see computeTypicalYear() in
 ! coweeta_aorc_elm.ipynb) with a genuine diurnal cycle already present in
 ! each hourly record, so plain linear interpolation between records is
 ! appropriate -- the default 'zenith' FSDS interpolant is designed for
 ! forcing coarser than the model timestep (e.g. 3-hourly) where the
 ! diurnal shape needs to be reconstructed from period averages; applied
 ! to already-hourly data it doesn't help and isn't needed.
 metdata_fsds_interpolant = 'linear'

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'

 ! Accelerated decomposition carbon spinup
 nyears_ad_carbon_only = 200
 spinup_mortality_factor = 10

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 stream_fldfilename_popdens = '$DIN_LOC_ROOT/lnd/clm2/firedata/clmforc.Li_2012_hdm_0.5x0.5_AVHRR_simyr1850-2010_c130401.nc'
 stream_fldfilename_ndep = '$DIN_LOC_ROOT/lnd/clm2/ndepdata/fndep_elm_cbgc_exp_simyr1849-2101_1.9x2.5_c190103.nc'

 ! DIAGNOSTIC: pin N deposition to a single fixed year (1850) instead of
 ! letting it advance through the real 1850-2005 historical record while
 ! ELM's own clock runs 200 cyclic-forcing years -- confirmed via
 ! NDEP_TO_SMINN history output that the un-pinned stream drifts smoothly
 ! year over year (not clamped, contrary to initial expectation) and is a
 ! candidate driver of unexplained interannual carbon-pool/LAI variation in
 ! accelerated-decomposition spinups. Spatially already constant across
 ! Coweeta's domain (confirmed via NDEP_TO_SMINN being bit-identical across
 ! all columns), so no synthetic spatially-uniform file is needed -- only the
 ! time dimension needs pinning. Remove these three lines once the
 ! hypothesis is confirmed/refuted.
 stream_year_first_ndep = 1850
 stream_year_last_ndep = 1850
 model_year_align_ndep = 1

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage1c'] = toOutput('user_nl_elm_stage1c')
    with open(output_filenames['user_nl_elm_stage1c'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage1c'])
    return content


writeUserNlElmStage1c(os.path.basename(output_filenames['elm_1c_surfdata']))


# ## Stage 2: ELM+ATS Cyclic Steady-State Spinup
#
# Stage 2 of the multi-stage spinup workflow (see `examples/coweeta/build_example.sh`)
# runs the fully coupled ELM+ATS system to cyclic steady state (~10 years of repeated
# "typical year" forcing), initialized from the stage 1a ATS checkpoint and a PFT-
# broadcast ELM restart derived from stage 1c.
#
# Builds the coupled ATS input by loading `elm_ats_template.xml` and injecting:
# 1. Mesh filename (coweeta_np<NTASKS>.exo / coweeta_np<NTASKS>.h5)
# 2. Labeled set regions from m3
# 3. Per-soil-type WRM parameters (van Genuchten alpha/n/Sr, permeability) from NRCS/Rosetta
#
# Porosity (`base_porosity` in ATS's `add_soil_type`) is **not** set here — ELM reads
# WATSAT from fsurdat (`watsat_type = 'from_file'`, set in `user_nl_elm_stage2` below,
# same NRCS/Rosetta `porosity [-]` values stage 1a passes ATS directly) and passes it to
# ATS at runtime via ExternalModelATS.F90.  The `compressible porosity` evaluator in the
# template XML uses ELM's `base_porosity` directly.
#
# This mesh/region/WRM construction is shared by **stages 2 and 3** (both couple ELM to
# ATS with the same PK tree and soil properties), so the cells below build one XML object
# that Stage 3 (further down) reuses, making an independent second copy with a different
# restart source. They differ only in **initial conditions**: each stage's `flow` PK must
# restart ATS's pressure field from the *previous* stage's final checkpoint (following the
# same pattern `coweeta_ats.ipynb`'s `write_transient()` uses) --
#
# - **Stage 2** restarts ATS pressure from stage 1a's `checkpoint_final.h5` (the
#   ATS-only steady-state spinup above). ELM's carbon/nitrogen/phosphorus/biomass state
#   for stage 2 instead comes from `examples/coweeta/distribute_pft_finidat.py`, which
#   broadcasts stage 1c's per-PFT restart onto the full mesh by PFT type and is run by
#   `build_example.sh` after stage 1c's `case.submit` -- see that stage's markdown
#   above. `finidat` in `user_nl_elm_stage2` (written below by `writeUserNlElmStage2()`)
#   points at that script's fixed output filename, not at any case's own restart
#   directory.
# - **Stage 3** (below) restarts ATS pressure from stage 2's `checkpoint_final.h5`, and
#   ELM's `finidat` comes from stage 2's own ELM restart (a real previous-case restart,
#   unlike stage 2's own IC).
#
# Contrast with the stage 1a XML above, which is a self-contained, ELM-free steady-state
# spinup with its own cold-start (`hydrostatic head`) initial condition -- stage 1a has
# no previous stage to restart from.
#
# **ATS input**: writes `coweeta_stage2.xml`, with the `flow` PK's `initial conditions`
# restarting ATS pressure from stage 1a's `checkpoint_final.h5`
# (`../../1a_ats_spinup/run/checkpoint_final.h5`, relative to the stage 2 run directory).
# `build_example.sh` copies this file into the stage 2 case directory and points
# `ats_inputfile` at it.
#
# **ELM namelist**: written below by `writeUserNlElmStage2()` -- cyclic, "typical year"
# AORC forcing (`DIN_LOC_CAMPAIGN/atm_forcing_spinup`, the same directory used by stage 1c),
# `use_ats = .true.`, `domain_decomp_type = 'ats'`, `ats_inputfile = 'coweeta_stage2.xml'`,
# `nyears_ad_carbon_only = 0` (unlike stage 1c -- carbon pools are already spun up), and
# `finidat = 'DIN_LOC_CAMPAIGN/coweeta_stage2_finidat.nc'` -- a fixed filename, not a
# stage-and-restart-year-derived one like stage 3's below, since this file is a
# freestanding NetCDF `distribute_pft_finidat.py` constructs (not any case's own restart
# output) and always lands at this same path in the campaign's shared inputdata/.
# `build_example.sh` correspondingly sets `RUN_TYPE=startup` for this case (not
# `hybrid`), since there is no real reference case to restart from any more.
#

# Mesh paths, written as a DIN_LOC_CAMPAIGN/... placeholder (see the stage 1a
# mesh-path cell above for why) -- build_example.sh sed's this to the
# campaign's actual shared inputdata/ directory for both stages 2 and 3,
# which reuse this same mesh/region/WRM xml object.
rel_exo = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.exo')
rel_h5  = os.path.join('DIN_LOC_CAMPAIGN', f'{mesh_name}.h5')

print('Mesh exo (DIN_LOC_CAMPAIGN placeholder):', rel_exo)
print('Mesh h5  (DIN_LOC_CAMPAIGN placeholder):', rel_h5)

# Build stage 2/3's ATS "main" input spec via the shared helper above (same
# mesh/region/WRM/observation construction used by stage 1a). porosity_from_ats=False:
# ELM provides base_porosity via ExternalModelATS.F90 at runtime for these
# coupled stages, unlike stage 1a.
main_props = build_main_elm_ats(rel_exo, porosity_from_ats=False)
main_xml = ats_input_spec.io.to_xml(main_props)

print(f'Built domains + region + WRM + permeability entries for {len(nrcs)} '
      f'soil types, plus the river-discharge observation.')

# Load the template XML and replace the MESH_FILENAME placeholders via string
# substitution, then splice in the mesh/regions/soil/observations built above
# via the shared helper -- same full-replace pattern as stage 1a, not the
# additive region-merge this cell used to do (see markdown cell above stage
# 1a's build_main_elm_ats definition for why a full replace, not a merge, is
# required for mesh->domain->partitioner to actually be read by ATS).
with open(ats_xml_template, 'r') as f:
    xml_str = f.read()

xml_str = xml_str.replace('MESH_FILENAME.exo', rel_exo)
xml_str = xml_str.replace('MESH_FILENAME.h5',  rel_h5)

xml = aio.fromString(xml_str)
populate_basic_properties_elm_ats(xml, main_xml)

print('Loaded template and spliced mesh, regions, observations, WRM parameters, '
      'and permeability (shared by stages 2 and 3).')

# Write coweeta_stage2.xml: same mesh/region/WRM xml built above, with the
# flow PK's initial conditions restarting ATS pressure from stage 1a's
# checkpoint. Stage 2's case dir is a sibling of stage 1a's, both one level
# below the campaign dir, so the restart path climbs back up to the campaign
# dir and down into stage 1a's own run/ directory (matching the mesh path
# convention above and the pattern used by coweeta_ats.ipynb's
# write_transient()).
restart_path_stage2 = os.path.join('..', '..', '1a_ats_spinup', 'run', 'checkpoint_final.h5')

stage2_xml_str = aio.toString(xml)
stage2_xml = aio.fromString(stage2_xml_str)  # independent deep copy via round-trip

# Replace the subsurface flow PK's cold-start initial conditions
# (hydrostatic head) with a restart from stage 1a's checkpoint. Overland
# flow's initial conditions are left as-is ('initialize surface head from
# subsurface'): it always derives its state from the just-restarted
# subsurface pressure rather than getting its own restart file, matching the
# pattern in cyclic_steadystate-template.xml / transient-template.xml.
flow_ic = asearch.find_path(stage2_xml, ['PKs', 'flow', 'initial conditions'], no_skip=True)
for child in list(flow_ic):
    flow_ic.remove(child)
flow_ic.setParameter('restart file', 'string', restart_path_stage2)

output_filenames['ats_xml_stage2'] = toOutput(f'{name}_stage2.xml')
aio.toFile(stage2_xml, output_filenames['ats_xml_stage2'])
print('Wrote stage2 ATS XML (restart from', restart_path_stage2 + '):',
      output_filenames['ats_xml_stage2'])

def writeUserNlElmStage2(surfdata_filename, ats_xml_filename, finidat_filename):
    """Write user_nl_elm for stage 2: coupled ELM+ATS cyclic steady-state.

    finidat restarts ELM state from stage 1c's restart, broadcast onto the
    full mesh by PFT type (see examples/coweeta/distribute_pft_finidat.py) --
    a freestanding file in the campaign's shared inputdata/, not any case's
    own restart directory, so (unlike stage 3's finidat below) this is a
    fixed filename, not one computed from a case directory name/run length.
    ATS pressure restart is instead baked into ats_xml_filename's own
    'restart file' parameter (see the ATS XML write cell above), so it is
    not repeated here.
    """
    finidat_path = f'DIN_LOC_CAMPAIGN/{finidat_filename}'
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! Porosity (watsat) read directly from fsurdat's WATSAT variable -- see
 ! watsat_type comment in writeUserNlElmStage1c() above.
 watsat_type = 'from_file'

 ! ELM IC from stage 1c's restart, broadcast onto the full mesh by PFT type
 ! (see examples/coweeta/distribute_pft_finidat.py)
 finidat = '{finidat_path}'

 ! Cyclic spinup forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_spinup'
 const_climate_hist = .true.
 ! atm_forcing_spinup is real hourly AORC data (see computeTypicalYear() in
 ! coweeta_aorc_elm.ipynb) with a genuine diurnal cycle already present in
 ! each hourly record, so plain linear interpolation between records is
 ! appropriate -- see writeUserNlElmStage1c() above for the full rationale.
 metdata_fsds_interpolant = 'linear'

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'

 ! No accelerated decomposition in coupled spinup
 nyears_ad_carbon_only = 0
 spinup_mortality_factor = 1

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 use_ats = .true.
 use_ats_hydro = .true.
 domain_decomp_type = 'ats'
 ats_inputdir = '..'
 ats_inputfile = '{ats_xml_filename}'

 stream_fldfilename_popdens = '$DIN_LOC_ROOT/lnd/clm2/firedata/clmforc.Li_2012_hdm_0.5x0.5_AVHRR_simyr1850-2010_c130401.nc'
 stream_fldfilename_ndep = '$DIN_LOC_ROOT/lnd/clm2/ndepdata/fndep_elm_cbgc_exp_simyr1849-2101_1.9x2.5_c190103.nc'

 ! DIAGNOSTIC: pin N deposition to a single fixed year (1850), same as stage
 ! 1c -- see writeUserNlElmStage1c() above for the rationale. Cyclic met
 ! forcing alone does not stop N deposition from advancing through the real
 ! 1850-2005 historical record year over year, so this stage needs the same
 ! pinning stage 1c uses to avoid reintroducing that inter-annual driver in
 ! the coupled cyclic steady-state. Remove these three lines once the
 ! hypothesis is confirmed/refuted.
 stream_year_first_ndep = 1850
 stream_year_last_ndep = 1850
 model_year_align_ndep = 1

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage2'] = toOutput('user_nl_elm_stage2')
    with open(output_filenames['user_nl_elm_stage2'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage2'])
    return content


writeUserNlElmStage2(
    os.path.basename(output_filenames['elm_surfdata']),
    os.path.basename(output_filenames['ats_xml_stage2']),
    finidat_filename='coweeta_stage2_finidat.nc')


# ## Stage 3: ELM+ATS Transient Run
# 
# Stage 3 of the multi-stage spinup workflow (see `examples/coweeta/build_example.sh`) is
# the final, fully coupled ELM+ATS transient run, initialized from the end of stage 2 and
# driven by real (non-cyclic) meteorology.
# 
# **ATS input**: reuses the same mesh/region/WRM `xml` object built in the Stage 2 section
# above (identical PK tree and soil properties -- only the restart source differs), making
# an independent second copy and writing `coweeta_stage3.xml`, with the `flow` PK's
# `initial conditions` restarting ATS pressure from stage 2's `checkpoint_final.h5`
# (`../../../2_cyclic_steadystate/run/checkpoint_final.h5`, relative to the stage 3 run
# directory).
# 
# **ELM namelist**: written below by `writeUserNlElmStage3()` -- transient AORC cpl_bypass
# forcing (`metdata_type = 'atssubdaily'`, `const_climate_hist = .false.`, pointing at
# `DIN_LOC_CAMPAIGN/atm_forcing_transient`), `use_ats = .true.`, `domain_decomp_type = 'ats'`,
# `ats_inputfile = 'coweeta_stage3.xml'`. The AORC zone files cover 2010-2022 at 1-hour
# resolution; `zone_mappings.txt` maps every Coweeta cell to zone 19. ELM's carbon/biomass
# state restarts from stage 2 via `finidat` (`../../2_cyclic_steadystate/run/....elm.r....nc`),
# computed via `elmRestartPath()` the same way stage 3's ATS checkpoint path is derived --
# a real previous case's own restart directory, unlike stage 2's own finidat (see the
# Stage 2 section above for why that one is a fixed filename instead).
#
# `user_nl_elm` for every ELM-running stage (1c, 2, 3) is written entirely by this
# notebook (`writeUserNlElmStage1c/2/3()`, defined in the "Writing user_nl_elm" section
# above), not by `build_example.sh` -- `user_nl_elm` is ELM's own input file, exactly like
# the ATS XML, so it belongs alongside the other domain-specific input generation here
# rather than in a general CIME-orchestration script. `build_example.sh` just copies the
# finished file into each case directory before `case.setup`, the same way it already
# copies the ATS XMLs.
#

# Write coweeta_stage3.xml: the same mesh/region/WRM xml built in the Stage 2
# section above, making an independent second copy with the flow PK's
# initial conditions instead restarting ATS pressure from stage 2's
# checkpoint.
restart_path_stage3 = os.path.join('..', '..', '2_cyclic_steadystate', 'run', 'checkpoint_final.h5')

stage3_xml_str = aio.toString(xml)
stage3_xml = aio.fromString(stage3_xml_str)  # independent deep copy via round-trip

flow_ic = asearch.find_path(stage3_xml, ['PKs', 'flow', 'initial conditions'], no_skip=True)
for child in list(flow_ic):
    flow_ic.remove(child)
flow_ic.setParameter('restart file', 'string', restart_path_stage3)

output_filenames['ats_xml_stage3'] = toOutput(f'{name}_stage3.xml')
aio.toFile(stage3_xml, output_filenames['ats_xml_stage3'])
print('Wrote stage3 ATS XML (restart from', restart_path_stage3 + '):',
      output_filenames['ats_xml_stage3'])

def writeUserNlElmStage3(surfdata_filename, ats_xml_filename,
                          prev_case_dir_name, prev_start_year, prev_stop_n):
    """Write user_nl_elm for stage 3: coupled ELM+ATS transient run.

    finidat restarts ELM state from stage 2; ATS pressure restart is baked
    into ats_xml_filename's own 'restart file' parameter (see the ATS XML
    write cell above).
    """
    finidat_path = elmRestartPath(prev_case_dir_name, prev_start_year, prev_stop_n)
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! Porosity (watsat) read directly from fsurdat's WATSAT variable -- see
 ! watsat_type comment in writeUserNlElmStage1c() above.
 watsat_type = 'from_file'

 ! ELM IC from end of stage 2
 finidat = '{finidat_path}'

 ! Transient forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_transient'
 const_climate_hist = .false.
 ! atm_forcing_transient is the real hourly AORC record (see
 ! coweeta_aorc_elm.ipynb) with a genuine diurnal cycle already present in
 ! each hourly record, so plain linear interpolation between records is
 ! appropriate -- see writeUserNlElmStage1c() above for the full rationale.
 metdata_fsds_interpolant = 'linear'

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'

 nyears_ad_carbon_only = 0
 spinup_mortality_factor = 1

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 use_ats = .true.
 use_ats_hydro = .true.
 domain_decomp_type = 'ats'
 ats_inputdir = '..'
 ats_inputfile = '{ats_xml_filename}'

 stream_fldfilename_popdens = '$DIN_LOC_ROOT/lnd/clm2/firedata/clmforc.Li_2012_hdm_0.5x0.5_AVHRR_simyr1850-2010_c130401.nc'
 stream_fldfilename_ndep = '$DIN_LOC_ROOT/lnd/clm2/ndepdata/fndep_elm_cbgc_exp_simyr1849-2101_1.9x2.5_c190103.nc'

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage3'] = toOutput('user_nl_elm_stage3')
    with open(output_filenames['user_nl_elm_stage3'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage3'])
    return content


writeUserNlElmStage3(
    os.path.basename(output_filenames['elm_surfdata']),
    os.path.basename(output_filenames['ats_xml_stage3']),
    prev_case_dir_name='2_cyclic_steadystate', prev_start_year=1, prev_stop_n=10)


# ## Stage 3b: ELM+ATS Transient Run, Satellite Phenology Mode
#
# Stage 3b (see `examples/coweeta/build_example.sh`) is a second, independent
# transient run alongside stage 3 -- fully coupled ELM+ATS, driven by the same
# real transient AORC meteorology -- but runs ELM in Satellite Phenology (SP)
# mode instead of prognostic CN, using the MODIS-derived monthly LAI
# climatology built above (`lai_month_pft_df`, `coweeta_stage3b_surfdata.nc`).
# This sidesteps stages 1c/2's CN spinup entirely: stage 3b restarts ATS
# pressure directly from **stage 1a**'s checkpoint (skipping stage 2's coupled
# spinup) and starts ELM cold -- no `finidat` -- since SP mode carries no
# carbon/nitrogen state to initialize, only physical soil moisture/temperature,
# which equilibrates under real transient forcing.
#
# **ATS input**: reuses the same mesh/region/WRM `xml` object built in the
# Stage 2 section above (identical PK tree and soil properties -- only the
# restart source differs), making an independent third copy and writing
# `coweeta_stage3b.xml`, with the `flow` PK's `initial conditions` restarting
# ATS pressure from stage 1a's `checkpoint_final.h5`
# (`../../1a_ats_spinup/run/checkpoint_final.h5`, relative to the stage 3b run
# directory) -- not stage 2's or stage 3's checkpoint.
#
# **ELM namelist**: written below by `writeUserNlElmStage3b()` -- same
# transient AORC forcing as stage 3, but `fsurdat` points at the SP-mode
# `coweeta_stage3b_surfdata.nc` (built above, carries `MONTHLY_LAI`/etc.), no
# `finidat` (cold start), and omits the CN/CNDV-only settings stage 3 needs
# (`stream_fldfilename_popdens`/`stream_fldfilename_ndep`, accelerated-
# decomposition params) -- SP mode errors if those streams are set (see
# `examples/oakharbor_column/user_nl_elm`, the working SP-mode example
# elsewhere in this repo). `build_example.sh` correspondingly builds this case
# with `COMPSET=ICBELMBC` (selects SP mode intrinsically, same compset
# `oakharbor_column` uses) instead of the campaign's default CN compset, and
# `RUN_TYPE=startup` (no ELM reference case, matching stage 2's rationale).

# Write coweeta_stage3b.xml: the same mesh/region/WRM xml built in the Stage 2
# section above, making an independent copy with the flow PK's initial
# conditions instead restarting ATS pressure from stage 1a's checkpoint --
# not stage 2's or stage 3's, since stage 3b skips the coupled cyclic
# steady-state spinup entirely.
restart_path_stage3b = os.path.join('..', '..', '1a_ats_spinup', 'run', 'checkpoint_final.h5')

stage3b_xml_str = aio.toString(xml)
stage3b_xml = aio.fromString(stage3b_xml_str)  # independent deep copy via round-trip

flow_ic = asearch.find_path(stage3b_xml, ['PKs', 'flow', 'initial conditions'], no_skip=True)
for child in list(flow_ic):
    flow_ic.remove(child)
flow_ic.setParameter('restart file', 'string', restart_path_stage3b)

output_filenames['ats_xml_stage3b'] = toOutput(f'{name}_stage3b.xml')
aio.toFile(stage3b_xml, output_filenames['ats_xml_stage3b'])
print('Wrote stage3b ATS XML (restart from', restart_path_stage3b + '):',
      output_filenames['ats_xml_stage3b'])

def writeUserNlElmStage3b(surfdata_filename, ats_xml_filename):
    """Write user_nl_elm for stage 3b: coupled ELM+ATS transient run in
    Satellite Phenology (SP) mode.

    No finidat -- SP mode carries no carbon/nitrogen state, and ELM starts
    cold; physical soil moisture/temperature equilibrates under real
    transient forcing. ATS pressure restart is baked into ats_xml_filename's
    own 'restart file' parameter (see the ATS XML write cell above), pointed
    at stage 1a's checkpoint rather than stage 2's. fsurdat points at the
    SP-mode surfdata file (carries MONTHLY_LAI/etc., built above), not the
    shared BGC-mode surfdata stages 1c/2/3 use. Omits
    stream_fldfilename_popdens/stream_fldfilename_ndep and the accelerated-
    decomposition params -- all CN/CNDV-only; SP mode errors if the popdens/
    ndep streams are set (see examples/oakharbor_column/user_nl_elm).
    """
    content = f"""
 fsurdat = 'DIN_LOC_CAMPAIGN/{surfdata_filename}'

 ! Porosity (watsat) read directly from fsurdat's WATSAT variable -- see
 ! watsat_type comment in writeUserNlElmStage1c() above.
 watsat_type = 'from_file'

 ! Transient forcing
 metdata_type = 'atssubdaily'
 metdata_bypass = 'DIN_LOC_CAMPAIGN/atm_forcing_transient'
 const_climate_hist = .false.
 ! atm_forcing_transient is the real hourly AORC record (see
 ! coweeta_aorc_elm.ipynb) with a genuine diurnal cycle already present in
 ! each hourly record, so plain linear interpolation between records is
 ! appropriate -- see writeUserNlElmStage1c() above for the full rationale.
 metdata_fsds_interpolant = 'linear'

 aero_file = '$DIN_LOC_ROOT/atm/cam/chem/trop_mozart_aero/aero/aerosoldep_monthly_1850_mean_1.9x2.5_c090421.nc'

 do_harvest = .false.
 do_transient_pfts = .false.
 flanduse_timeseries = ''
 use_nofire = .true.

 use_ats = .true.
 use_ats_hydro = .true.
 domain_decomp_type = 'ats'
 ats_inputdir = '..'
 ats_inputfile = '{ats_xml_filename}'

 check_finidat_fsurdat_consistency = .false.
 check_finidat_year_consistency = .false.
"""
    output_filenames['user_nl_elm_stage3b'] = toOutput('user_nl_elm_stage3b')
    with open(output_filenames['user_nl_elm_stage3b'], 'w') as f:
        f.write(content)
    print('Wrote:', output_filenames['user_nl_elm_stage3b'])
    return content


writeUserNlElmStage3b(
    os.path.basename(output_filenames['elm_surfdata_stage3b']),
    os.path.basename(output_filenames['ats_xml_stage3b']))


# ## Summary and Next Steps
# 
# After this script completes, its `elm_output_data/` directory holds everything
# needed for the multi-stage spinup workflow -- no copy step required.
# `build_example.sh` (in `examples/coweeta/`) and its `stage_inputdata.sh` helper
# read directly from `watershed_workflow/examples/Coweeta/elm_output_data/` and
# stage what they need into the campaign's shared `inputdata/` automatically.
# 
# `user_nl_elm` -- ELM's own input file -- is written entirely by this script
# (`writeUserNlElmStage1c/2/3()`, defined in the "Writing user_nl_elm"/Stage 2/3
# sections above), exactly like the ATS XMLs, rather than as a heredoc in
# `build_example.sh`:
# `user_nl_elm` and the ATS XML are both domain-specific input generation that
# belongs alongside the rest of this script's mesh/region/soil work, not in a
# general CIME-orchestration script. `build_example.sh` only copies each
# finished file into its case directory before `case.setup`.
# 
# `build_example.sh` also stages the handful of global ELM datasets (aerosol
# deposition, CO2, population density, N deposition) into the campaign's
# `inputdata/` automatically (via `stage_inputdata.sh`), reading them directly
# from `ELM_ATS_SRC_DIR/inputdata` -- you do not need to clone the full
# `inputdata` repo onto machines that only run an already-built campaign.
# 
# Also run `coweeta_aorc_elm.ipynb` (writes meteorology to the same
# `elm_output_data/` directory), then build the full multi-stage spinup
# workflow (stages 1a, 1c, 2, 3):
#
# ```bash
# cd ../../../examples/coweeta
# export NTASKS=${NTASKS}
# ./build_example.sh
# ```
#
# See `examples/coweeta/README.md` for the full run order (stage 1a's manual
# ATS invocation, then `case.submit` for stages 1c/2/3 in order -- stage 2's
# build step also runs `distribute_pft_finidat.py` against stage 1c's actual
# restart output, so stage 1c's `case.submit` must complete before stage 2 can
# be built, not just before it's run).

print(f'{"role":<35}: filename')
print('-' * 36 + ': ' + '-' * 50)
for k, v in output_filenames.items():
    print(f'{k:<35}: {v}')


