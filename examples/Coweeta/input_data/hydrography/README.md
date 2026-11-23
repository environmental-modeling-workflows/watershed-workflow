`NHDPlusHR_Coweeta.parquet` is the output of the NHDPlus HR hydrography
source for the Coweeta basin, saved so that `coweeta_stream_aligned_mesh.ipynb`
(and CI) do not depend on the USGS NHDPlus HR service.  It is GeoParquet
rather than a shapefile because it carries two geometry columns (the
flowline and its `catchment`).

To regenerate, from the `examples/` directory:

```python
import watershed_workflow, watershed_workflow.sources

crs = watershed_workflow.crs.default_crs
coweeta = watershed_workflow.sources.ManagerShapefile(
    './Coweeta/input_data/coweeta_basin.shp', id_name='BASIN_CODE').getShapes(out_crs=crs)
watershed = watershed_workflow.Watershed(coweeta)

nhd = watershed_workflow.sources.sources['geometry']['NHDPlusHR-pynhd_nhdplushr']
reaches = nhd.getShapesByGeometry(watershed.exterior, crs, out_crs=watershed_workflow.crs.latlon_crs)
reaches.to_parquet('./Coweeta/input_data/hydrography/NHDPlusHR_Coweeta.parquet')
```
