
ivar = 'tmax'
islice = 200

fig, axes = plt.subplots(1, 2)

ax = axes[0]
extent = rasterio.transform.array_bounds(daymet_profile['height'], daymet_profile['width'], daymet_profile['transform']) # (x0, y0, x1, y1)
plot_extent = extent[0], extent[2], extent[1], extent[3]

iraster = raw[ivar][islice, :, :]

with fiona.open(watershed_shapefile, mode='r') as fid:
    bnd_profile = fid.profile
    bnd = [r for (i,r) in fid.items()]
daymet_crs = watershed_workflow.crs.daymet_crs()

# convert to destination crs
native_crs = watershed_workflow.crs.from_fiona(bnd_profile['crs'])
reproj_bnd = watershed_workflow.warp.shape(bnd[0], native_crs, daymet_crs)
reproj_bnd_shply = watershed_workflow.utils.shply(reproj_bnd)

cax = ax.matshow(iraster, extent=plot_extent, alpha=1)
ax.plot(*reproj_bnd_shply.exterior.xy, 'r')
ax.set_title("Raw Daymet")


ax = axes[1]
extent = new_extent # (x0, y0, x1, y1)
plot_extent = extent[0], extent[2], extent[1], extent[3] # (x0, x1, y0, y1)

iraster = new_dat[ivar][islice, :, :]

# set nodata to NaN to avoid plotting
iraster[iraster == -9999] = np.nan

watershed_workflow.plot.hucs(watershed, crs, ax=ax, color='r', linewidth=1)
im = ax.matshow(iraster, extent=plot_extent)
ax.set_title("Reprojected Daymet")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax)



#####################
ivar = 'prcp'
extent = new_extent # (x0, y0, x1, y1)
plot_extent = extent[0], extent[2], extent[1], extent[3] # (x0, x1, y0, y1)
iraster = new_dat[ivar][islice, :, :]
# set nodata to NaN to avoid plotting
iraster[iraster == -9999] = np.nan

fig, ax = plt.subplots(1, 1,figsize=[10,10])

watershed_workflow.plot.hucs(watershed, crs, ax=ax, color='r', linewidth=1)
im = ax.matshow(iraster, extent=plot_extent)
ax.plot(xv, yv, 'or')
ax.plot(xv[0:20], yv[0:20], 'ob')
ax.set_title("Daymet")
cbar_ax = fig.add_axes([.92, 0.25, 0.02, 0.5])
fig.colorbar(im, cax = cbar_ax)

fig.show()

# X, Y = np.meshgrid(new_x, new_y)
# xv = np.reshape(X,(X.size,1))
# yv = np.reshape(Y,(Y.size,1))
# xy = np.concatenate((xv,yv),axis=1)
# watershed_polygon = watershed.polygon(0)
# in_waterhsed_v = [watershed_polygon.contains(shapely.geometry.Point(theCoords)) for theCoords in xy]
# in_waterhsed_m = np.reshape(in_waterhsed_v,X.shape)
# i_inWatershed, j_inWatershed = np.where(in_waterhsed_m)




fig, ax = plt.subplots(1, 1,figsize=[10,10])

# im = ax.matshow(iraster, extent=plot_extent)
im = ax.matshow(in_waterhsed_m, extent=plot_extent)
xb,yb = bounds.exterior.xy
ax.plot(xb,yb)
ax.plot(xv, yv, 'ok')
ax.plot(xv[in_waterhsed], yv[in_waterhsed], 'or')
cbar_ax = fig.add_axes([.92, 0.25, 0.02, 0.5])
fig.colorbar(im, cax = cbar_ax)

fig.show()

# points = shapely.geometry.MultiPoint(xy)






xpoly, ypoly = watershed_polygon.exterior.xy
xpoint, ypoint = points.exterior.xy

map(shapely.geometry.Point, zip(xv, yv))

[watershed_polygon.contains(thePoint) for thePoint in points]
