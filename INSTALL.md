# for users
conda create -n ats_meshing -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs


# for developers
conda create -n ats_meshing_dev -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs pytest





