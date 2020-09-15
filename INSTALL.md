# for users
conda create -n watershed_workflow -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes ipykernel requests sortedcontainers attrs pysheds jupyterlab 


# for developers
x conda create -n watershed_workflow_dev -c conda-forge -c defaults python=3 ipython numpy matplotlib scipy meshpy fiona rasterio shapely cartopy descartes pysheds jupyterlab ipykernel requests sortedcontainers attrs pytest sphinx nbsphinx sphinx_rtd_theme





