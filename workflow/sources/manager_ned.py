"""Manager for interacting with NED datasets."""
import os,sys
import logging
import numpy as np
import shapely
import rasterio.merge
import requests
import requests.exceptions


import workflow.utils
import workflow.sources.utils as source_utils
import workflow.conf
import workflow.warp
import workflow.sources.names



class FileManagerNED:
    """Watershed Workflow leverages the USGS's National Elevation Dataset (NED), a
    precursor to and currently part of the USGS's 3D Elevation Program (3DEP)
    [NED]_.  It is available seamlessly at a variety of resolutions ranging
    from 2 arc-seconds to 1/3 arc-seconds (~60m and 10m, respectively) in the
    conterminous United States and comparable resolution through most of
    Alaska.  Like the NHD data, these datasets are available through The
    National Map's [TNM]_ REST API, and are provided in 1-degree tiles.
    Watershed Workflow manages querying for URLs, downloading these tiles on
    demand, and forming the mosaic of images through underlying capability in
    rasterio to provide a single raster across the watershed requested.  Higher
    resolution products, including LiDAR products across the conterminous US
    and IfSAR products across Alaska are coming available, but these are not
    currently supported by Watershed Workflow.

    .. [NED]

    Parameters
    ----------
    resolution : str, optional
      Resolution of the desired product.  One of:
      * `"1/3 arc-second`" (default)
      * `"1 arc-second`" 
    file_format : str, optional
      Desired output format.  Default and universally available is `"IMG`".
    """
    def __init__(self, resolution='1/3 arc-second', file_format='IMG'):
        """Create the manager."""
        self.name = 'National Elevation Dataset (NED); resolution: {}'.format(resolution)
        self.file_format = file_format

        if resolution == '1/3 arc-second':
            self.short_res = '13as'
        elif resolution == '1 arc-second':
            self.short_res = '1as'
        else:
            raise ValueError("{}: invalid resolution '{}', must be one of '1/9 arc-second', '1/3 arc-second', or '1 arc-second'".format(self.name))

        self.resolution = resolution        
        self.names = workflow.sources.names.Names(self.name, 'dem', None,
                                                  'USGS_NED_%s_n{:02}_w{:03}.%s'%(self.short_res,file_format.lower()),
                                                  self.short_res+"_raw")
        self.crs = workflow.crs.from_epsg(4269)

    def get_raster(self, shape, crs, force = False):
        """Download and read a DEM for this shape, clipping to the shape.
        
        Parameters
        ----------
        shape : fiona or shapely shape
          Shape to provide bounds of the raster.
        crs : CRS
          CRS of the shape.

        Returns
        -------
        profile : rasterio profile
          Profile of the raster.
        raster : np.ndarray
          Array containing the elevation data.

        Note that the raster provided is in its native CRS (which is in the
        rasterio profile), not the shape's CRS.
        """
        if type(shape) is dict:
            shape = workflow.utils.shply(shape)
        
        # warp to my crs
        shply = workflow.warp.shply(shape, crs, self.crs)

        # get the bounds and download
        bounds = shply.bounds
        feather_bounds = list(bounds[:])
        feather_bounds[0] = feather_bounds[0] - .01
        feather_bounds[1] = feather_bounds[1] - .01
        feather_bounds[2] = feather_bounds[2] + .01
        feather_bounds[3] = feather_bounds[3] + .01
        files = self.download(feather_bounds, force = force)

        # merge into a single raster
        datasets = [rasterio.open(f) for f in files]
        profile = datasets[0].profile
        dest, output_transform = rasterio.merge.merge(datasets, bounds=feather_bounds, nodata=np.nan)
        dest = np.where(dest < -1.e-10, np.nan, dest)

        # set the profile
        profile['transform'] = output_transform
        profile['height'] = dest.shape[1]
        profile['width'] = dest.shape[2]
        profile['count'] = dest.shape[0]
        profile['nodata'] = np.nan
        return profile, dest[0]

    def request(self, bounds):
        """Forms the REST API get to find URLs.

        Parameters
        ----------
        bounds : [xmin, ymin, xmax, ymax]
          Desired bounds, in the raster's native CRS.

        Returns
        -------
        js : json dict
          JSON response of the formed request.
        """
        rest_url = 'https://viewer.nationalmap.gov/tnmaccess/api/products'
        rest_dataset = self.name + ' ' + self.resolution

        rest_bounds = ','.join(str(b) for b in bounds)
        try:
            r = requests.get(rest_url, params={'datasets':rest_dataset,
                                               'bbox':rest_bounds,
                                               'prodFormats':self.file_format})

        except requests.exceptions.ConnectionError as err:
            logging.error('{}: Failed to access REST API for NED DEM products.'.format(self.name))
            raise err

        assert(r.json()['total'] > 0)
        return r.json()

    def download(self, bounds, force=False):
        """Download the files covering the bounds.

        Parameters
        ----------
        bounds : [xmin, ymin, xmax, ymax]
          Desired bounds, in the raster's native CRS.
        force : bool, optional
          If true, re-download even if a file already exists.

        Returns
        -------
        filenames : list(str)
          List of raster files tiling the bounds.
        """
        logging.info("Collecting DEMs to tile bounds: {}".format(bounds))
        
        # check directory structure
        os.makedirs(self.names.data_dir(), exist_ok=True)
        os.makedirs(self.names.raw_folder_name(), exist_ok=True)

        # NOTE: we could get these from the REST API, but I would
        # prefer to not REQUIRE an internet connection if the data
        # already exists.

        # tile the bounds in lat/long 1-degree increments
        west = int(np.floor(bounds[0]))
        south = int(np.floor(bounds[1]))
        east = int(np.ceil(bounds[2]))
        north = int(np.ceil(bounds[3]))

        # generate the list of files needed
        filenames = [self.names.file_name(j+1, -i) for j in range(south, north) for i in range(west, east)]
        logging.info('  Need:')
        for fname in filenames:
            logging.info('    {}'.format(fname))

        filenames_success = []
        if (any(not os.path.exists(f) for f in filenames) or force):

            request = self.request(bounds)
            for r in request['items']:
                url = r['downloadURL']
                north = int(np.round(r['boundingBox']['maxY']))
                west = int(np.round(r['boundingBox']['minX']))

                filename = self.names.file_name(north, -west)
                if filename not in filenames:
                    # randomly some extra matches are found?
                    continue
                
                filenames.remove(filename)

                if not os.path.exists(filename) or force:
                    downloadfilename = url.split("/")[-1]
                    downloadfile = os.path.join(self.names.raw_folder_name(north,west), downloadfilename)
                    assert(downloadfile.endswith('.ZIP') or downloadfile.endswith('.zip'))

                    logging.info("Attempting to download source for target '%s'"%filename)
                    work_dir = self.names.raw_folder_name(north, west)

                    if not os.path.exists(downloadfile) or force:
                        source_utils.download(url, downloadfile, force)
                    source_utils.unzip(downloadfile, work_dir)
                    unzip_filename = downloadfilename[0:-4]

                    # hope we can find it?
                    img_files = []
                    if os.path.isdir(os.path.join(work_dir, unzip_filename)):
                        img_files = [os.path.join(unzip_filename,f) for f in os.listdir(os.path.join(work_dir, unzip_filename)) if f.endswith('.'+self.file_format.lower())]
                        if len(img_files) == 0:
                            img_files = [os.path.join(unzip_filename,f) for f in os.listdir(os.path.join(work_dir, unzip_filename)) if f.endswith('.'+self.file_format.upper())]
                            
                    if len(img_files) == 0:
                        img_files = [f for f in os.listdir(work_dir) if f.endswith('.'+self.file_format.lower())]
                    if len(img_files) == 0:
                        img_files = [f for f in os.listdir(work_dir) if f.endswith('.'+self.file_format.upper())]

                    if len(img_files) == 0:
                        raise RuntimeError("{}: Downloaded and unzipped '{}', but cannot find the img file.".format(self.name, downloadfile))
                    else:
                        logging.debug("  Found '{}'".format(os.path.join(work_dir, img_files[0])))

                    source_utils.move(os.path.join(work_dir, img_files[0]), filename)
                    

                if not os.path.exists(filename):
                    raise RuntimeError('{}: Cannot find or download file for source target "{}"'.format(self.name, filename))
                else:
                    filenames_success.append(filename)
                    
            if len(filenames) != 0:
                logging.warn('Potentially missing tiles in the DEM covering bounds: {}'.format(bounds))
                logging.warn('This may be a REST API error, or it may be that some tiles are oceanic and not needed.')
                logging.warn('Continuing, but consider this issue if some elevation data is missing.')
                logging.warn('Missing Tiles:')
                for fname in filenames:
                    logging.warn('  {}'.format(fname))
        else:
            logging.info('source files already exist!')
            filenames_success = filenames

        return filenames_success
        


