"""Common data sources."""
import workflow.files

huc_wbd = workflow.files.HucFileSystem(name='NHD High Resolution Water Boundary Data',
                                          url='https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/',
                                          base_folder='hydrologic_units',
                                          folder_template='WBD_{0}_HU2_Shape',
                                          file_template='WBDHU{1}.shp',
                                          digits=2)

dem_ned = workflow.files.LatLonFileSystem(name='National Elevation Dataset',
                                             url='https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/IMG/',
                                             base_folder='dem',
                                             folder_template=None,
                                             file_template='USGS_NED_13_{0}{1}_IMG.img',
                                             download_template='USGS_NED_13_{0}{1}_IMG.zip')

hydro_nhd = workflow.files.HucFileSystem(name='NHD High Resoluton Hydrography',
                                            url='https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/HighResolution/Shape/',
                                            base_folder='hydrography',
                                            folder_template='NHD_H_{0}_HU8_Shape',
                                            file_template='NHDFlowline.shp',
                                            digits=8)


sources_avail = {'HUC.WBD' : huc_wbd,
                 'DEM.NED' : dem_ned,
                 'Hydro.NHD' : hydro_nhd,
                }

def get_sources(args=None):
    if args is not None:
        sources = {'HUC' : workflow.files.FileManager([sources_avail[args.source_huc],]),
                   'DEM' : workflow.files.TiledFileManager([sources_avail[args.source_dem],]),
                   'Hydro' : workflow.files.FileManager([sources_avail[args.source_hydro],]),
                   }
    else:
        sources = {'HUC' : workflow.files.FileManager([huc_wbd,]),
                   'DEM' : workflow.files.TiledFileManager([dem_ned,]),
                   'Hydro' : workflow.files.FileManager([hydro_nhd,]),
                   }
    return sources

    

