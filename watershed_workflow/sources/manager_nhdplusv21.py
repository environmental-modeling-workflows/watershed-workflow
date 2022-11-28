import os, sys
import logging
import fiona
import shapely
import attr
import requests
import shutil
import numpy as np
from requests_html import HTMLSession
import pandas as pd

import watershed_workflow.sources.utils as source_utils
import watershed_workflow.config
import watershed_workflow.sources.names
import watershed_workflow.warp
import watershed_workflow.utils


class FileManagerNHDPlusV21:
    """Manager for interacting with USGS National Hydrography v2.1, from EPA.

    This works using the EPA's website...

    .. [EPA] https://www.usgs.gov/core-science-systems/ngp/national-hydrography

    """
    def __init__(self):
        """The name manager will use the following nomenclature:

        1) For 'folder_template'
            NHDPlusV21_<dd>_<VPUid>_componentname_<vv>
        2) For 'file_template'
            NHDPlusV21_<dd>_<VPUid>_<RPUid>_<componentname>_<vv>
        
        where 
            dd = the drainage area identifier

            VPUid = the VPU identifier
            
            RPUid = the RPU identifier
            
            componentname = the name of the NHDPlusV2 component contained in the file
            
            vv = the data content version, 01, 02, ... for the component
        """
        self.name = 'NHD Plus Medium Res v2.1 (EPA)'
        self.name_manager = watershed_workflow.sources.names.Names(self.name, 'hydrography',
                                                                   'NHDPlusV21_{}_{}', 'NHDPlusV21_{}_{}_{}_{}_{}')        
        self.boundary_unit_file = self._get_v21_boundary_unit_file()
        self.wbd = watershed_workflow.sources.manager_nhd.FileManagerWBD()
        self._componentnames_vpu_wide = self._componentnames_vpu_wide_main
        self._componentnames_rpu_wide = self._componentnames_rpu_wide_main

    # variables needed from attribute files
    _componentnames_vpu_wide_all = ["EROMExtension", "NHDPlusAttributes",
                                   "NHDPlusBurnComponents", "NHDPlusCatchment",
                                   "NHDSnapshotFGDB", "NHDSnapshot",
                                   "VPUAttributeExtension", "VogelExtension",
                                   "WBDSnapshot"]

    _componentnames_rpu_wide_all = ["CatSeed", "FdrFac", "FdrNull",
                                   "FilledAreas", "HydroDem", "NEDSnapshot"]

    _componentnames_vpu_wide_main = ["EROMExtension", "NHDPlusCatchment",
                                    "NHDPlusAttributes", "NHDSnapshot",
                                    "VPUAttributeExtension", "WBDSnapshot"]

    _componentnames_rpu_wide_main = ["NEDSnapshot"]

    _nhdplus_vaa = dict({
        'StreamOrder': 'StreamOrde',
        'StreamLevel': 'StreamLeve',
        'HydrologicSequence': 'Hydroseq',
        'DownstreamMainPathHydroSeq': 'DnHydroseq',
        'UpstreamMainPathHydroSeq': 'UpHydroseq',
        'DivergenceCode': 'Divergence',
        'CatchmentAreaSqKm': 'AreaSqKM',
        'TotalDrainageAreaSqKm': 'TotDASqKM',
        'FromNode': 'FromNode',
        'ToNode': 'ToNode'
    })

    _nhdplus_elevslope = dict({
        'MinimumElevationSmoothed': 'MINELEVSMO',
        'MaximumElevationSmoothed': 'MAXELEVSMO',
        'MinimumElevationRaw': 'MINELEVRAW',
        'MaximumElevationRaw': 'MAXELEVRAW',
        'Slope': 'SLOPE',
    })

    _nhdplus_eromma = dict({
        'MeanAnnualFlow': 'Q0001E',
        'MeanAnnualVelocity': 'V0001E',
        'MeanAnnualIncrementalFlow': 'Qincr0001E',
        'MeanAnnualFlowGaugeAdj': 'Q0001E'
    })

    _vpu_name_url_info = dict({
        "01": {"url_name": "https://www.epa.gov/waterdata/nhdplus-northeast-data-vector-processing-unit-01", "vpu_name": "Northeast"},
        "02": {"url_name": "https://www.epa.gov/waterdata/nhdplus-mid-atlantic-data-vector-processing-unit-02", "vpu_name": "Mid Atlantic"},
        "03N": {"url_name": "https://www.epa.gov/waterdata/nhdplus-south-atlantic-north-data-vector-processing-unit-03n", "vpu_name": "South Atlantic North"},
        "03S": {"url_name": "https://www.epa.gov/waterdata/nhdplus-south-atlantic-south-data-vector-processing-unit-03s", "vpu_name": "South Atlantic South"},
        "03W": {"url_name": "https://www.epa.gov/waterdata/nhdplus-south-atlantic-west-data-vector-processing-unit-03w", "vpu_name": "South Atlantic West"},
        "04": {"url_name": "https://www.epa.gov/waterdata/nhdplus-great-lakes-data-vector-processing-unit-04", "vpu_name": "Great Lakes"},
        "05": {"url_name": "https://www.epa.gov/waterdata/nhdplus-ohio-data-vector-processing-unit-05", "vpu_name": "Ohio"},
        "06": {"url_name": "https://www.epa.gov/waterdata/nhdplus-tennessee-data-vector-processing-unit-06", "vpu_name": "Tennessee"},
        "07": {"url_name": "https://www.epa.gov/waterdata/nhdplus-upper-mississippi-data-vector-processing-unit-07", "vpu_name": "Upper Mississippi"},
        "08": {"url_name": "https://www.epa.gov/waterdata/nhdplus-lower-mississippi-data-vector-processing-unit-08", "vpu_name": "Lower Mississippi"},
        "09": {"url_name": "https://www.epa.gov/waterdata/nhdplus-souris-red-rainy-data-vector-processing-unit-09", "vpu_name": "Souris-Red-Rainy"},
        "10U": {"url_name": "https://www.epa.gov/waterdata/nhdplus-upper-missouri-data-vector-processing-unit-10u", "vpu_name": "Upper Missouri"},
        "10L": {"url_name": "https://www.epa.gov/waterdata/nhdplus-lower-missouri-data-vector-processing-unit-10l", "vpu_name": "Lower Missouri"},
        "11": {"url_name": "https://www.epa.gov/waterdata/nhdplus-ark-red-white-data-vector-processing-unit-11", "vpu_name": "Ark-Red-White"},
        "12": {"url_name": "https://www.epa.gov/waterdata/nhdplus-texas-data-vector-processing-unit-12", "vpu_name": "Texas"},
        "13": {"url_name": "https://www.epa.gov/waterdata/nhdplus-rio-grande-data-vector-processing-unit-13", "vpu_name": "Rio Grande"},
        "14": {"url_name": "https://www.epa.gov/waterdata/nhdplus-upper-colorado-data-vector-processing-unit-14", "vpu_name": "Upper Colorado"},
        "15": {"url_name": "https://www.epa.gov/waterdata/nhdplus-lower-colorado-data-vector-processing-unit-15", "vpu_name": "Lower Colorado"},
        "16": {"url_name": "https://www.epa.gov/waterdata/nhdplus-great-basin-data-vector-processing-unit-16", "vpu_name": "Great Basin"},
        "17": {"url_name": "https://www.epa.gov/waterdata/nhdplus-pacific-northwest-data-vector-processing-unit-17", "vpu_name": "Pacific Northwest"},
        "18": {"url_name": "https://www.epa.gov/waterdata/nhdplus-california-data-vector-processing-unit-18", "vpu_name": "California"},
        "20": {"url_name": "https://www.epa.gov/waterdata/nhdplus-hawaii-data-vector-processing-unit-20", "vpu_name": "Hawaii"},
        "21": {"url_name": "https://www.epa.gov/waterdata/nhdplus-puerto-rico-us-virgin-islands-data-vector-processing-unit-21", "vpu_name": "Puerto Rico/U.S. Virgin Islands"},
        "22A": {"url_name": "https://www.epa.gov/waterdata/nhdplus-american-samoa-data-vector-processing-unit-22a", "vpu_name": "American Samoa"},
        "22G": {"url_name": "https://www.epa.gov/waterdata/nhdplus-guam-data-vector-processing-unit-22g", "vpu_name": "Guam"},
        "22M": {"url_name": "https://www.epa.gov/waterdata/nhdplus-northern-mariana-islands-data-vector-processing-unit-22m", "vpu_name": "Northern Mariana Islands"}
    })
    
    def _get_v21_boundary_unit_file(self):
        """This just downloads the NHD v2.1 Boundary Unit file, which contains
        VPUs, RPUs, and Drainage Area IDs."""
        # check directory structure
        os.makedirs(self.name_manager.data_dir(), exist_ok=True)
        loc = os.path.join(self.name_manager.data_dir(), 'NHDPlusV21_BoundaryUnit')
        final_loc = os.path.join(loc, 'NHDPlusGlobalData', 'BoundaryUnit.shp')

        if not os.path.isfile(final_loc):
            # download and unzip
            url = 'https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/GlobalData/NHDPlusV21_NHDPlusGlobalData_03.7z'
            source_utils.download(url, loc+'.7z', force=False)
            source_utils.unzip(loc+'.7z', loc)
            assert(os.path.isfile(final_loc))
        return final_loc


    def _get_v21_boundary_units(self, huc, enforce_VPUs = True):
        """Given a list of HUCs, figure out which VPU and HRU and DAID we are in."""
        wbd_profile, wbd_huc = self.wbd.get_huc(huc)
        huc_bounds = watershed_workflow.utils.bounds(wbd_huc)

        with fiona.open(self.boundary_unit_file, 'r') as fid:
            # Get the CRS for the Boundary Units
            BoundaryUnits_crs = watershed_workflow.crs.from_fiona(fid.profile['crs'])

            # Project the watershed boundary to the CRS for the Boundary Units
            bounds = watershed_workflow.warp.bounds(
                huc_bounds, wbd_profile['crs'], BoundaryUnits_crs)

            # Get the boundary Units that intersect with the watershed
            BUs = [r for (i, r) in fid.items(bbox=bounds)]

        # Consolidate information from the selected Boundary Units
        UnitType = []
        UnitID = []
        DrainageID = []
        for pp in BUs:
            UnitType.append(pp['properties']['UnitType'])
            UnitID.append(pp['properties']['UnitID'])
            DrainageID.append(pp['properties']['DrainageID'])

        UnitType = np.array(UnitType)
        UnitID = np.array(UnitID)
        DrainageID = np.array(DrainageID)

        # Find tuples of Drainage Areas, VPUs, and RPUs
        daID_vpu_rpu = []  # list of lists with the Drainage Areas, VPUs, and RPUs
        daID_unique = np.unique(DrainageID)

        for dd in daID_unique:
            vpu_unique = np.unique(
                UnitID[np.argwhere((UnitType == 'VPU') & (DrainageID == dd))])

            for vv in vpu_unique:
                daID_vpu_rpu += [[dd, vv, UnitID[ii]] for ii in range(len(UnitType))
                                if (('RPU' in UnitType[ii]) & (vv[0:2] in UnitID[ii]))]

        if enforce_VPUs:
            print("--------- Enforcing VPUs ---------")
            if isinstance(huc, str):
                enforce_VPUs = np.array([huc[0:2]])
            else: 
                enforce_VPUs = np.unique([tt[0:2] for tt in huc])

            toKeep = np.zeros((1,len(daID_vpu_rpu)), dtype=bool)
            for vpu in enforce_VPUs:
                print(vpu)
                toKeep += [vpu in vv[1] for vv in daID_vpu_rpu]
            daID_vpu_rpu = [i for (i, v) in zip(daID_vpu_rpu, toKeep[0]) if v]

        return daID_vpu_rpu

    def _get_v21_urls(self, daID_vpu_rpu):
        # Get the base URLs
        base_URLs = [self._vpu_name_url_info[tmp[1]]['url_name'] for tmp in daID_vpu_rpu] 
        base_URLs = np.unique(base_URLs).tolist()

        # Check for url verification of certificates?  
        verify = watershed_workflow.config.rcParams['DEFAULT']['ssl_cert']
        logging.info('       cert: "%s"' % verify)
        if verify == "True":
            verify = True
        elif verify == "False":
            verify = False
        
        my_url = []
        my_daID = []
        my_vpu = []
        my_rpu = []
        my_componentname = []
        my_dataversion = []

        for idx, b_url in enumerate(base_URLs):

            daID, vpu, rpu = daID_vpu_rpu[idx]

            data_links = self._get_v21_data_url_from_base_url(b_url, verify=verify)

            # VPU-wide
            url_vpu_wide = [self._get_v21_url_NHD_dataset(data_links, cc)[0] for cc  in self._componentnames_vpu_wide]
            vv_vpu = [uu.split('/')[-1].split('.7z')[0].split('_')[-1] for uu in url_vpu_wide]
            cc_vpu = [uu.split('/')[-1].split('.7z')[0].split('_')[-2] for uu in url_vpu_wide]
            
            my_url += url_vpu_wide
            my_dataversion += vv_vpu
            my_componentname += cc_vpu 
            my_daID += [daID]*len(url_vpu_wide)
            my_vpu += [vpu]*len(url_vpu_wide)
            my_rpu += [vpu]*len(url_vpu_wide)

            # VPU-wide
            url_rpu_wide = [self._get_v21_url_NHD_dataset(data_links, cc)[0] for cc  in self._componentnames_rpu_wide]
            vv_rpu = [uu.split('/')[-1].split('.7z')[0].split('_')[-1] for uu in url_rpu_wide]
            cc_rpu = [uu.split('/')[-1].split('.7z')[0].split('_')[-2] for uu in url_rpu_wide]

            my_url += url_rpu_wide
            my_dataversion += vv_rpu
            my_componentname += cc_rpu 
            my_daID += [daID]*len(url_rpu_wide)
            my_vpu += [vpu]*len(url_rpu_wide)
            my_rpu += [rpu]*len(url_rpu_wide)

        url_data_info_df = pd.DataFrame(list(zip(my_componentname, my_daID, my_vpu, my_rpu,
                my_dataversion, my_url)),
                columns =['component_name', 'drainage_area_ID', 'vpu', 'rpu', 
                'data_version', 'data_url'])
        
        return url_data_info_df
     
    def _get_v21_data_url_from_base_url(self, url, verify=True):

        with HTMLSession() as session:
            response = session.get(url, verify=verify)
            response.raise_for_status()
            status_code = response.status_code  # A status code of 200 means it was accepted
            print("Status code:" + str(status_code))
            html = response.html
            html.render()
            all_links = html.absolute_links
        return [ll for ll in list(all_links) if ".7z" in ll]

    def _get_v21_url_NHD_dataset(self, data_links, nhd_name):
        return [match for match in data_links if nhd_name in match]

    def get_huc(self, huc, force_download=False, enforce_VPUs = True):
        """Get the specified HUC in its native CRS.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code
        enforce_VPUs : bool
            Enforces the download to the VPU that mathces the level 2 hydrologic unit from huc 
            (i.e., first two digits of huc) and not all the VPUs intersecting the rectangle that
            encloses the huc polygon 

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        hu : dict
          Fiona shape object representing the hydrologic unit.

        Note this finds and downloads files as needed.
        """
        huc = source_utils.huc_str(huc)
        profile, hus = self.get_hucs(huc, len(huc), force_download, enforce_VPUs)
        # assert (len(hus) == 1)
        # return profile, hus[0]
        return profile, hus

    def get_hucs(self, huc, level, force_download=False, enforce_VPUs = True):
        """Get all sub-catchments of a given HUC level within a given HUC.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code
        level : int
          Level of requested sub-catchments.  Must be larger or equal to the
          level of the input huc.
        force_download : bool
          Download or re-download the file if true.

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        hus : list(dict)
          List of fiona shape objects representing the hydrologic units.

        Note this finds and downloads files as needed.
        """
        huc = source_utils.huc_str(huc)
        huc_level = len(huc)

        # get drainage area ID, vpu, and rpu info
        self._daID_vpu_rpu = self._get_v21_boundary_units(huc, enforce_VPUs)
        self._url_data_info_df = self._get_v21_urls(self._daID_vpu_rpu)

        # Download the WBD for the NHD Plus V2 dataset
        idx = self._url_data_info_df.index[self._url_data_info_df['component_name'] == 'WBDSnapshot'].tolist()[0]
        url_wbd = self._url_data_info_df['data_url'][idx]
        daID = self._url_data_info_df['drainage_area_ID'][idx]
        vpu = self._url_data_info_df['vpu'][idx]
        rpu = self._url_data_info_df['rpu'][idx]
        cc = self._url_data_info_df['component_name'][idx]
        vv = self._url_data_info_df['data_version'][idx]

        my_wbd_folder_name = self.name_manager.folder_name(daID, vpu)
        my_wbd_file_name = self.name_manager.file_name(daID, vpu, rpu, cc, vv)
        
        # check directory structure
        os.makedirs(my_wbd_folder_name, exist_ok=True)
        os.makedirs(my_wbd_file_name, exist_ok=True)
        os.makedirs(os.path.join(my_wbd_file_name,'raw'), exist_ok=True)
        

        loc = my_wbd_file_name
        final_loc = os.path.join(loc, 'WBD_Subwatershed.shp')

        if not os.path.isfile(final_loc):
            # download and unzip
            location_7z = os.path.join(loc, 'raw')
            location_unzip = os.path.join(location_7z, 'unzipped')
            os.makedirs(location_unzip, exist_ok=True)
            source_utils.download(url_wbd, os.path.join(location_7z, loc.split('/')[-1]+'.7z'), force=False)
            source_utils.unzip(os.path.join(location_7z, loc.split('/')[-1]+'.7z'), location_unzip)

            # Get the directory tree for the files unzipped

            my_dirnames = []
            # my_filenames = []
            for root, dirs, files in os.walk(location_unzip, topdown=True):
                # for name in files:
                #     my_filenames.append(os.path.join(root, name))
                for name in dirs:
                    my_dirnames.append(os.path.join(root, name))

            dir_shps = my_dirnames[-1]
            files_to_move = os.listdir(dir_shps)
            [source_utils.move(os.path.join(dir_shps, ff),loc) for ff in files_to_move]
            shutil.rmtree(location_unzip)
            assert(os.path.isfile(final_loc))

        # read the file
        with fiona.open(final_loc, mode='r') as fid:
            hus = [hu for hu in fid if hu['properties']['HUC_12'].startswith(huc)]
            profile = fid.profile
        profile['always_xy'] = True
        self._path_wbd_shp = final_loc
            # self.name_manager.data_dir() # For example '/Users/8n8/Documents/data_research/hydrography'
            # self.name_manager.folder_name(daID, vpu) # For example '/Users/8n8/Documents/data_research/hydrography/NHDPlusV21_CO_14'
            # self.name_manager.file_name(daID, vpu, rpu, cc_vpu[0], vv_vpu[0]) # For example '/Users/8n8/Documents/data_research/hydrography/NHDPlusV21_CO_14/NHDPlusV21_CO_14_14b_EROMExtension_05'
        return profile, hus







        # # error checking on the levels, require file_level <= huc_level <= level <= lowest_level
        # if self.lowest_level < level:
        #     raise ValueError("{}: files include HUs at max level {}.".format(
        #         self.name, self.lowest_level))
        # if level < huc_level:
        #     raise ValueError("{}: cannot ask for HUs at level {} contained in {}.".format(
        #         self.name, level, huc_level))
        # if huc_level < self.file_level:
        #     raise ValueError(
        #         "{}: files are organized at HUC level {}, so cannot ask for a larger HUC than that level."
        #         .format(self.name, self.file_level))

        # # download the file
        # filename = self._download(huc[0:self.file_level], force=force_download)
        # logging.info('Using HUC file "{}"'.format(filename))

        # # read the file
        # layer = 'WBDHU{}'.format(level)
        # logging.debug("{}: opening '{}' layer '{}' for HUCs in '{}'".format(
        #     self.name, filename, layer, huc))

        # with fiona.open(filename, mode='r', layer=layer) as fid:
        #     hus = [hu for hu in fid if source_utils.get_code(hu, level).startswith(huc)]
        #     profile = fid.profile
        # profile['always_xy'] = True
        # return profile, hus
    

    def get_hydro(self,
                  huc,
                  bounds=None,
                  bounds_crs=None,
                  in_network=True,
                  properties=None,
                  include_catchments=False,
                  force_download=False):
        """Get all reaches within a given HUC and/or coordinate bounds.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code
        bounds : [xmin, ymin, xmax, ymax], optional
          Coordinate bounds to filter reaches returned.  If this is provided,
          bounds_crs must also be provided.
        bounds_crs : CRS, optional
          CRS of the above bounds.
        in_network : bool, optional
          If True (default), remove reaches that are not "in" the NHD network

        properties : list(str) or bool, optional
          A list of property aliases to be added to reaches.  See
          alias names in Table 16 (NHDPlusFlowlineVAA) or 17
          (NHDPlusEROMMA) of NHDPlus User Guide).  This is only
          supported for NHDPlus.  Commonly used properties include: 

           - 'TotalDrainageAreaKmSq' : total drainage area
           - 'CatchmentAreaKmSq' : differential catchment contributing area
           - 'HydrologicSequence' : VAA sequence information
           - 'DownstreamMainPathHydroSeq' : VAA sequence information
           - 'UpstreamMainPathHydroSeq' : VAA sequence information
           - 'catchment' : catchment polygon geometry

          If bool is provided and the value is True, a standard
          default set of VAA and EROMMA attributes are added as
          properties.

        include_catchments : bool, optional 
          If True, adds catchment polygons for each reach in the river tree from 'NHDPlusCatchment' layer
        force_download : bool Download
          or re-download the file if true.

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        reaches : list(dict)
          List of fiona shape objects representing the stream reaches.

        Note this finds and downloads files as needed.

        """
        if properties is True:
            properties = list(self._nhdplus_vaa.keys()) + list(self._nhdplus_elevslope.keys()) + list(self._nhdplus_eromma.keys())

        if 'WBD' in self.name:
            raise RuntimeError(f'{self.name}: does not provide hydrographic data.')

        huc = source_utils.huc_str(huc)
        hint_level = len(huc)

        # try to get bounds if not provided
        if bounds is None:
            # can we infer a bounds by getting the HUC?
            profile, hu = self.get_huc(huc)
            bounds = watershed_workflow.utils.bounds(hu)
            bounds_crs = watershed_workflow.crs.from_fiona(profile['crs'])

        # error checking on the levels, require file_level <= huc_level <= lowest_level
        if hint_level < self.file_level:
            raise ValueError(
                f"{self.name}: files are organized at HUC level {self.file_level}, so cannot ask for a larger HUC level."
            )

        # download the file
        filename = self._download(huc[0:self.file_level], force=force_download)
        logging.info('  Using Hydrography file "{}"'.format(filename))

        # find and open the hydrography layer
        filename = self.name_manager.file_name(huc[0:self.file_level])
        layer = 'NHDFlowline'
        logging.info(
            f"  {self.name}: opening '{filename}' layer '{layer}' for streams in '{bounds}'")
        with fiona.open(filename, mode='r', layer=layer) as fid:
            profile = fid.profile
            bounds = watershed_workflow.warp.bounds(
                bounds, bounds_crs, watershed_workflow.crs.from_fiona(profile['crs']))
            reaches = [r for (i, r) in fid.items(bbox=bounds)]
            logging.info(f"  Found total of {len(reaches)} in bounds.")

        # filter not in network
        if 'NHDPlus' in self.name and in_network:
            logging.info("  Filtering reaches not in-network")
            reaches = [
                r for r in reaches
                if 'InNetwork' in r['properties'] and r['properties']['InNetwork'] == 1
            ]

        # associate catchment areas with the reaches if NHDPlus
        if 'Plus' in self.name and properties != None:
            reach_dict = dict((r['properties']['NHDPlusID'], r) for r in reaches)

            # validation of properties
            valid_props = list(self._nhdplus_vaa.keys()) + list(
                self._nhdplus_eromma.keys()) + ['catchment', ]
            for prop in properties:
                if prop not in valid_props:
                    raise ValueError(
                        f'Unrecognized NHDPlus property {prop}.  If you are sure this is valid, add the alias and variable name to the nhdplus tables in FileManagerNHDPlus.'
                    )

            # flags for which layers will be needed
            if include_catchments:
                layer = 'NHDPlusCatchment'
                logging.info(
                    f"  {self.name}: opening '{filename}' layer '{layer}' for catchments in '{bounds}'"
                )
                for r in reaches:
                    r['properties']['catchment'] = None
                with fiona.open(filename, mode='r', layer=layer) as fid:
                    for catchment in fid.values():
                        reach = reach_dict.get(catchment['properties']['NHDPlusID'])
                        if reach is not None:
                            reach['properties']['catchment'] = catchment

            if len(set(self._nhdplus_vaa.keys()).intersection(set(properties))) > 0:
                layer = 'NHDPlusFlowlineVAA'
                logging.info(
                    f"  {self.name}: opening '{filename}' layer '{layer}' for river network properties in '{bounds}'"
                )
                with fiona.open(filename, mode='r', layer=layer) as fid:
                    for flowline in fid.values():
                        reach = reach_dict.get(flowline['properties']['NHDPlusID'])
                        if reach is not None:
                            for prop in properties:
                                if prop in list(self._nhdplus_vaa.keys()):
                                    prop_code = self._nhdplus_vaa[prop]
                                    reach['properties'][prop] = flowline['properties'][prop_code]

            if len(set(self._nhdplus_eromma.keys()).intersection(set(properties))) > 0:
                layer = 'NHDPlusEROMMA'
                logging.info(
                    f"  {self.name}: opening '{filename}' layer '{layer}' for river network properties in '{bounds}'"
                )
                with fiona.open(filename, mode='r', layer=layer) as fid:
                    for flowline in fid.values():
                        reach = reach_dict.get(flowline['properties']['NHDPlusID'])
                        if reach is not None:
                            for prop in properties:
                                if prop in list(self._nhdplus_eromma.keys()):
                                    prop_code = self._nhdplus_eromma[prop]
                                    reach['properties'][prop] = flowline['properties'][prop_code]

        return profile, reaches

    def get_waterbodies(self, huc, bounds=None, bounds_crs=None, force_download=False):
        """Get all water bodies, e.g. lakes, reservoirs, etc, within a given HUC and/or coordinate bounds.

        Parameters
        ----------
        huc : int or str
          The USGS Hydrologic Unit Code
        bounds : [xmin, ymin, xmax, ymax], optional
          Coordinate bounds to filter reaches returned.  If this is provided,
          bounds_crs must also be provided.
        bounds_crs : CRS, optional
          CRS of the above bounds.
        force_download : bool Download
          or re-download the file if true.

        Returns
        -------
        profile : dict
          The fiona shapefile profile (see Fiona documentation).
        reaches : list(dict)
          List of fiona shape objects representing the stream reaches.

        Note this finds and downloads files as needed.

        """

        if 'NHDPlus' not in self.name:
            raise RuntimeError(f'{self.name}: does not provide water body data.')

        huc = source_utils.huc_str(huc)
        hint_level = len(huc)

        # try to get bounds if not provided
        if bounds is None:
            # can we infer a bounds by getting the HUC?
            profile, hu = self.get_huc(huc)
            bounds = watershed_workflow.utils.bounds(hu)
            bounds_crs = watershed_workflow.crs.from_fiona(profile['crs'])

        # error checking on the levels, require file_level <= huc_level <= lowest_level
        if hint_level < self.file_level:
            raise ValueError(
                f"{self.name}: files are organized at HUC level {self.file_level}, so cannot ask for a larger HUC level."
            )

        # download the file
        filename = self._download(huc[0:self.file_level], force=force_download)
        logging.info('  Using Hydrography file "{}"'.format(filename))

        # find and open the waterbody layer
        filename = self.name_manager.file_name(huc[0:self.file_level])
        layer = 'NHDWaterbody'
        logging.info(
            f"  {self.name}: opening '{filename}' layer '{layer}' for streams in '{bounds}'")
        with fiona.open(filename, mode='r', layer=layer) as fid:
            profile = fid.profile
            bounds = watershed_workflow.warp.bounds(
                bounds, bounds_crs, watershed_workflow.crs.from_fiona(profile['crs']))
            bodies = [r for (i, r) in fid.items(bbox=bounds)]
            logging.info(f"  Found total of {len(bodies)} in bounds.")

        return profile, bodies

    def _url(self, hucstr):
        """Use the USGS REST API to find the URL to download a file for a given huc.

        Parameters
        ----------
        hucstr : str
          The USGS Hydrologic Unit Code

        Returns
        -------
        url : str
          The URL to download a file containing shapes for the HUC.
        """
        rest_url = 'https://tnmaccess.nationalmap.gov/api/v1/products'
        hucstr = hucstr[0:self.file_level]

        def attempt(params):
            r = requests.get(rest_url, params=params, verify=False)
            logging.info(f'  REST URL: {r.url}')
            try:
                r.raise_for_status()
            except Exception as e:
                logging.error(e)
                return 1, e

            json = r.json()
            #logging.debug(json)

            matches = [(m, self._valid_url(i, m, hucstr)) for (i, m) in enumerate(json['items'])]
            logging.debug(f'     found {len(matches)} matches')
            matches_f = list(filter(lambda tup: tup[1], matches))
            logging.debug(f'     found {len(matches_f)} valid matches')
            if len(matches_f) == 0:
                logging.error('{}: no matches for HUC {} ({})'.format(self.name, hucstr,
                                                                      len(matches)))
                return 1, '{}: not able to find HUC {}'.format(self.name, hucstr)
            if len(matches_f) > 1:
                logging.error('{}: too many matches for HUC {} ({})'.format(
                    self.name, hucstr, len(matches)))
                for (m, url) in matches_f:
                    logging.error(' {}\n   {}'.format(m['title'], url))
                return 1, '{}: too many matches for HUC {}'.format(self.name, hucstr)
            return 0, matches_f[0][1]

        # cheaper if it works, may not work in alaska?
        a1 = attempt({
            'datasets': self.name,
            'polyType': 'huc{}'.format(self.file_level),
            'polyCode': hucstr
        })
        if not a1[0]:
            logging.debug('  REST query with polyCode... SUCCESS')
            logging.debug(f'  REST query: {a1[1]}')
            return a1[1]
        else:
            logging.debug('  REST query with polyCode... FAIL')

        # may find via huc4?
        if (self.file_level >= 4):
            a2 = attempt({ 'datasets': self.name, 'polyType': 'huc4', 'polyCode': hucstr[0:4] })
            if not a2[0]:
                logging.debug('  REST query with polyCode... SUCCESS')
                logging.debug(f'  REST query: {a2[1]}')
                return a2[1]
            else:
                logging.debug('  REST query with polyCode... FAIL')

        # # works more univerasally but is a BIG lookup, then filter locally
        # a2 = attempt({'datasets':self.name})
        # if not a2[0]:
        #     logging.debug('  REST query without polyCODE... SUCCESS')
        #     logging.debug(f'  REST query: {a2[1]}')
        #     return a2[1]

        # logging.debug('  REST query without polyCODE... FAIL')
        raise ValueError('{}: cannot find HUC {}'.format(self.name, hucstr))

    def _valid_url(self, i, match, huc, gdb_only=True):
        """Helper function that returns the URL if valid, or False if not."""
        ok = True
        logging.info(f'Checking match for {huc}? {match["downloadURL"]}')

        if ok:
            ok = "format" in match
            logging.info(f'format in match? {ok}')
        if ok:
            ok = "urls" in match
            logging.info(f'urls in match? {ok}')
        if ok:
            # search for a GDB url
            try:
                url_type = next(ut for ut in match['urls'] if 'GDB' in ut or 'GeoDatabase' in ut)
            except StopIteration:
                logging.info(f'Cannot find GDB url: {list(match["urls"].keys())}')
                return False
            else:
                url = match['urls'][url_type]

        # we have a url, is it actually this huc?
        url_split = url.split('/')
        logging.info(f'YAY: {url}')
        logging.info(f'Checking match {i}: {url_split[-2]}, {url_split[-1]}')

        # check the title contains (NHD) if NHD, or (NHDPlus HR) if NHDPlus
        if ok:
            ok = "title" in match
            logging.debug(f'title in match? {ok}')
        if ok:
            for abbrev in ['NHD', 'NHDPlus', 'NHDPlus HR', 'WBD']:
                my_abbrev = f'({abbrev})'.lower()
                if my_abbrev in self.name.lower():
                    break
            ok = my_abbrev in match["title"].lower()
            logging.debug(f'name in title? {ok}')
        if not ok:
            return False

        if huc not in url_split[-1]:
            # not the right HUC
            return False
        if gdb_only and 'GDB' != url_split[-2]:
            # not a GDB
            return False
        return url

    def _download(self, hucstr, force=False):
        """Find and download data from a given HUC.

        Parameters
        ----------
        hucstr : str
          The USGS Hydrologic Unit Code
        force : bool, optional
          If true, re-download even if a file already exists.

        Returns
        -------
        filename : str
          The path to the resulting downloaded dataset.
        """
        # check directory structure
        os.makedirs(self.name_manager.data_dir(), exist_ok=True)
        os.makedirs(self.name_manager.folder_name(hucstr), exist_ok=True)

        work_folder = self.name_manager.raw_folder_name(hucstr)
        os.makedirs(work_folder, exist_ok=True)

        filename = self.name_manager.file_name(hucstr)
        if not os.path.exists(filename) or force:
            url = self._url(hucstr)

            downloadfile = os.path.join(work_folder, url.split("/")[-1])
            if not os.path.exists(downloadfile) or force:
                logging.info("Attempting to download source for target '%s'" % filename)
                source_utils.download(url, downloadfile, force)

            source_utils.unzip(downloadfile, work_folder)

            # hope we can find it?
            gdb_files = [f for f in os.listdir(work_folder) if f.endswith('.gdb')]
            assert (len(gdb_files) == 1)

            if os.path.exists(filename):
                shutil.rmtree(filename)
            source_utils.move(os.path.join(work_folder, gdb_files[0]), filename)

        if not os.path.exists(filename):
            raise RuntimeError("Cannot find or download file for source target '%s'" % filename)
        return filename

