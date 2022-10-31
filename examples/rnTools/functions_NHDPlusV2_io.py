from requests_html import HTMLSession
import fiona
import os
import watershed_workflow

## Component names for the NHDPlus V2 datasets
_componentnames = ["CatSeed", "FdrFac", "FdrNull", "FilledAreas", "Hydrodem", "NEDSnapshot" \
, "EROMExtension", "NHDPlusAttributes", "NHDPlusBurnComponents", "NHDPlusCatchment" \
, "NHDSnapshotFGDB", "NHDSnapshot", "VPUAttributeExtension", "VogelExtension", "WBDSnapshot"]



def get_NHDPlusV2_URLs_from_EPA_url(url, verify=True):

    with HTMLSession() as session:
        response = session.get(url, verify=verify)
        response.raise_for_status()
        status_code = response.status_code  # A status code of 200 means it was accepted
        print("Status code:" + str(status_code))
        html = response.html
        html.render()
        all_links = html.absolute_links

    return [ll for ll in list(all_links) if ".7z" in ll]
    
def get_NHDPlusV2_component_url(data_links, componentnames): 
    return [get_url_NHD_dataset(data_links, cc)[0] for cc  in componentnames]

def get_url_NHD_dataset(data_links, nhd_name):
    return [match for match in data_links if nhd_name in match]

def get_BoundaryUnit_Info(bounds, bounds_crs,BoundaryUnitFile):
    # bounds =  watershed bounds
    # bounds_crs = CRS for the watershed bounds 

    with fiona.open(BoundaryUnitFile) as fid:
        # Get the CRS for the Boundary Units
        BoundaryUnits_crs = watershed_workflow.crs.from_fiona(fid.profile['crs'])
        # Project the watershed boundary to the CRS for the Boundary Units
        bounds = watershed_workflow.warp.bounds(
            bounds, bounds_crs, BoundaryUnits_crs)
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
    return daID_vpu_rpu

def get_URLs_VPU(daID_vpu_rpu):
    url_base = 'https://www.epa.gov/waterdata/nhdplus-tennessee-data-vector-processing-unit-'
    return [url_base + tmp[1] for tmp in daID_vpu_rpu]