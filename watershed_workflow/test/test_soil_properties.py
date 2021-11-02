"""Test van Genuchten from Rosetta"""
import numpy as np
import watershed_workflow.soil_properties

def test_vgm():
    # headers: sand %, silt %, clay %, bulk dens
    data = np.array([70,15,15,1.4])

    vgm = watershed_workflow.soil_properties.vgm_Rosetta(data, 3)
    print(vgm)


def test_vgm2():
    # headers: sand %, silt %, clay %, bulk dens
    data = np.array([[70,15,15,1.4],
                     [50,25,25,1.4]]).transpose()

    vgm = watershed_workflow.soil_properties.vgm_Rosetta(data, 3)
    ats = watershed_workflow.soil_properties.to_ATS(vgm)
    print(ats.keys())
    assert(all(ats['residual saturation [-]'] < 1))
    assert(all(ats['residual saturation [-]'] >= 0))
    assert(all(ats['Rosetta porosity [-]'] < 1))
    assert(all(ats['Rosetta porosity [-]'] >= 0))
    assert(all(ats['Rosetta porosity [-]'] > ats['residual saturation [-]']))
    assert(all(ats['van Genuchten alpha [Pa^-1]'] > 0))
    assert(all(ats['van Genuchten alpha [Pa^-1]'] < 1.e-2))    
    assert(all(ats['van Genuchten n [-]'] > 1))
    assert(all(ats['van Genuchten n [-]'] < 12))
    assert(all(ats['Rosetta permeability [m^2]'] > 0))
    assert(all(ats['Rosetta permeability [m^2]'] < 1.e-10))
    
    

def test_cluster():
    arr_in = np.array([[1.01,1,1],
                         [1,2,2],
                         [2,2.01,2]])
    arr_gd = np.array([[1,1,1],
                       [1,0,0],
                       [0,0,0]])

    arr_in = np.expand_dims(arr_in, -1)
    codebook, arr_out, dists = watershed_workflow.soil_properties.cluster(arr_in, 2)
    print(arr_out)

    assert((arr_out[arr_gd == 0] == arr_out[-1,-1]).all())
    assert((arr_out[arr_gd == 1] == arr_out[0,0]).all())
    
