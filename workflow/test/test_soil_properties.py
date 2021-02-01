"""Test van Genuchten from Rosetta"""
import numpy as np
import workflow.soil_properties

def test_vgm():
    # headers: sand %, silt %, clay %, bulk dens
    data = np.array([70,15,15,1.4])

    vgm = workflow.soil_properties.vgm_Rosetta(data, 3)
    print(vgm)


def test_vgm2():
    # headers: sand %, silt %, clay %, bulk dens
    data = np.array([[70,15,15,1.4],
                     [50,25,25,1.4]]).transpose()

    vgm = workflow.soil_properties.vgm_Rosetta(data, 3)
    ats = workflow.soil_properties.to_ATS(vgm)
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
    
    
