"""Test van Genuchten from Rosetta"""
import numpy as np
import pandas as pd
import watershed_workflow.properties.soil


def test_vgm():
    data = pd.DataFrame({
        'sand pct [%]': [70],
        'silt pct [%]': [15],
        'clay pct [%]': [15],
        'bulk density [g/cm^3]': [1.4],
    })

    vgm = watershed_workflow.properties.soil.computeVanGenuchtenModel_Rosetta(data)
    print(vgm)


def test_vgm2():
    data = pd.DataFrame({
        'sand pct [%]': [70, 50],
        'silt pct [%]': [15, 25],
        'clay pct [%]': [15, 25],
        'bulk density [g/cm^3]': [1.4, 1.4],
    })

    vgm = watershed_workflow.properties.soil.computeVanGenuchtenModel_Rosetta(data)
    ats = watershed_workflow.properties.soil.convertRosettaToATS(vgm)
    print(ats.keys())
    assert (all(ats['residual saturation [-]'] < 1))
    assert (all(ats['residual saturation [-]'] >= 0))
    assert (all(ats['Rosetta porosity [-]'] < 1))
    assert (all(ats['Rosetta porosity [-]'] >= 0))
    assert (all(ats['Rosetta porosity [-]'] > ats['residual saturation [-]']))
    assert (all(ats['van Genuchten alpha [Pa^-1]'] > 0))
    assert (all(ats['van Genuchten alpha [Pa^-1]'] < 1.e-2))
    assert (all(ats['van Genuchten n [-]'] > 1))
    assert (all(ats['van Genuchten n [-]'] < 12))
    assert (all(ats['Rosetta permeability [m^2]'] > 0))
    assert (all(ats['Rosetta permeability [m^2]'] < 1.e-10))


def test_cluster():
    arr_in = np.array([[1.01, 1, 1], [1, 2, 2], [2, 2.01, 2]])
    arr_gd = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])

    arr_in = np.expand_dims(arr_in, -1)
    codebook, arr_out, dists = watershed_workflow.properties.soil.cluster(arr_in, 2)
    print(arr_out)

    assert ((arr_out[arr_gd == 0] == arr_out[-1, -1]).all())
    assert ((arr_out[arr_gd == 1] == arr_out[0, 0]).all())
