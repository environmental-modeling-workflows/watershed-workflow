import pytest

import watershed_workflow.sources.manager_nrcs

from fixtures import coweeta


def test_nrcs2(coweeta):
    # get imgs
    nrcs = watershed_workflow.sources.manager_nrcs.ManagerNRCS()
    df = nrcs.getShapesByGeometry(coweeta.geometry[0], coweeta.crs, force_download=True)

    # check df
    mukeys = set(df['ID'])
    assert len(df) == len(mukeys) # one per unique key
    assert len(df) == 42
    assert df.crs is not None
