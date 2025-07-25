import pytest

import os
import shapely
import numpy as np

import watershed_workflow.config
import watershed_workflow.utils
import watershed_workflow.sources.manager_glhymps
import watershed_workflow

from fixtures import coweeta

def test_glhymps(coweeta):
    # get imgs
    glhymps = watershed_workflow.sources.manager_glhymps.ManagerGLHYMPS()
    data = glhymps.getShapesByGeometry(coweeta.geometry[0], coweeta.crs)

    # check df
    assert len(data) == 1
    assert data.crs is not None
