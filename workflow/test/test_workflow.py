import pytest

import numpy as np
import shapely.geometry
import workflow.utils
import workflow.hucs

from workflow.test.shapes import *


def test_full_workflow(two_boxes, rivers):
    # build the HUC object
    tb = workflow.hucs.HUCs(two_boxes)

    # bin and split the rivers
    #bins = workflow.hydrography.cut_and_bin(tb, rivers)
