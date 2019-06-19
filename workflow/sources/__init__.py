"""Generic namespace for working with sources.
"""
from workflow.sources.manager_nhdplus import FileManagerNHDPlus
from workflow.sources.manager_ned import FileManagerNED
from workflow.sources.manager_shape import FileManagerShape

def get_default_sources():
    sources = dict()
    sources['HUC'] = FileManagerNHDPlus()
    sources['rivers'] = sources['HUC']
    sources['DEM'] = FileManagerNED()
    sources['land cover'] = None
    sources['soil thickness'] = None
    sources['soil type'] = None
    return sources
