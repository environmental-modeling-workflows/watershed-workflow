# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Watershed Workflow'
copyright = '2019-202X, UT Battelle, Ethan Coon'
author = 'Ethan Coon'

# The short X.Y version
version = 'dev'
# The full version, including alpha/beta/rc tags
release = 'dev'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinxcontrib.jquery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = 'en'

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_book_theme'
html_theme = "pydata_sphinx_theme"
html_title = "Watershed Workflow"
#html_favicon = "_static/images/favicon.ico"

html_sidebars = {
    "**" : ["version",
            "version-switcher",
            "sidebar-nav-bs.html",
            "page-toc.html",
            ]
}

html_theme_options = {
    # "logo": {
    #     "alt_text": "Watershed Workflow documentation -- Home",
        # "image_light": "_static/images/logo_full.png",
        # "image_dark": "_static/images/logo_full.png", # todo -- make a dark logo!
    # },
    "secondary_sidebar_items": [],
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/environmental-modeling-workflows/watershed-workflow/master/docs/source/_static/versions.json",
        "version_match": 'v2.0',
    },
#    "navbar_start" : ["navbar-logo", ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_css_files = [
    'https://cdn.datatables.net/2.0.8/css/dataTables.dataTables.css',
    'styles/custom_theme.css',
]

html_js_files = [
    'https://cdn.datatables.net/2.0.8/js/dataTables.js',
    'main.js',
]



nb_execution_excludepatterns = ['*',]
#nb_execution_excludepatterns = ['IHMIP_units.ipynb', 'mesh_gen.ipynb']



