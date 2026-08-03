.. _utilities:

Utilities
=========

The :mod:`watershed_workflow.utils` subpackage provides coordinate
transforms, raster/vector data manipulation, and package configuration.
Geometry helpers and the generic tree data structure are kept in their
own sub-namespaces rather than flattened here.

.. _package configuration:

Package Configuration
---------------------

.. autofunction:: watershed_workflow.utils.config.setDataDirectory
.. autodata:: watershed_workflow.utils.config.rcParams

.. automodule:: watershed_workflow.utils
   :members:
   :imported-members:
   :exclude-members: setDataDirectory, rcParams

Geometry
--------

.. automodule:: watershed_workflow.utils.geometry
   :members:

Tree
----

.. automodule:: watershed_workflow.utils.tinytree
   :members:
