High-Level API
==============

The top-level :mod:`watershed_workflow` namespace provides the
high-level, "do what the user means" functions that most workflows are
built from, plus the :class:`~watershed_workflow.crs.CRS`,
:class:`~watershed_workflow.hydro.river.River`,
:class:`~watershed_workflow.hydro.watershed.Watershed`,
:class:`~watershed_workflow.mesh.mesh.Mesh2D`, and
:class:`~watershed_workflow.mesh.mesh.Mesh3D` types used throughout.

.. autofunction:: watershed_workflow.findHUC
.. autofunction:: watershed_workflow.reduceRivers
.. autofunction:: watershed_workflow.simplify
.. autofunction:: watershed_workflow.triangulate
.. autofunction:: watershed_workflow.tessalateRiverAligned
.. autofunction:: watershed_workflow.elevate
.. autofunction:: watershed_workflow.getDatasetOnMesh
.. autofunction:: watershed_workflow.getShapePropertiesOnMesh
.. autofunction:: watershed_workflow.makeMap

.. _crs:

Coordinate Reference Systems
----------------------------

.. automodule:: watershed_workflow.crs
   :members:
