Data Structures and Shape Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Several custom data structures are used in the manipulation of
geometry to form a consistent mesh.  Two are the most important: the
SplitHUC object defines a set of polygons that partition the domain
into sub-catchments of the full domain (e.g. HUC 12s in a a HUC8
domain, or differential contributing areas to a series of gages).  The
RiverTree object defines a tree data structure defined by a
child-parent relationship where children of a reach are all reaches
that flow into that reach.  

SplitHUCs
+++++++++
.. automodule:: watershed_workflow.split_hucs
   :members:

RiverTree
+++++++++
.. automodule:: watershed_workflow.river_tree
   :members:

Hydrography
+++++++++++
.. automodule:: watershed_workflow.hydrography                
   :members:

