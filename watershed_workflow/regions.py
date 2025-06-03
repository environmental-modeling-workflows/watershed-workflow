"""
This namespace hosts functions to add labelsets to define regions
"""
from typing import Optional, List, Dict, Tuple
import logging
import warnings

import numpy as np
import shapely

from watershed_workflow.split_hucs import SplitHUCs
from watershed_workflow.river_tree import River
from watershed_workflow.mesh import Mesh2D

import watershed_workflow.sources.standard_names as names
import watershed_workflow.mesh
import watershed_workflow.utils



def addSurfaceRegions(m2 : Mesh2D,
                      column : str = 'land_cover',
                      names : Optional[dict[int, str]] = None):
    """Add labeled sets to a mesh -- one per unique color.

    Parameters
    ----------
    m2 : mesh.Mesh2D
      The mesh to label.
    names : dict, optional
      Dictionary mapping colors to color name.

    """
    assert m2.cell_data is not None
    colors = m2.cell_data[column].astype(int)
    inds = np.unique(colors)

    if names is None:
        names = dict((i, str(i)) for i in inds)

    for ind in inds:
        ent_ids = list(np.where(colors == ind)[0])
        ls = watershed_workflow.mesh.LabeledSet(names[ind], int(ind), 'CELL', ent_ids)
        m2.labeled_sets.append(ls)


def addPolygonalRegions(m2 : Mesh2D,
                        polygons : SplitHUCs | List[Tuple[shapely.Polygon, str]],
                        volume : bool = False) -> List[List[int]]:
    """Label m2 with region(s) for each polygon.

    Always adds a surface region; if volume also adds a volume region
    for extrusion in 3D.

    Parameters
    ----------
    m2 : mesh.Mesh2D
      The mesh to label.
    polygons : SplitHUCs | List[shapely.Polygon]
      The polygons covering each watershed.
    volume : bool, optional
      If true, also add the volumetric region below the polygon that
      will be extruded in a 3D mesh eventually.

    Returns
    -------
    partitions : List[List[int]]
      A list of length polygons, each entry of which is the list of
      cell indices in that polygon.
    """
    if isinstance(polygons, SplitHUCs):
        polygons = list(zip(polygons.df['geometry'], polygons.df[names.NAME]))
    logging.info(f"Adding regions for {len(polygons)} polygons")

    partitions : List[List[int]] = [list() for p in polygons]
    for c in range(m2.num_cells):
        cc = m2.centroids[c]
        cc = shapely.geometry.Point(cc[0], cc[1])

        done = False
        for i, (p,l) in enumerate(polygons):
            if p.contains(cc):
                done = True
                partitions[i].append(c)
                break

    for (p,label), part in zip(polygons, partitions):
        if len(part) > 0:
            # add a region, denoting this one as "to extrude".  This
            # will become the volume region
            setid = m2.getNextAvailableLabeledSetID()
            ls = watershed_workflow.mesh.LabeledSet(label, int(setid), 'CELL', part)
            m2.labeled_sets.append(ls)
            ls.to_extrude = True

            # add a second region, denoting this one as the top surface of faces
            setid2 = m2.getNextAvailableLabeledSetID()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)

    return partitions


def addWatershedAndOutletRegions(m2 : Mesh2D,
                                  hucs : SplitHUCs,
                                  outlet_width : float = 300,
                                  exterior_outlet=True):
    """Add four labeled sets to m2 for each polygon:

    - cells in the polygon, to be extruded
    - cells in the polygon, to be kept as faces upon extrusion
    - boundary of the polygon (edges, kept as faces)
    - outlet of the polygon (edges, kept as faces, within outlet width
      of the outlet)

    Parameters
    ----------
    m2 : mesh.Mesh2D
      The mesh to label.
    hucs : iterable[shapely.Polygon] or SplitHUCs
      Watershed polygons.
    outlet_width : float, optional
      How wide should the outlet region be?  Note this should include
      not just the river width, but more realistically the best
      resolved floodplain, or 1-2 face-widths, whichever is bigger.
    exterior_outlet : bool, optional
      If true, find the outlet point that intersects the boundary and
      include regions around that outlet as well.

    """
    # this adds sets for all m2 cells in each polygon of hucs
    partitions = addPolygonalRegions(m2, hucs, volume=True)

    # add a set for the boundary of each polygon and the outlet
    #
    def isInsideBall(outlet, edge):
        """Helper function -- is edge centroid within outlet_width of outlet?"""
        n1 = m2.coords[edge[0]]
        n2 = m2.coords[edge[1]]
        c = (n1+n2) / 2.
        close = watershed_workflow.utils.isClose(outlet, tuple(c[0:2]), outlet_width)
        return close


    # find a list of faces on the boundary of the partitioned sets created above
    for label, partition, outlet in zip(hucs.df[names.NAME],
                                        partitions,
                                        hucs.df[names.OUTLET]):

        # create a subdomain mesh for the partition.  This is not a
        # valid mesh, but it is sufficient to get the boundary edges
        subdomain_conn = [list(m2.conn[cell]) for cell in partition]
        subdomain_nodes = set([c for e in subdomain_conn for c in e])
        subdomain_coords = np.array([m2.coords[c] for c in subdomain_nodes])
        m2h = Mesh2D(subdomain_coords, subdomain_conn, check_handedness=False, validate=False)

        partition_boundary_edges = [(int(e[0]), int(e[1])) for e in m2h.boundary_edges]

        # create the partition boundary region of faces
        ls = watershed_workflow.mesh.LabeledSet(label + ' boundary',
                                                m2.getNextAvailableLabeledSetID(),
                                                'FACE', partition_boundary_edges)
        ls.to_extrude = True
        m2.labeled_sets.append(ls)

        # every polygon now has an outlet -- find the boundary faces near that outlet
        if outlet is not None:
            outlet_faces = [e for e in m2h.boundary_edges if isInsideBall(outlet, e)]
            edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
            if (len(edges) == 0):
                warnings.warn(f"Outlet region found 0 faces for polygon {label} near outlet "
                              f"at {outlet}.  Probably bad outlet data?")
            ls = watershed_workflow.mesh.LabeledSet(label + ' outlet',
                                                    m2.getNextAvailableLabeledSetID(),
                                                    'FACE', edges)
            ls.to_extrude = True
            m2.labeled_sets.append(ls)

    # also write one for the full domain
    if exterior_outlet:
        if hasattr(hucs, "exterior_outlet"):
            exterior_outlet_point = hucs.exterior_outlet
        else:
            try:
                boundary = hucs.exterior()
            except AttributeError:
                boundary = shapely.ops.unary_union(hucs)
            exterior_outlet_point = next(outlet for outlet in hucs.df[names.OUTLET]
                                         if outlet.buffer(500).intersects(boundary))

        outlet_faces = [e for e in m2.boundary_edges if isInsideBall(exterior_outlet_point, e)]
        edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
        ls2 = watershed_workflow.mesh.LabeledSet('surface domain outlet',
                                                 m2.getNextAvailableLabeledSetID(), 'FACE', edges)
        ls2.to_extrude = True
        m2.labeled_sets.append(ls2)


def addRiverCorridorRegions(m2 : Mesh2D,
                            rivers : List[River],
                            labels : Optional[List[str]] = None):
    """Add labeled sets to m2 for each river corridor.
     
    Parameters:
    -----------
    m2: Mesh2D
      2D mesh elevated on DEMs
    rivers: list(watershed_workflow.river_tree.RiverTree)
      List of rivers used to create the river corridors.
    labels: list(str), optional
      List of names, one per river.
    """
    if labels is None:
        labels = [f'river corridor {i}' for (i,r) in enumerate(rivers)]
    else:
        assert (len(labels) == len(rivers))

    for label, river in zip(labels, rivers):
        river_elements = [i for r in river.preOrder() for i in range(r[names.ELEMS_GID_START], r[names.ELEMS_GID_START] + len(r[names.ELEMS]))]

        if len(river_elements) > 0:
            setid2 = m2.getNextAvailableLabeledSetID()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL',
                                                     river_elements)
            m2.labeled_sets.append(ls2)


def addStreamOrderRegions(m2 : Mesh2D,
                          rivers : List[River]) -> None:
    """Add labeled sets to m2 for reaches of each stream order .
     
    Parameters:
    -----------
    m2: Mesh2D
      2D mesh elevated on DEMs
    rivers : List[watershed_workflow.river_tree.River]
      River used to create the river corridors.
    """
    from collections import defaultdict

    regions : Dict[int, List[int]] = defaultdict(list)
    for river in rivers:
        for reach in river.preOrder():
            order = reach[names.ORDER]
            regions[order].extend(range(reach[names.ELEMS_GID_START], reach[names.ELEMS_GID_START] + len(reach[names.ELEMS])))

    labels = [f'stream order {order}' for order in regions.keys()]
    partitions = [regions[order] for order in regions.keys()]

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            setid2 = m2.getNextAvailableLabeledSetID()
            ls2 = watershed_workflow.mesh.LabeledSet(label, setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)


def addReachIDRegions(m2 : Mesh2D,
                     river : River):
    """Add labeled sets to m2 for reaches of each stream order .
     
    Parameters:
    -----------
    m2: Mesh2D
      2D mesh elevated on DEMs
    river: watershed_workflow.river_tree.RiverTree
      List of rivers used to create the river corridors.
    reaches: list(str)
      list of NHDID IDs to be labeled.
    """

    from collections import defaultdict

    regions : Dict[str, List[int]] = defaultdict(list)
    for reach in river.preOrder():
        regions[f'reach {reach[names.ID]}'] = list(range(reach[names.ELEMS_GID_START], reach[names.ELEMS_GID_START] + len(reach[names.ELEMS])))

    for label, part in regions.items():
        if len(part) > 0:
            setid2 = m2.getNextAvailableLabeledSetID()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)


