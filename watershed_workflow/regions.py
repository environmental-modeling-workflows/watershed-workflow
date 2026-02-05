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
                        volume : bool = False,
                        return_partitions : bool = False) -> Optional[List[List[int]]]:
    """Label m2 with region(s) for each polygon.

    Always adds a surface region; if volume also adds a volume region
    for extrusion in 3D.

    Parameters
    ----------
    m2 : mesh.Mesh2D
      The mesh to label.
    polygons : SplitHUCs | List[Tuple[shapely.Polygon, str]]
      The polygons covering each region. Either a SplitHUCs object or
      a list of tuples containing (polygon, name) pairs.
    volume : bool, optional
      If true, also add the volumetric region below the polygon that
      will be extruded in a 3D mesh eventually.
    return_partitions : bool, optional
      If true, return the list of cell indices for each polygon.
      Default is False.

    Returns
    -------
    partitions : List[List[int]] or None
      A list of length polygons, each entry of which is the list of
      cell indices in that polygon. Only returned if return_partitions is True.
    """
    if isinstance(polygons, SplitHUCs):
        polygons = list(zip(polygons.df['geometry'], polygons.df[names.NAME]))
    elif isinstance(polygons, list):
        # polygons should already be a list of (polygon, name) tuples
        pass
    else:
        raise ValueError("polygons must be either SplitHUCs or List[Tuple[shapely.Polygon, str]]")
    
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

    if return_partitions:
        return partitions
    return None


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
    partitions = addPolygonalRegions(m2, hucs, volume=True, return_partitions=True)

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
            logging.info(f'Exterior outlet point (from attribute): {exterior_outlet_point}')
        else:
            try:
                boundary = hucs.exterior.exterior
            except AttributeError:
                boundary = shapely.ops.unary_union(hucs).exterior
            exterior_outlet_point = next(outlet for outlet in hucs.df[names.OUTLET]
                                         if outlet.buffer(500).intersects(boundary))
            logging.info(f'Exterior outlet point (from search): {exterior_outlet_point}')

        outlet_faces = [e for e in m2.boundary_edges if isInsideBall(exterior_outlet_point, e)]
        edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
        if len(edges) == 0:
            logging.warn('...unable to find any outlet edges')
        else:
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

def addDischargeRegions(m2, discharge_points, labels=None, include_cells=True, buffer_width=1):
    """Add labeled sets for three faces for each discharge point in the river corridor.
    The three faces include downstream shorter edge of the quad and two edges connecting 
    two downstream vertices of the quad and non-quad vertice on the bank triangle. 
    Corresponding upstreams cells (a quad and two triangles) are also added if include_cells is True,
    which should be use with 
    <Parameter name="direction normalized flux relative to region" type="string" value="discharge_cell_region_name" />
    in the ATS observation parameter list in the input file 

    Parameters
    ----------
    m2 : watershed_workflow.mesh.Mesh2D
        The 2D mesh containing river corridor elements
    discharge_points : list of (x,y) coordinates
        List of discharge point locations to add regions for
    labels : list of str, optional
        Custom labels for each discharge point. If not provided, defaults to
        'discharge point 0', 'discharge point 1', etc.
    include_cells : bool, optional
        If True, add a labeled set for the cells just upstream of discharge faces. Default is True.
    buffer_width : float, optional
        Buffer width to identify quad elements containing the discharge point. Default is 1.

    Notes
    -----
    For each discharge point, this creates a labeled set containing three edges:
    1. The downstream edge of the quad element containing the point
    2. Edge connecting downstream right vertex to right bank
    3. Edge connecting downstream left vertex to left bank
    and labeled set containing three cells:
    1. the quad element containing the point
    2. the two triangles sharing edges with the quad
    """
    if labels is None:
        labels = ['discharge region ' + str(i) for i in range(len(discharge_points))]
    
    # Process each discharge point
    for discharge_point, label in zip(discharge_points, labels):
        # Convert coordinates to shapely Point
        if isinstance(discharge_point, tuple):
          discharge_point = shapely.geometry.Point(discharge_point)
        
        # Find the three edges around this discharge point
        if include_cells:
          discharge_edges, discharge_cells = findDischargeEdgesCells(m2, discharge_point, include_cells, buffer_width)
        else:
          discharge_edges = findDischargeEdgesCells(m2, discharge_point, include_cells, buffer_width)
    
        if discharge_edges:  # Only create labeled set if edges were found
            ls2 = watershed_workflow.mesh.LabeledSet(label,
                                                     m2.getNextAvailableLabeledSetID(), 'FACE', discharge_edges)
            ls2.to_extrude = True
            m2.labeled_sets.append(ls2)
            
            if include_cells and len(discharge_cells) > 0:
              ls2 = watershed_workflow.mesh.LabeledSet(label + ' cells',
                                                     m2.getNextAvailableLabeledSetID(), 'CELL', discharge_cells)
              m2.labeled_sets.append(ls2)
        else:
            print(f"No discharge edges found for point {discharge_point}")
            
            
def findDischargeEdgesCells(m2, discharge_point, include_cells=True, buffer_width=1):
    """Find the edges around a discharge point in a river corridor mesh.

    Parameters
    ----------
    m2 : watershed_workflow.mesh.Mesh2D
        The 2D mesh containing river corridor elements
    discharge_point : shapely.geometry.Point
        The discharge point location
    include_cells : bool, optional
        If True, include cells in the labeled set. Default is True.
    buffer_width : float, optional
        Buffer width to identify quad elements containing the discharge point. Default is 1.

    Returns
    -------
    list of tuples
        List of (vertex1, vertex2) pairs defining edges.
        if include_cells is True, also returns the cells just upstream of the edges.
    """
    # find the quad element that contains the discharge point
    discharge_quad = None
    discharge_point_buffer = discharge_point.buffer(buffer_width)
    for c, conn in enumerate(m2.conn):
        if len(conn) > 3:
          poly = shapely.geometry.Polygon(m2.coords[conn])
          if poly.intersects(discharge_point_buffer):
              discharge_quad = conn
              discharge_quad_id = c
              break   
    if discharge_quad is None:
        return []

    # get bank nodes and construct edges
    bank_node_ids = watershed_workflow.condition._findBankVerticesFromElem(m2, discharge_quad)  
    discharge_edges = [
        (discharge_quad[0], bank_node_ids[0]),  # edge to right bank
        (discharge_quad[-1], bank_node_ids[1]), # edge to left bank 
        (discharge_quad[-1], discharge_quad[0])  # downstream edge
    ]
    
    if include_cells:
        discharge_cells=[]
        # edge on the right as we look from the downstream direction
        edge_r = list(m2.cell_edges(discharge_quad))[0]
        cell_ids = m2.edges_to_cells[edge_r]
        bank_cell_id = next(cell_id for cell_id, conn in zip(cell_ids, [m2.conn[cell_id] for cell_id in cell_ids]) if len(conn) == 3)
        discharge_cells.append(bank_cell_id)
        
        # edge on the left as we look from the downstream direction
        edge_l = list(m2.cell_edges(discharge_quad))[-2]
        cell_ids = m2.edges_to_cells[edge_l]
        bank_cell_id = next(cell_id for cell_id, conn in zip(cell_ids, [m2.conn[cell_id] for cell_id in cell_ids]) if len(conn) == 3)
        discharge_cells.append(bank_cell_id)
        
        # quad element
        discharge_cells.append(discharge_quad_id)
        return discharge_edges, discharge_cells
      
    return discharge_edges
 