"""
This namespace hosts functions to add labelsets to define regions
"""

import numpy as np
import shapely

import watershed_workflow.mesh
import watershed_workflow.utils


def add_nlcd_labeled_sets(m2, nlcd_colors, nlcd_names=None):
    """Add labeled sets to a mesh -- one per unique color.

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to label.
    nlcd_colors : np.array
      1D array of len m2.num_cells of cell colors.
    nlcd_names : dict, optional
      Dictionary mapping colors to color name.

    """
    inds = np.unique(nlcd_colors)

    if nlcd_names is None:
        nlcd_names = dict((i, str(i)) for i in inds)

    for ind in inds:
        ent_ids = list(np.where(nlcd_colors == ind)[0])
        ls = watershed_workflow.mesh.LabeledSet(nlcd_names[ind], int(ind), 'CELL', ent_ids)
        m2.labeled_sets.append(ls)


def _get_labels(polygons, kind='polygon'):
    if isinstance(polygons, watershed_workflow.split_hucs.SplitHUCs):
        labels = [prop['ID'] for prop in polygons.properties]
    else:
        labels = [f'{kind} {i}' for i in range(len(polygons))]
    return labels


def add_polygonal_regions(m2, polygons, labels=None, kind='watershed', volume=False):
    """Label m2 with region(s) for each polygon.

    Always adds a surface region; if volume also adds a volume region
    for extrusion in 3D.

    Parameters
    ----------
    m2 : mesh.Mesh2D
      The mesh to label.
    polygons : iterable[shapely.Polygon]
      The polygons covering each watershed.
    labels : iterable[str], optional
      Names of the polygons (often HUC strings)
    kind : str, optional
      The kind of polygon to add -- this is only used to generate
      labels.  Default is 'watershed'.
    volume : bool, optional
      If true, also add the volumetric region below the polygon that
      will be extruded in a 3D mesh eventually.

    Returns
    -------
    partitions : list[list[int]]
      A list of length polygons, each entry of which is the list of
      cell indices in that polygon.
    """
    if labels is None:
        labels = _get_labels(polygons)
    assert (len(labels) == len(polygons))

    partitions = [list() for p in polygons]
    for c in range(m2.num_cells):
        cc = m2.centroids[c]
        cc = shapely.geometry.Point(cc[0], cc[1])

        for i, p in enumerate(polygons):
            if p.contains(cc):
                partitions[i].append(c)

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            # add a region, denoting this one as "to extrude".  This
            # will become the volume region
            setid = m2.next_available_labeled_setid()
            ls = watershed_workflow.mesh.LabeledSet(label, int(setid), 'CELL', part)
            m2.labeled_sets.append(ls)
            ls.to_extrude = True

            # add a second region, denoting this one as the top surface of faces
            setid2 = m2.next_available_labeled_setid()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)

    return partitions


def add_watershed_regions(m2, polygons, labels=None):
    """Deprecated -- kept for backward compatibility."""
    return add_polygonal_regions(m2, polygons, labels, volume=True)


def add_watershed_regions_and_outlets(m2,
                                      hucs,
                                      outlets=None,
                                      outlet_width=300,
                                      labels=None,
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
    hucs : iterable[shapely.Polygon] or SplitHUCs object
      Watershed polygons.
    outlets : iterable[shapely.Point], optional
      If provided, the outlet points.  If SplitHUCs are provided,
      outlets are in that object and this must be None.
    outlet_width : float, optional
      How wide should the outlet region be?  Note this should include
      not just the river width, but more realistically the best
      resolved floodplain, or 1-2 face-widths, whichever is bigger.
    labels : iterable[str], optional
      Name of the polygons.  If SplitHUCs are provided, HUC names are
      used as the default.
    exterior_outlet : bool, optional
      If true, find the outlet point that intersects the boundary and
      include regions around that outlet as well.

    """
    if isinstance(hucs, list):
        polygons = hucs
    else:
        polygons = list(hucs.polygons())
        if outlets is None and hasattr(hucs, 'polygon_outlets'):
            outlets = hucs.polygon_outlets

    if labels is None:
        labels = _get_labels(hucs)

    if outlets is None:
        outlets = [None, ] * len(polygons)

    assert (len(labels) == len(polygons))

    # this adds the first two sets
    partitions = add_polygonal_regions(m2, polygons, labels, volume=True)

    # find a list of faces on the boundary of these sets of triangles
    #
    # UGLY HACK -- create a Mesh2D object with nan coordinates and
    # edges that are unordered nodes.  This could never be used for
    # anything geometric, but may allow us to exploit the existing
    # topologic routines in Mesh2D
    def inside_ball(outlet, edge):
        n1 = m2.coords[edge[0]]
        n2 = m2.coords[edge[1]]
        c = (n1+n2) / 2.
        close = watershed_workflow.utils.close(outlet, tuple(c[0:2]), outlet_width)
        return close

    for label, partition, outlet in zip(labels, partitions, outlets):
        subdomain_conn = [list(m2.conn[cell]) for cell in partition]
        subdomain_nodes = set([c for e in subdomain_conn for c in e])
        subdomain_coords = np.array([m2.coords[c] for c in subdomain_nodes])
        m2h = watershed_workflow.mesh.Mesh2D(subdomain_coords,
                                             subdomain_conn,
                                             check_handedness=False,
                                             validate=False)

        edges = [(int(e[0]), int(e[1])) for e in m2h.boundary_edges]
        ls = watershed_workflow.mesh.LabeledSet(label + ' boundary',
                                                m2.next_available_labeled_setid(), 'FACE', edges)
        ls.to_extrude = True  # this marker tells the extrusion routine
        # to not limit it to the surface
        m2.labeled_sets.append(ls)

        # every polygon now has an outlet -- find the boundary faces near that outlet
        if outlet is not None:
            outlet_faces = [e for e in m2h.boundary_edges if inside_ball(outlet, e)]
            edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
            if (len(edges) == 0):
                warnings.warn(f'Outlet region found 0 faces for polygon {label}')
            ls = watershed_workflow.mesh.LabeledSet(label + ' outlet',
                                                    m2.next_available_labeled_setid(), 'FACE',
                                                    edges)
            ls.to_extrude = True  # this marker tells the extrusion routine
            # to not limit it to the surface
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
            exterior_outlet_point = next(outlet for outlet in outlets
                                         if outlet.buffer(500).intersects(boundary))

        outlet_faces = [e for e in m2.boundary_edges if inside_ball(exterior_outlet_point, e)]
        edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
        ls2 = watershed_workflow.mesh.LabeledSet('surface domain outlet',
                                                 m2.next_available_labeled_setid(), 'FACE', edges)
        ls2.to_extrude = True
        m2.labeled_sets.append(ls2)


def add_discharge_regions(m2, discharge_points, labels=None, include_cells=True, buffer_width=1):
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
          discharge_edges, discharge_cells = find_discharge_edges_cells(m2, discharge_point, include_cells, buffer_width)
        else:
          discharge_edges = find_discharge_edges_cells(m2, discharge_point, include_cells, buffer_width)
    
        if discharge_edges:  # Only create labeled set if edges were found
            ls2 = watershed_workflow.mesh.LabeledSet(label,
                                                     m2.next_available_labeled_setid(), 'FACE', discharge_edges)
            ls2.to_extrude = True
            m2.labeled_sets.append(ls2)
            
            if include_cells and len(discharge_cells) > 0:
              ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface',
                                                     m2.next_available_labeled_setid(), 'CELL', discharge_cells)
              m2.labeled_sets.append(ls2)
        else:
            print(f"No discharge edges found for point {discharge_point}")
            
            
def find_discharge_edges_cells(m2, discharge_point, include_cells=True, buffer_width=1):
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
    bank_node_ids = watershed_workflow.condition._bank_nodes_from_elem(discharge_quad, m2)  
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
 
 
def add_river_corridor_regions(m2, rivers, labels=None):
    """Add labeled sets to m2 for each river corridor.
     
    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
      2D mesh elevated on DEMs
    rivers: list(watershed_workflow.river_tree.RiverTree)
      List of rivers used to create the river corridors.
    labels: list(str), optional
      List of names, one per river.
    """
    if labels is None:
        labels = []
        for i, p in enumerate(rivers):
            label = f'river_corridor {i}'
            labels.append(label)
    else:
        assert (len(labels) == len(rivers))

    for label, river in zip(labels, rivers):
        gid_start = river.properties['gid_start']
        river_elements = list()
        for node in river.preOrder():
            river_elements.extend(range(gid_start, gid_start + len(node.elements)))
            gid_start += len(node.elements)

        if len(river_elements) > 0:
            setid2 = m2.next_available_labeled_setid()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL',
                                                     river_elements)
            m2.labeled_sets.append(ls2)


def add_regions_by_stream_order_rivers(m2, rivers, labels=None):
    """Add labeled sets to m2 for reaches of each stream order for each river.
     
    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
      2D mesh elevated on DEMs
    rivers: list(watershed_workflow.river_tree.RiverTree)
      List of rivers used to create the river corridors.
    labels: list(str), optional
      List of names, one per river.
    """
    if labels is None:
        labels = []
        for i, p in enumerate(rivers):
            label = f'river {i}'
            labels.append(label)
    else:
        assert (len(labels) == len(rivers))

    for label, river in zip(labels, rivers):
        add_regions_by_stream_order(m2, river, river_id=label)


def add_regions_by_stream_order(m2, river, river_id=0):
    """Add labeled sets to m2 for reaches of each stream order .
     
    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
      2D mesh elevated on DEMs
    rivers: list(watershed_workflow.river_tree.RiverTree)
      List of rivers used to create the river corridors.
    river_id: str/int, optional
      river identifier/name.
    """
    from collections import defaultdict

    gid_start = river.properties['gid_start']
    regions = defaultdict(list)
    for node in river.preOrder():
        order = node.properties['StreamOrder']
        regions[order].extend(range(gid_start, gid_start + len(node.elements)))
        gid_start += len(node.elements)

    labels = []
    for order in regions.keys():
        label = f'reaches of StreamOrder {order} in {river_id}'
        labels.append(label)

    partitions = [regions[order] for order in regions.keys()]

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            setid2 = m2.next_available_labeled_setid()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)


def add_region_by_reach_id(m2, river, reach_ids=None, labels=None):
    """Add labeled sets to m2 for reaches of each stream order .
     
    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
      2D mesh elevated on DEMs
    river: watershed_workflow.river_tree.RiverTree
      List of rivers used to create the river corridors.
    reaches: list(str)
      list of NHDID IDs to be labeled.
    """

    from collections import defaultdict

    if labels == None:
        labels = []
        for id in reach_ids:
            label = f'reach with id {id}'
            labels.append(label)

    gid_start = river.properties['gid_start']
    regions = defaultdict(list)
    for node in river.preOrder():
        id = node.properties['ID']
        if id in reach_ids:
            regions[id].extend(range(gid_start, gid_start + len(node.elements)))
        gid_start += len(node.elements)

    partitions = [regions[order] for order in regions.keys()]

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            setid2 = m2.next_available_labeled_setid()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)


def getNode(nhd_id, rivers):
    """return node given NHDID"""
    node = next(river.getNode(nhd_id) for river in rivers if river.getNode(nhd_id) != None)
    return node
