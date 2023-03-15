"""
This script hosts functions to add labelsets to define regions
"""


import numpy as np
import attr
import shapely


import watershed_workflow.mesh
import watershed_workflow.utils



def add_nlcd_labeled_sets(m2, nlcd_colors, nlcd_names):
    """Given a 2D mesh and an array of length m2.num_cells of indices, add labeled sets."""
    inds = np.unique(nlcd_colors)
    for ind in inds:
        ent_ids = list(np.where(nlcd_colors == ind)[0])
        ls = watershed_workflow.mesh.LabeledSet(nlcd_names[ind], int(ind), 'CELL', ent_ids)
        m2.labeled_sets.append(ls)


def _get_labels(polygons):
    labels = []
    for i, p in enumerate(polygons):
        label = f'watershed {i}'
        labels.append(label)
    return labels


def add_watershed_regions(m2, polygons, labels=None):
    """Add labeled sets to m2 for each polygon."""
    import shapely
    if labels is None:
        labels = _get_labels(polygons)
    else:
        assert (len(labels) == len(polygons))

    partitions = [list() for p in polygons]
    for c in range(m2.num_cells):
        cc = m2.compute_centroid(c)
        cc = shapely.geometry.Point(cc[0], cc[1])
        try:
            ip = next(i for (i, p) in enumerate(polygons) if p.contains(cc))
        except StopIteration:
            pass
        else:
            partitions[ip].append(c)

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



def add_watershed_regions_and_outlets(m2, hucs, outlets=None, outlet_width=None, labels=None, exterior_outlet= True):
    """Add four labeled sets to m2 for each polygon:

    - cells in the polygon, to be extruded
    - cells in the polygon, to be kept as faces upon extrusion
    - boundary of the polygon (edges)
    - outlet of the polygon (edges within outlet width of the outlet)

    """
    if isinstance(hucs, list):
        polygons = hucs
        if outlets is None:
            outlets = [None, ] * len(polygons)
    else:
        polygons = list(hucs.polygons())
        assert (outlets is None)
        outlets = hucs.polygon_outlets
    if labels is None:
        labels = _get_labels(polygons)
    else:
        assert (len(labels) == len(polygons))

    if outlet_width is None:
        outlet_width = 300

    # this adds the first two sets
    partitions = add_watershed_regions(m2, polygons, labels)

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

    for label, tris, outlet in zip(labels, partitions, outlets):
        subdomain_conn = [list(m2.conn[tri]) for tri in tris]
        subdomain_nodes = set([c for e in subdomain_conn for c in e])
        subdomain_coords = np.array([m2.coords[c] for c in subdomain_nodes])
        m2h = watershed_workflow.mesh.Mesh2D(subdomain_coords, subdomain_conn, check_handedness=False, validate=False)

        edges = [(int(e[0]), int(e[1])) for e in m2h.boundary_edges]
        ls = watershed_workflow.mesh.LabeledSet(label + ' boundary', m2.next_available_labeled_setid(), 'FACE', edges)
        ls.to_extrude = True  # this marker tells the extrusion routine
        # to not limit it to the surface
        m2.labeled_sets.append(ls)

        # every polygon now has an outlet -- find the boundary faces near that outlet
        if outlet is not None:
            outlet_faces = [e for e in m2h.boundary_edges if inside_ball(outlet, e)]
            edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
            ls = watershed_workflow.mesh.LabeledSet(label + ' outlet', m2.next_available_labeled_setid(), 'FACE', edges)
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
            exterior_outlet_point = next(outlet for outlet in outlets if outlet.buffer(500).intersects(boundary))

        outlet_faces = [e for e in m2.boundary_edges if inside_ball(exterior_outlet_point, e)]
        edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
        ls2 = watershed_workflow.mesh.LabeledSet('surface domain outlet', m2.next_available_labeled_setid(), 'FACE', edges)
        ls2.to_extrude = True
        m2.labeled_sets.append(ls2)


    
def add_river_corridor_regions(m2, rivers, labels=None):
    """Add labeled sets to m2 for each river corridor.
     
    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
        2D mesh elevated on DEMs
    rivers: list(watershed_workflow.river_tree.RiverTree)
        list of rivers used to  create river corridors 
    labels: list(str), optional
    """
    rivers_mls = [shapely.geometry.MultiLineString(list(river)) for river in rivers]
    if labels is None:
        labels = []
        for i, p in enumerate(rivers_mls):
            label = f'river_corridor {i}'
            labels.append(label)
    else:
        assert (len(labels) == len(rivers))

    partitions = [list() for p in rivers]
    for c, conn in enumerate(m2.conn):
        cell_poly = shapely.geometry.Polygon(m2.coords[conn])
        try:
            ip = next(i for (i, p) in enumerate(rivers_mls) if p.intersects(cell_poly.buffer(-1))) 
            # this shrinking is done to avoid non-river-corridor triangles at the tip of the headwater 
            # reach that might be touching the reach end from being included in the region
        except StopIteration:
            pass
        else:
            partitions[ip].append(c)

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            setid2 = m2.next_available_labeled_setid()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)


def add_poly_regions(m2, polys, labels=None):
    """Add labeled sets to m2 for list of areas (for example, reservoir, detention pond, etc.) defined by polygons.
    
    Parameters:
    -----------
    m2: watershed_workflow.mesh.Mesh2D object
        2D mesh elevated on DEMs
    polys: list(shapely.geometry.Polygon)
    labels: list(str), optional
    """
    if labels is None:
        labels = []
        for i, p in enumerate(polys):
            label = f'area {i}'
            labels.append(label)
    else:
        assert (len(labels) == len(polys))

    partitions = [list() for p in polys]
    for c in range(m2.num_cells):
        cc = m2.compute_centroid(c)
        cc = shapely.geometry.Point(cc[0], cc[1])
        try:
            ip = next(i for (i, p) in enumerate(polys) if p.contains(cc))
        except StopIteration:
            pass
        else:
            partitions[ip].append(c)

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            setid2 = m2.next_available_labeled_setid()
            ls2 = watershed_workflow.mesh.LabeledSet(label + ' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)
