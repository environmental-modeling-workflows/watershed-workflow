"""Tools for turning data into meshes, then writing them to file.

Works with and assumes all polyhedra cells (and polygon faces).

Requires building a reasonably recent version of Exodus to get
the associated exodus.py wrappers in python3.

Note that this is typically done in your standard ATS installation,
assuming you have built your Amanzi TPLs with shared libraries (the
default through bootstrap).

In that case, simply ensure that ${AMANZI_TPLS_DIR}/SEACAS/lib is in
your PYTHONPATH.

"""

import os
import numpy as np
import collections
import logging
import attr
import scipy.optimize
import shapely
import warnings
import functools

import watershed_workflow.vtk_io
import watershed_workflow.utils
import watershed_workflow.plot
import watershed_workflow.colors

try:
    import exodus
except Exception:
    try:
        import exodus3 as exodus
    except Exception:
        exodus = None


#
# A caching decorator, like functools.cache, but works in this case?
#
# Note, this can only work with __dict__ classes, not slotted classes.
#
def cache(func):
    @functools.wraps(func)
    def cache_func(self):
        cname = '_'+func.__name__
        if not hasattr(self, cname):
            self.__setattr__(cname, func(self))
        return self.__getattribute__(cname)
    return cache_func
    
    
@attr.define
class SideSet:
    """A collection of faces in cells."""
    name : str
    setid : int
    cell_list : list[int]
    side_list : list[int]

    def validate(self, cell_faces):
        ncells = len(cell_faces)
        for c,s in zip(self.cell_list, self.side_list):
            assert(0 <= c < ncells)
            assert(0 <= s < len(cell_faces[c]))


@attr.define
class LabeledSet:
    """A generic collection of entities."""
    name : str
    setid : int
    entity : str
    ent_ids : list[int]
    to_extrude : bool = False

    def validate(self, size, is_tuple=False):
        if is_tuple:
            # edges
            self.ent_ids = [(int(e[0]), int(e[1])) for e in self.ent_ids]
            for e in self.ent_ids:
                assert(-1 < e[0] < size)
                assert(-1 < e[1] < size)
        else:
            self.ent_ids = [int(i) for i in self.ent_ids]
            for i in self.ent_ids:
                assert(-1 < i < size)


@attr.define
class _ExtrusionHelper:
    """Helper class for extruding 2D --> 3D"""
    ncells_tall : int
    ncells_2D : list[int]

    def col_to_id(self, column, z_cell):
        """Maps 2D cell ID and index in the vertical to a 3D cell ID"""
        return z_cell + column * self.ncells_tall

    def node_to_id(self, node, z_node):
        """Maps 2D node ID and index in the vertical to a 3D node ID"""
        return z_node + node * (self.ncells_tall+1)

    def edge_to_id(self, edge, z_cell):
        """Maps 2D edge hash and index in the vertical to a 3D face ID of a vertical face"""
        return (self.ncells_tall + 1) * self.ncells_2D + z_cell + edge * self.ncells_tall


@attr.define(slots=False)
class Mesh2D:
    """A 2D mesh class.

    Parameters
    ----------
    coords : np.ndarray(NCOORDS,NDIMS)
        Array of coordinates of the 2D mesh.
    conn : list(lists)
        List of lists of indices into coords that form the cells.
    labeled_sets : list(LabeledSet), optional
        List of labeled sets to add to the mesh.
    crs : CRS
        Keep this as a property for future reference.
    eps : float, optional=0.01
        A small measure of length between coords.
    check_handedness : bool, optional=True
        If true, ensure all cells are oriented so that right-hand rule
        ordering of the nodes points up.
    validate : bool, optional=False
        If true, validate coordinates and connections
        post-construction.

    Note: (coords, conn) may be output provided by a
    watershed_workflow.triangulation.triangulate() call.

    """
    coords = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    _conn = attr.ib()
    labeled_sets = attr.ib(factory=list)
    crs = attr.ib(default=None)
    eps = attr.ib(default=0.001)
    _check_handedness = attr.ib(default=True)
    _validate = attr.ib(default=False)

    def __attrs_post_init__(self):
        if self._check_handedness:
            self.check_handedness()
        if self._validate:
            self.validate()
        del self._validate

    @property
    def conn(self):
        """Note that conn is immutable because changing this breaks all the
        other properties, which may be cached.  To change topology, one must construct a new mesh!"""        
        return self._conn

    @property
    def dim(self):
        return self.coords.shape[1]

    @property
    def num_cells(self):
        return len(self.conn)

    @property
    def num_nodes(self):
        return self.coords.shape[0]

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def edges(self):
        return self.edges_to_cells.keys()

    @staticmethod
    def edge_hash(i,j):
        """Hashes edges in a direction-independent way."""
        return tuple(sorted((i,j)))

    @property
    @cache
    def edge_counts(self):
        items = self.edges_to_cells.items()
        return dict((e,len(v)) for (e,v) in items)

    @property
    @cache
    def edges_to_cells(self):
        e2c = collections.defaultdict(list)
        for i,c in enumerate(self.conn):
            for e in self.cell_edges(c):
                e2c[e].append(i)
        return e2c

    def cell_edges(self, conn):
        for i in range(len(conn)):
            yield self.edge_hash(conn[i], conn[(i+1)%len(conn)])

    @property
    @cache
    def boundary_edges(self):
        """Return edges in the boundary of the mesh, ordered around the boundary."""
        be = sorted([k for (k,count) in self.edge_counts.items() if count == 1], key=lambda a : a[0])
        seed = be.pop(0)
        be_ordered = [seed,]

        done = False
        while not done:
            try:
                new_e = next( e for e in be if e[0] == be_ordered[-1][1])
                be.remove(new_e)
            except StopIteration:
                new_e = next( e for e in be if e[1] == be_ordered[-1][1])
                be.remove(new_e)
                new_e = tuple(reversed(new_e))

            be_ordered.append(new_e)
            done = len(be) == 0
        assert(be_ordered[-1][-1] == be_ordered[0][0])
        return be_ordered

    @property
    @cache
    def boundary_nodes(self):
        return [e[0] for e in self.boundary_edges]

    def check_handedness(self):
        """Ensures all cells are oriented via the right-hand-rule, i.e. in the +z direction."""
        for conn in self._conn:
            points = np.array([self.coords[c] for c in conn])
            cross = 0
            for i in range(len(points)):
                im = i - 1
                ip = i + 1
                if ip == len(points):
                    ip = 0

                p = points[ip] - points[i]
                m = points[i] - points[im]
                cross = cross + p[1] * m[0] - p[0] * m[1]
            if cross < 0:
                conn.reverse()

    def validate(self):
        # validate coords
        assert(isinstance(self.coords, np.ndarray))
        assert(len(self.coords.shape) == 2)
        assert(self.coords.shape[1] in [2,3])

        # validate conn
        assert(min(c for conn in self.conn for c in conn) >= 0)
        assert(max(c for conn in self.conn for c in conn) < self.coords.shape[0])

        # validate labeled_sets
        assert(len(set(ls.setid for ls in self.labeled_sets)) == len(self.labeled_sets))

        for ls in self.labeled_sets:
            is_tuple = False
            if ls.entity == 'CELL':
               size = self.num_cells
            elif ls.entity == 'FACE':
                size = self.num_nodes # note there are no faces, edges are tuples of nodes
                is_tuple = True
            elif ls.entity == 'NODE':
                size = self.num_nodes
            ls.validate(size, is_tuple)

    def next_available_labeled_setid(self):
        """Returns next available LS id."""
        i = 10000
        while any(i == ls.setid for ls in self.labeled_sets):
            i += 1
        return i

    def compute_centroid(self, c):
        """Computes, based on coords, the centroid of a cell with ID c"""
        points = np.array([self.coords[i] for i in self.conn[c]])
        return points.mean(axis=0)

    @property
    @cache
    def centroids(self):
        """Calculate surface mesh centroids."""
        return np.array([self.compute_centroid(c) for c in range(self.num_cells)])

    def plot(self, color=None, ax=None):
        """Plot the flattened 2D mesh."""
        if color is None:
            cm = watershed_workflow.colors.cm_mapper(0,self.num_cells-1)
            colors = [cm(i) for i in range(self.num_cells)]
        else:
            colors = color

        verts = [[self.coords[i,0:2] for i in f] for f in self.conn]
        from matplotlib import collections
        gons = collections.PolyCollection(verts, facecolors=colors)
        from matplotlib import pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1,1)
        ax.add_collection(gons)
        ax.autoscale_view()

    def write_vtk(self, filename):
        """Writes to VTK."""
        assert(all(len(c) == 3 for c in self.conn))
        watershed_workflow.vtk_io.write(filename, self.coords, {'triangle':np.array(self.conn)})

    def transform(self, mat=None, shift=None):
        """Transform a 2D mesh"""
        if mat is None:
            mat = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
        if shift is None:
            shift = np.array([0,0,0])

        new_coords = []
        for c in self.coords:
            assert(c.shape == (3,))
            tc = mat @ c + shift
            assert(tc.shape == (3,))
            new_coords.append(tc)
        new_coords = np.array(new_coords)
        assert(new_coords.shape == self.coords.shape)
        self.coords = new_coords

        # toss geometry cache
        if hasattr(self, '_centroids'):
            del self._centroids

    @classmethod
    def read_VTK(cls, filename):
        """Constructor from a VTK file."""
        try:
            return cls.read_VTK_Simplices(filename)
        except AssertionError:
            return cls.read_VTK_Unstructured(filename)

    @classmethod
    def read_VTK_Unstructured(cls, filename):
        """Constructor from an unstructured VTK file."""
        with open(filename,'rb') as fid:
            points_found = False
            polygons_found = False
            while True:
                line = fid.readline().decode('utf-8')
                if not line:
                    # EOF
                    break

                line = line.strip()
                if len(line) == 0:
                    continue

                split = line.split()
                section = split[0]

                if section == 'POINTS':
                    ncoords = int(split[1])
                    points = np.fromfile(fid, count=ncoords*3, sep=' ', dtype='d')
                    points = points.reshape(ncoords, 3)
                    points_found = True

                elif section == 'POLYGONS':
                    ncells = int(split[1])
                    n_to_read = int(split[2])

                    gons = []

                    data = np.fromfile(fid, count=n_to_read, sep=' ', dtype='i')
                    idx = 0
                    for i in range(ncells):
                        n_in_gon = data[idx]
                        gon = list(data[idx+1:idx+1+n_in_gon])

                        # check handedness -- need normals to point up!
                        cross = []
                        for i in range(len(gon)):
                            if i == len(gon)-1:
                                ip = 0
                                ipp = 1
                            elif i == len(gon)-2:
                                ip = i+1
                                ipp = 0
                            else:
                                ip = i+1
                                ipp = i+2
                            d2 = points[gon[ipp]] - points[gon[ip]]
                            d1 = points[gon[i]] - points[gon[ip]]
                            cross.append(np.cross(d2, d1))
                        if (np.array([c[2] for c in cross]).mean() < 0):
                            gon.reverse()

                        gons.append(gon)

                        idx += n_in_gon + 1
                    assert(idx == n_to_read)
                    polygons_found = True

        if not points_found:
            raise RuntimeError("Unstructured VTK must contain sections 'POINTS'")
        if not polygons_found:
            raise RuntimeError("Unstructured VTK must contain sections 'POLYGONS'")
        return cls(points, gons)

    @classmethod
    def read_VTK_Simplices(cls, filename):
        """Constructor from an structured VTK file.
        """
        with open(filename,'rb') as fid:
            data = watershed_workflow.vtk_io.read_buffer(fid)

        points = data[0]
        if len(data[1]) != 1:
            raise RuntimeError("Simplex VTK file is readable by vtk_io but not by meshing_ats.  Includes: %r"%data[1].keys())

        gons = next(v for v in data[1].values())
        gons = gons.tolist()

        # check handedness
        for gon in gons:
            cross = []
            for i in range(len(gon)):
                if i == len(gon)-1:
                    ip = 0
                    ipp = 1
                elif i == len(gon)-2:
                    ip = i+1
                    ipp = 0
                else:
                    ip = i+1
                    ipp = i+2
                d2 = points[gon[ipp]] - points[gon[ip]]
                d1 = points[gon[i]] - points[gon[ip]]
                cross.append(np.cross(d2, d1))
            if (np.array([c[2] for c in cross]).mean() < 0):
                gon.reverse()

        return cls(points, gons)

    @classmethod
    def from_Transect(cls, x, z, width=1, **kwargs):
        """Creates a 2D surface strip mesh from transect data"""
        # coordinates
        if (type(width) is list or type(width) is np.ndarray):
            variable_width = True
            y = np.array([0,1])
        else:
            variable_width = False
            y = np.array([-width/2,width/2])

        Xc, Yc = np.meshgrid(x, y)
        if variable_width:
            assert(Yc.shape[0] == 2)
            assert(len(width) == Yc.shape[1])
            assert(min(width) > 0.)
            Yc[0,:] = -width/2.
            Yc[1,:] = width/2.

        Xc = Xc.flatten()
        Yc = Yc.flatten()
        Zc = np.concatenate([z,z])

        # connectivity
        nsurf_cells = len(x)-1
        conn = []
        for i in range(nsurf_cells):
            conn.append([i, i+1, nsurf_cells + i + 2, nsurf_cells + i + 1])

        points = np.array([Xc, Yc, Zc])
        return cls(points.transpose(), conn, **kwargs)


    def to_dual(self):
        """Creates a 2D surface mesh from a primal 2D triangular surface mesh.

        Returns
        -------
        dual_nodes : np.ndarray
        dual_conn : list(lists)
            The nodes and cell_to_node_conn of a 2D, polygonal, nearly Voronoi
            mesh that is the truncated dual of self.  Here we say nearly because
            the boundary triangles (see note below) are not Voronoi.
        dual_from_primal_mapping : np.array( (len(dual_conn),), 'i')
            Mapping from a dual cell to the primal node it is based on.
            Without the boundary, this would be simply
            arange(0,len(dual_conn)), but beacuse we added triangles on the
            boundary, this mapping is useful.

        Note
        ----
        - Nodes of the dual are on circumcenters of the primal triangles.
        - Centers of the dual are numbered by nodes of the primal mesh, modulo
          the boundary.
        - At the boundary, the truncated dual polygon may be non-convex.  To
          avoid this issue, we triangulate boundary polygons.

        We do not simply construct a mesh because likely the user needs to
        elevate, generate material IDs, labeled sets, etc, prior to mesh
        construction.

        """
        logging.info("Constructing Mesh2D as dual of a triangulation.")
        logging.info("-- confirming triangulation (note, not checking delaunay, buyer beware)")
        for c in self.conn:
            assert(len(c) == 3) # check all triangles

        def circumcenter(p1,p2,p3):
            d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] *
                     (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

            xv = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2)
                  * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
            yv = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2)
                  * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
            return (xv, yv)

        # dual nodes are given by:
        # - primal cell circumcenters
        # - primal nodes on the boundary
        # - primal edge midpoints on the boundary (note this length is the same as primal nodes on the boundary)

        # dual cells are given by:
        # - primal cell nodes (modulo the boundary)

        coords = self.coords[:,0:2]
        logging.info("-- computing primary boundary edges")
        boundary_edges = self.boundary_edges
        n_dual_nodes = len(self.conn) + 2*len(boundary_edges)
        logging.info("     n_primal_cell = {}, n_boundary_edges = {}, n_dual_nodes = "
                    "{}".format(len(self.conn), len(boundary_edges), n_dual_nodes))

        # create space to populate dual coordinates
        dual_nodes = np.zeros((n_dual_nodes, 2), 'd')
        dual_from_primal_mapping = list(range(len(coords)))

        # Create a list of lists for dual cells -- at this point the truncated
        # dual (e.g. allow non-convex boundary polygons), so one per primal
        # node.  Also create a flag array for whether a given dual cell is on the
        # boundary and so might be non-convex.
        #
        # Note that both of these will be indexed by the index of the
        # corresponding primal node.
        dual_cells = [list() for i in range(len(coords))]
        is_boundary = np.zeros(len(dual_cells), 'i')

        # Loop over all primal cells (triangles), adding the circumcenter
        # as a dual node and sticking that node in three dual cells rooted
        # at the three primal nodes.
        logging.info("-- computing dual nodes")
        i_dual_node = 0
        for j, c in enumerate(self.conn):
            dual_nodes[i_dual_node][:] = circumcenter(coords[c[0]], coords[c[1]], coords[c[2]])
            dual_cells[c[0]].append(i_dual_node)
            dual_cells[c[1]].append(i_dual_node)
            dual_cells[c[2]].append(i_dual_node)
            i_dual_node += 1

        logging.info("    added {} tri centroid nodes".format(i_dual_node))

        # Loop over the boundary, adding both the primal nodes and the edge
        # midpoints as dual nodes.
        #
        # Add the primal node and two midpoints on either side to the list
        # of dual nodes in the cell "rooted at" the primal node.
        for i, e in enumerate(boundary_edges):
            # add the primal node always
            my_primal_node_dual_node = i_dual_node
            dual_nodes[i_dual_node][:] = coords[e[0]]
            i_dual_node += 1

            my_cell = list()

            # stick in the previous midpoint node, add to my_cell
            if i == 0:
                # reserve a spot for the last midpoint
                first_cell_added = e[0]
                my_cell.append(-1)
            else:
                my_cell.append(prev_midp_n)

            # stick in the next midpoint node, add to my_cell
            next_midp_n = i_dual_node
            next_midp = (coords[e[0]][:] + coords[e[1]][:])/2.
            dual_nodes[i_dual_node][:] = next_midp
            i_dual_node += 1
            my_cell.append(next_midp_n)

            # add the primal node to my_cell
            my_cell.append(my_primal_node_dual_node)
            dual_cells[e[0]].extend(my_cell)

            is_boundary[e[0]] = 1
            prev_midp_n = next_midp_n

        # patch up the first cell added
        assert(dual_cells[first_cell_added][-3] == -1)
        dual_cells[first_cell_added][-3] = prev_midp_n

        logging.info("    added {} boundary nodes".format(len(boundary_edges)))

        #
        # Now every dual cell has a list of nodes (in no particular order).
        # But some of these nodes may be duplicated -- either the circumcenter
        # of two adjacent triangles may both be on the boundary, allowing for
        # coincident points at the midpoint of the coincident faces, or (more
        # likely) there was a circumcenter on the boundary, meaning it is
        # coincident with the boundary edge's midpoint.
        #
        # Order those nodes, and collect a list of coincident nodes for removal.
        logging.info("-- Finding duplicates and ordering conn_cell_to_node")
        nodes_to_kill = dict() # coincident nodes (key = node to remove, val = coincident node)
        for i in range(len(dual_cells)):
            c = dual_cells[i]

            if is_boundary[i]:
                # check for duplicate nodes
                to_pop = []
                for k in range(len(c)):
                    for j in range(k+1, len(c)):
                        if (np.linalg.norm(dual_nodes[c[k]] - dual_nodes[c[j]]) < self.eps):
                            logging.info("    found dup on boundary cell {} = {}".format(i, c))
                            logging.info("        indices = {}, {}".format(k,j))
                            logging.info("        nodes = {}, {}".format(c[k],c[j]))
                            logging.info("        coords = {}, {}".format(dual_nodes[c[k]], dual_nodes[c[j]]))

                            if c[k] in nodes_to_kill:
                                assert c[j] == nodes_to_kill[c[k]]
                                assert k < len(c) - 3
                                to_pop.append(k)
                            elif c[j] in nodes_to_kill:
                                assert c[k] == nodes_to_kill[c[j]]
                                assert j < len(c) - 3
                                to_pop.append(j)
                            else:
                                assert k < len(c)-3
                                nodes_to_kill[c[k]] = c[j]
                                to_pop.append(k)

                # remove the duplicated nodes from the cell_to_node_conn
                for j in reversed(sorted(to_pop)):
                    c.pop(j)

                # may not be convex -- triangulate
                c_orig = c[:]
                c0 = c[-1] # the primal node
                cup = c[-2] # boundary midpoint, one direction
                cdn = c[-3] # boundary midpoint, the other direction

                # order the nodes (minus the primal node) clockwise around the primal node
                cell_coords = np.array([dual_nodes[cj] for cj in c[:-1]]) - dual_nodes[c0]
                angle = np.array([np.arctan2(cell_coords[j,1], cell_coords[j,0]) for j in range(len(cell_coords))])
                order = np.argsort(angle)
                c = [c[j] for j in order]

                # now find what is "inside" and what is "outside" the domain by
                # finding up and dn, making one the 0th point and the other the
                # last point
                up_i = c.index(cup)
                dn_i = c.index(cdn)

                if dn_i == (up_i + 1)%len(c):
                    cn = c[dn_i:] + c[0:dn_i]
                elif up_i == (dn_i + 1)%len(c):
                    cn = c[up_i:] + c[0:up_i]
                else:
                    # I screwed up... debug me!
                    print("Uh oh bad geom: up_i = {}, dn_i = {}, c = {}".format(up_i, dn_i, c))
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    cc_sorted = np.array([cell_coords[k] for k in order]) + dual_nodes[c0]
                    cb_sorted = np.array([dual_nodes[cdn], dual_nodes[c], dual_nodes[cup]])
                    ax.plot(cc_sorted[:,0], cc_sorted[:,1], 'k-x')
                    ax.scatter(dual_nodes[c0,0], dual_nodes[c0,1], color='r')
                    ax.scatter(dual_nodes[cup,0], dual_nodes[cup,1], color='m')
                    ax.scatter(dual_nodes[cdn,0], dual_nodes[cdn,1], color='b')
                    plt.show()

                    fig = plt.figure(figsize=figsize)
                    ax = watershed_workflow.plot.get_ax(crs, fig)

                    mp = watershed_workflow.plot.triangulation(mesh_points3, mesh_tris, crs, ax=ax,
                                         color='elevation', edgecolor='white', linewidth=0.4)
                    cbar = fig.colorbar(mp, orientation="horizontal", pad=0.05)
                    #watershed_workflow.plot.hucs(shapes, crs, ax=ax, color='k', linewidth=1)
                    watershed_workflow.plot.shply([shapely.geometry.LineString(cc_sorted),], crs, ax=ax, color='red', linewidth=1)
                    watershed_workflow.plot.shply([shapely.geometry.LineString(cb_sorted),], crs, ax=ax, color='blue', linewidth=1)
                    ax.set_aspect('equal', 'datalim')

                    raise RuntimeError('uh oh borked geom')

                # triangulate the truncated dual polygon, always including the
                # primal node to guarantee all triangles exist and partition
                # the polygon.
                for k in range(len(cn)-1):
                    if k == 0:
                        dual_cells[i] = [c0, cn[k+1], cn[k]]
                    else:
                        dual_cells.append([c0, cn[k+1], cn[k]])
                        dual_from_primal_mapping.append(i)

            else:
                # NOT a boundary polygon.  Simply order and check for duplicate nodes.
                to_pop = []
                for k in range(len(c)):
                    for j in range(k+1, len(c)):
                        if (np.linalg.norm(dual_nodes[c[k]] - dual_nodes[c[j]]) < self.eps):
                            logging.info("    found dup on interior cell {} = {}".format(i, c))
                            logging.info("        indices = {}, {}".format(k,j))
                            logging.info("        nodes = {}, {}".format(c[k],c[j]))
                            logging.info("        coords = {}, {}".format(dual_nodes[c[k]], dual_nodes[c[j]]))

                            if c[k] in nodes_to_kill:
                                assert c[j] == nodes_to_kill[c[k]]
                                to_pop.append(k)
                            elif c[j] in nodes_to_kill:
                                assert c[k] == nodes_to_kill[c[j]]
                                to_pop.append(j)
                            else:
                                nodes_to_kill[c[k]] = c[j]
                                to_pop.append(k)

                # remove the duplicated nodes from the cell_to_node_conn
                for j in reversed(sorted(to_pop)):
                    c.pop(j)

                # order around the primal node (now dual cell centroid)
                cell_coords = np.array([dual_nodes[j] for j in c]) - coords[i]
                angle = np.array([np.arctan2(cell_coords[j,1], cell_coords[j,0]) for j in range(len(cell_coords))])
                order = np.argsort(angle)
                dual_cells[i] = [c[j] for j in order]

        logging.info("-- removing duplicate nodes")
        # note this requires both removing the duplicates from the coordinate
        # list, but also remapping to new numbers that range in [0,
        # num_not_removed_nodes).  To do the latter, we create a map from old
        # indices to new indices, where removed indices are mapped to their
        # coincident, new index.  These old nodes may have appeared in cells
        # that did not actually include its duplicate, so must get remapped to
        # the coincident node.
        i = 0
        compression_map = -np.ones(len(dual_nodes), 'i')
        for j in range(len(dual_nodes)):
            if j not in nodes_to_kill:
                compression_map[j] = i
                i += 1
        for j in nodes_to_kill.keys():
            compression_map[j] = compression_map[nodes_to_kill[j]]
        assert(compression_map.min() >= 0)

        # -- delete the nodes
        to_kill = sorted(list(nodes_to_kill.keys()))
        dual_nodes = np.delete(dual_nodes, to_kill, axis=0)
        # -- remap the conn
        for c in dual_cells:
            for j in range(len(c)):
                c[j] = compression_map[c[j]]
            assert(min(c) >= 0)

        return dual_nodes, dual_cells, np.array(dual_from_primal_mapping)

    
@attr.define(slots=False)
class Mesh3D:
    """A 3D mesh class.

    Parameters
    ----------
    coords : np.ndarray(NCOORDS,3)
        Array of coordinates of the 3D mesh.
    face_to_node_conn : list(lists)
        List of lists of indices into coords that form the faces.
    cell_to_face_conn : list(lists)
        List of lists of indices into face_to_node_conn that form the
        cells.
    side_sets : list(SideSet), optional
        List of side sets to add to the mesh.
    labeled_sets : list(LabeledSet), optional
        List of labeled sets to add to the mesh.
    material_ids : np.array((len(cell_to_face_conn),),'i'), optional
        Array of length num_cells that specifies material IDs
    crs : CRS
        Keep the coordinate system for reference.
    eps : float, optional=0.01
        A small measure of length between coords.

    Note: (coords, conn) may be output provided by a
    watershed_workflow.triangulation.triangulate() call.
    """
    coords = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    _face_to_node_conn = attr.ib()
    _cell_to_face_conn = attr.ib()
    labeled_sets = attr.ib(factory=list)
    side_sets = attr.ib(factory=list)
    material_ids = attr.ib(default=None)
    crs = attr.ib(default=None)
    eps = attr.ib(default=0.001)
    _validate = attr.ib(default=False)
                     
    def __attrs_post_init__(self):
        if self._validate:
            self.validate()
        del self._validate

    @property
    def face_to_node_conn(self):
        return self._face_to_node_conn

    @property
    def cell_to_face_conn(self):
        return self._cell_to_face_conn

    @property
    def material_ids_list(self):
        return sorted(list(set(self.material_ids)))
    
    @property
    def dim(self):
        return self.coords.shape[1]

    @property
    def num_cells(self):
        return len(self.cell_to_face_conn)

    @property
    def num_nodes(self):
        return self.coords.shape[0]

    @property
    def num_faces(self):
        return len(self.face_to_node_conn)
        
    
    def validate(self):
        """Checks the validity of the mesh, or throws an AssertionError."""
        # validate coords
        assert(isinstance(self.coords, np.ndarray))
        assert(len(self.coords.shape) == 2)
        assert(self.coords.shape[1] == 3)

        # validate conn
        assert(min(c for conn in self.face_to_node_conn for c in conn) >= 0)
        assert(max(c for conn in self.face_to_node_conn for c in conn) < len(self.coords))

        assert(min(c for conn in self.cell_to_face_conn for c in conn) >= 0)
        assert(max(c for conn in self.cell_to_face_conn for c in conn) < len(self.face_to_node_conn))

        # validate labeled_sets and side sets
        ls_ss_ids = [ls.setid for ls in self.labeled_sets] + \
            [ss.setid for ss in self.side_sets]
        assert(len(set(ls_ss_ids)) == len(ls_ss_ids))

        for ls in self.labeled_sets:
            if ls.entity == 'CELL':
               size = self.num_cells
            elif ls.entity == 'NODE':
                size = self.num_nodes
            else:
                assert(False) # no face or edge sets
            ls.validate(size, False)

        for ss in self.side_sets:
            ss.validate(self.cell_to_face_conn)

        # validate material ids
        assert(self.material_ids is not None)
        assert(len(self.material_ids) == len(self.cell_to_face_conn))

    def next_available_labeled_setid(self):
        """Returns next available LS id."""
        i = 10000
        while any(i == ls.setid for ls in self.labeled_sets) or \
              any(i == ls.setid for ls in self.side_sets):
            i += 1
        return i

    def write_vtk(self, filename):
        """Writes to VTK.

        Note, this just writes the topology/geometry information, for
        WEDGE type meshes (extruded triangles).  No labeled sets are
        written.  Prefer to use write_exodus() for a fully featured
        mesh.

        """
        assert(all(len(c) == 5 for c in self.cell_to_face_conn))
        wedges = []
        for c2f in self.cell_to_face_conn:
            fup = c2f[0]
            fdn = c2f[1]
            assert(len(self.face_to_node_conn[fup]) == 3 and
                   len(self.face_to_node_conn[fdn]) == 3)
            wedges.append(self.face_to_node_conn[fup] + self.face_to_node_conn[fdn])
        watershed_workflow.vtk_io.write(filename, self.coords, {'wedge':np.array(wedges)})

    def write_exodus(self, filename, face_block_mode="one block"):
        """Write the 3D mesh to ExodusII using arbitrary polyhedra spec"""
        if exodus is None:
            raise ImportError("The python ExodusII wrappers were not found, please see the installation documentation to install Exodus")
        
        # Note exodus uses the term element instead of cell, so we
        # swap to that term in this method.

        # put elems in with blocks, which renumbers the elems, so we
        # have to track sidesets.  Therefore we keep a map of old elem
        # to new elem ordering
        #
        # also, though not required by the spec, paraview and visit
        # seem to crash if num_face_blocks != num_elem_blocks.  So
        # make face blocks here too, which requires renumbering the faces.

        # -- first pass, form all element blocks and make the map from old-to-new
        new_to_old_elems = []
        elem_blks = []
        for i_m,m_id in enumerate(self.material_ids_list):
            # split out elems of this material, save new_to_old map
            elems_tuple = [(i,c) for (i,c) in enumerate(self.cell_to_face_conn) if self.material_ids[i] == m_id]
            new_to_old_elems.extend([i for (i,c) in elems_tuple])
            elems = [c for (i,c) in elems_tuple]
            elem_blks.append(elems)

        old_to_new_elems = sorted([(old,i) for (i,old) in enumerate(new_to_old_elems)], key=lambda a: a[0])

        # -- deal with faces, form all face blocks and make the map from old-to-new
        face_blks = []
        if face_block_mode == "one block":
            # no reordering of faces needed
            face_blks.append(self.face_to_node_conn)

        elif face_block_mode == "n blocks, not duplicated":
            used_faces = np.zeros((len(self.face_to_node_conn),),'bool')
            new_to_old_faces = []
            for i_m,m_id in enumerate(self.material_ids_list):
                # split out faces of this material, save new_to_old map
                def used(f):
                    result = used_faces[f]
                    used_faces[f] = True
                    return result

                elem_blk = elem_blks[i_m]
                faces_tuple = [(f,self.face_to_node_conn[f]) for c in elem_blk for (j,f) in enumerate(c) if not used(f)]
                new_to_old_faces.extend([j for (j,f) in faces_tuple])
                faces = [f for (j,f) in faces_tuple]
                face_blks.append(faces)

            # get the renumbering in the elems
            old_to_new_faces = sorted([(old,j) for (j,old) in enumerate(new_to_old_faces)], key=lambda a: a[0])
            elem_blks = [[[old_to_new_faces[f][1] for f in c] for c in elem_blk] for elem_blk in elem_blks]

        elif face_block_mode == "n blocks, duplicated":
            elem_blks_new = []
            offset = 0
            for i_m, m_id in enumerate(self.material_ids_list):
                used_faces = np.zeros((len(self.face_to_node_conn),),'bool')
                def used(f):
                    result = used_faces[f]
                    used_faces[f] = True
                    return result

                elem_blk = elem_blks[i_m]

                tuple_old_f = [(f,self.face_to_node_conn[f]) for c in elem_blk for f in c if not used(f)]
                tuple_new_old_f = [(new,old,f) for (new,(old,f)) in enumerate(tuple_old_f)]

                old_to_new_blk = np.zeros((len(self.face_to_node_conn),),'i')-1
                for new,old,f in tuple_new_old_f:
                    old_to_new_blk[old] = new + offset

                elem_blk_new = [[old_to_new_blk[f] for f in c] for c in elem_blk]
                #offset = offset + len(ftuple_new)

                elem_blks_new.append(elem_blk_new)
                face_blks.append([f for i,j,f in tuple_new_old_f])
            elem_blks = elem_blks_new
        elif face_block_mode == "one block, repeated":
            # no reordering of faces needed, just repeat
            for eblock in elem_blks:
                face_blks.append(self.face_to_node_conn)
        else:
            raise RuntimeError("Invalid face_block_mode: '%s', valid='one block', 'n blocks, duplicated', 'n blocks, not duplicated'"%face_block_mode)


        # open the mesh file
        num_elems = sum(len(elem_blk) for elem_blk in elem_blks)
        num_faces = sum(len(face_blk) for face_blk in face_blks)

        filename_base = os.path.split(filename)[-1]
        ep = exodus.ex_init_params(title=filename_base.encode('ascii'),
                                   num_dim=3,
                                   num_nodes=self.num_nodes,
                                   num_face=num_faces,
                                   num_face_blk=len(face_blks),
                                   num_elem=num_elems,
                                   num_elem_blk=len(elem_blks),
                                   num_side_sets=len(self.side_sets),
                                   num_elem_sets=sum(1 for ls in self.labeled_sets if ls.entity == 'CELL'),
                                   )
        e = exodus.exodus(filename, mode='w', array_type='numpy', init_params=ep)

        # put the coordinates
        e.put_coord_names(['coordX', 'coordY', 'coordZ'])
        e.put_coords(self.coords[:,0], self.coords[:,1], self.coords[:,2])

        # put the face blocks
        for i_blk, face_blk in enumerate(face_blks):
            face_raveled = [n for f in face_blk for n in f]
            e.put_polyhedra_face_blk(i_blk+1, len(face_blk), len(face_raveled), 0)
            e.put_node_count_per_face(i_blk+1, np.array([len(f) for f in face_blk]))
            e.put_face_node_conn(i_blk+1, np.array(face_raveled)+1)

        # put the elem blocks
        assert len(elem_blks) == len(self.material_ids_list)
        for i_blk, (m_id, elem_blk) in enumerate(zip(self.material_ids_list, elem_blks)):
            elems_raveled = [f for c in elem_blk for f in c]

            e.put_polyhedra_elem_blk(m_id, len(elem_blk), len(elems_raveled), 0)
            e.put_elem_blk_name(m_id, 'MATERIAL_ID_%d'%m_id)
            e.put_face_count_per_polyhedra(m_id, np.array([len(c) for c in elem_blk]))
            e.put_elem_face_conn(m_id, np.array(elems_raveled)+1)

        # add sidesets
        for ss in self.side_sets:
            logging.info(f'adding side set: {ss.setid}')
            for elem in ss.cell_list:
                assert old_to_new_elems[elem][0] == elem
            new_elem_list = sorted([(old_to_new_elems[elem][1],side) for (elem,side) in zip(ss.cell_list, ss.side_list)])
            e.put_side_set_params(ss.setid, len(ss.cell_list), 0)
            e.put_side_set_name(ss.setid, ss.name)
            e.put_side_set(ss.setid,
                           np.array([e_id for (e_id, side) in new_elem_list])+1,
                           np.array([side for (e_id, side) in new_elem_list])+1)

        # add labeled sets
        for ls in self.labeled_sets:
            if ls.entity == 'CELL':
                logging.info(f'adding elem set: {ls.setid}')
                new_elem_list = sorted([old_to_new_elems[elem][1] for elem in ls.ent_ids])
                e.put_elem_set_params(ls.setid, len(new_elem_list), None)
                e.put_elem_set_name(ls.setid, ls.name)
                e.put_elem_set(ls.setid, np.array(new_elem_list)+1)
            else:
                warnings.warning(f'Cannot write labeled set of type {ls.entity}')

        # finish and close
        e.close()


    @staticmethod
    def summarize_extrusion(layer_types, layer_data, ncells_per_layer, mat_ids, surface_cell_id=0):
        """
        Summarizes extruded data by printing info to log file.

        This is useful in rapidly debugging and understanding the layering before
        you do the extrusion process.
        """
        count = 0
        logging.info("Cell summary:")
        logging.info("-"*60)
        logging.info("l_id\t| c_id\t|mat_id\t| dz\t\t| z_top")
        logging.info("-"*60)
        rep_z = 0.
        for i,thick in enumerate(layer_data):
            for j in range(ncells_per_layer[i]):
                try:
                    mat_id = mat_ids[i][surface_cell_id]
                except TypeError:
                    mat_id = mat_ids[i]
                logging.info(" %02i \t| %02i \t| %4i \t| %10.6f \t| %10.6f"%(i,
                             count,mat_id,thick/ncells_per_layer[i], rep_z))
                count += 1
                rep_z += thick/ncells_per_layer[i]


    @classmethod
    def extruded_Mesh2D(cls, mesh2D, layer_types, layer_data, ncells_per_layer, mat_ids):
        """
        Regularly extrude a 2D mesh to make a 3D mesh.

        mesh2D              | a Mesh2D object
        layer_types         | either a string (type) or list of strings (types)
        layer_data          | array of data needed (specific to the type)
        ncells_per_layer    | either a single integer (same number of cells in all
                            | layers) or a list of number of cells in the layer
        mat_ids             | either a single integer (one mat_id for all layers)
                            | or a list of integers (mat_id for each layer)
                            | or a 2D numpy array of integers (mat_id for each layer and
                            | each column: [layer_id, surface_cell_id])

        types:
          - 'constant'      | (data=float thickness) uniform thickness
          - 'function'      | (data=function or functor) thickness as a function
                            | of (x,y)
          - 'snapped'       | (data=float z coordinate) snap the layer to
                            | provided z coordinate, telescoping as needed
          - 'node'          | thickness provided on each node of the surface domain
          - 'cell'          | thickness provided on each cell of the surface domain,
                            | interpolate to nodes

        NOTE: dz is uniform through the layer in all but the 'snapped' case
        NOTE: 2D mesh is always labeled 'surface', extrusion is always downwards
        """

        # make the data all lists
        # ---------------------------------
        def is_list(data):
            if type(data) is str:
                return False
            try:
                len(data)
            except TypeError:
                return False
            else:
                return True

        if is_list(layer_types):
            if not is_list(layer_data):
                layer_data = [layer_data,]*len(layer_types)
            else:
                assert len(layer_data) == len(layer_types)

            if not is_list(ncells_per_layer):
                ncells_per_layer = [ncells_per_layer,]*len(layer_types)
            else:
                assert len(ncells_per_layer) == len(layer_types)

        elif is_list(layer_data):
            layer_types = [layer_types,]*len(layer_data)

            if not is_list(ncells_per_layer):
                ncells_per_layer = [ncells_per_layer,]*len(layer_data)
            else:
                assert len(ncells_per_layer) == len(layer_data)

        elif is_list(ncells_per_layer):
            layer_type = [layer_type,]*len(ncells_per_layer)
            layer_data = [layer_data,]*len(ncells_per_layer)
        else:
            layer_type = [layer_type,]
            layer_data = [layer_data,]
            ncells_per_layer = [ncells_per_layer,]

        # helper data and functions for mapping indices from 2D to 3D
        # ------------------------------------------------------------------
        if min(ncells_per_layer) < 0:
            raise RuntimeError("Invalid number of cells, negative value provided.")
        ncells_tall = sum(ncells_per_layer)
        ncells_total = ncells_tall * mesh2D.num_cells
        nfaces_total = (ncells_tall+1) * mesh2D.num_cells + ncells_tall * mesh2D.num_edges
        nnodes_total = (ncells_tall+1) * mesh2D.num_nodes

        eh = _ExtrusionHelper(ncells_tall, mesh2D.num_cells)

        np_mat_ids = np.array(mat_ids, dtype=int)
        if np_mat_ids.size == np.size(np_mat_ids, 0):
            if np_mat_ids.size == 1:
                np_mat_ids = np.full((len(ncells_per_layer), mesh2D.num_cells), mat_ids[0], dtype=int)
            else:
                np_mat_ids = np.empty((len(ncells_per_layer), mesh2D.num_cells), dtype=int)
                for ilay in range(len(ncells_per_layer)):
                    np_mat_ids[ilay, :] = np.full(mesh2D.num_cells, mat_ids[ilay], dtype=int)


        # create coordinates
        # ---------------------------------
        coords = np.zeros((mesh2D.coords.shape[0],ncells_tall+1, 3),'d')
        coords[:,:,0:2] = np.expand_dims(mesh2D.coords[:,0:2],1)

        if mesh2D.dim == 3:
            coords[:,0,2] = mesh2D.coords[:,2]
        # else the surface is at 0 depth

        cell_layer_start = 0
        for layer_type, layer_datum, ncells in zip(layer_types, layer_data, ncells_per_layer):
            if layer_type.lower() == 'constant':
                dz = float(layer_datum) / ncells
                for i in range(1,ncells+1):
                    coords[:,cell_layer_start+i,2] = coords[:,cell_layer_start,2] - i * dz

            else:
                # allocate an array of coordinates for the bottom of the layer
                layer_bottom = np.zeros((mesh2D.coords.shape[0],),'d')

                if layer_type.lower() == 'snapped':
                    # layer bottom is uniform
                    layer_bottom[:] = layer_datum

                elif layer_type.lower() == 'function':
                    # layer thickness is given by a function evaluation of x,y
                    for node_col in range(mesh2D.coords.shape[0]):
                        layer_bottom[node_col] = coords[node_col,cell_layer_start,2] - layer_datum(coords[node_col,0,0], coords[node_col,0,1])

                elif layer_type.lower() == 'node':
                    # layer bottom specifically provided through thickness
                    layer_bottom[:] = coords[:,cell_layer_start,2] - layer_datum

                elif layer_type.lower() == 'cell':
                    # interpolate cell thicknesses to node thicknesses
                    import scipy.interpolate
                    centroids = mesh2D.centroids
                    interp = scipy.interpolate.interp2d(centroids[:,0], centroids[:,1], layer_datum, kind='linear')
                    layer_bottom[:] = coords[:,cell_layer_start,2] - interp(mesh2D.coords[:,0], mesh2D.coords[:,1])

                else:
                    raise RuntimeError("Unrecognized layer_type '%s'"%layer_type)

                # linspace from bottom of previous layer to bottom of this layer
                for node_col in range(mesh2D.coords.shape[0]):
                    coords[node_col,cell_layer_start:cell_layer_start+ncells+1,2] = np.linspace(coords[node_col,cell_layer_start,2], layer_bottom[node_col], ncells+1)

            cell_layer_start = cell_layer_start + ncells

        # create faces, face sets, cells
        bottom = []
        surface = []
        faces = []
        cells = [list() for c in range(ncells_total)]

        # -- loop over the columns, adding the horizontal faces
        for col in range(mesh2D.num_cells):
            nodes_2 = mesh2D.conn[col]
            surface.append(eh.col_to_id(col,0))
            for z_face in range(ncells_tall + 1):
                i_f = len(faces)
                f = [eh.node_to_id(n, z_face) for n in nodes_2]

                if z_face != ncells_tall:
                    cells[eh.col_to_id(col, z_face)].append(i_f)
                if z_face != 0:
                    cells[eh.col_to_id(col, z_face-1)].append(i_f)

                faces.append(f)
            bottom.append(eh.col_to_id(col,ncells_tall-1))

        # -- loop over the columns, adding the vertical faces
        added = dict()
        vertical_side_cells = []
        vertical_side_indices = []
        for col in range(mesh2D.num_cells):
            nodes_2 = mesh2D.conn[col]
            for i in range(len(nodes_2)):
                edge = mesh2D.edge_hash(nodes_2[i], nodes_2[(i+1)%len(nodes_2)])
                try:
                    i_e = added[edge]
                except KeyError:
                    # faces not yet added to facelist
                    i_e = len(added.keys())
                    added[edge] = i_e

                    for z_face in range(ncells_tall):
                        i_f = len(faces)
                        assert i_f == eh.edge_to_id(i_e, z_face)
                        f = [eh.node_to_id(edge[0], z_face),
                             eh.node_to_id(edge[1], z_face),
                             eh.node_to_id(edge[1], z_face+1),
                             eh.node_to_id(edge[0], z_face+1)]
                        faces.append(f)
                        face_cell = eh.col_to_id(col, z_face)
                        cells[face_cell].append(i_f)

                        # check if this is an external
                        if mesh2D.edge_counts[edge] == 1:
                            vertical_side_cells.append(face_cell)
                            vertical_side_indices.append(len(cells[face_cell])-1)

                else:
                    # faces already added from previous column
                    for z_face in range(ncells_tall):
                        i_f = eh.edge_to_id(i_e, z_face)
                        cells[eh.col_to_id(col, z_face)].append(i_f)


        # Do some idiot checking
        # -- check we got the expected number of faces
        assert len(faces) == nfaces_total
        # -- check every cell is at least a tet
        for c in cells:
            assert len(c) > 4
        # -- check surface sideset has the right number of entries
        assert len(surface) == mesh2D.num_cells
        # -- check bottom sideset has the right number of entries
        assert len(bottom) == mesh2D.num_cells

        # -- len of vertical sides sideset is number of external edges * number of cells, no pinchouts here
        num_sides = ncells_tall * sum(1 for e,c in mesh2D.edge_counts.items() if c == 1)
        assert num_sides == len(vertical_side_cells)
        assert num_sides == len(vertical_side_indices)

        # make the material ids
        material_ids = np.zeros((len(cells),),'i')
        for col in range(mesh2D.num_cells):
            z_cell = 0
            for ilay in range(len(ncells_per_layer)):
                ncells = ncells_per_layer[ilay]
                for i in range(z_cell, z_cell+ncells):
                    material_ids[eh.col_to_id(col, i)] = np_mat_ids[ilay, col]
                z_cell = z_cell + ncells

        # make the side sets
        side_sets = []
        side_sets.append(SideSet("bottom", 1, bottom, [1,]*len(bottom)))
        side_sets.append(SideSet("surface", 2, surface, [0,]*len(surface)))
        side_sets.append(SideSet("external_sides", 3, vertical_side_cells, vertical_side_indices))

        labeled_sets = []
        for ls in mesh2D.labeled_sets:
            if ls.entity == 'CELL':
                if hasattr(ls, 'to_extrude') and ls.to_extrude:
                    # top surface cells, to become columns of cells
                    elem_list = [eh.col_to_id(c,j) for j in range(ncells_tall) for c in ls.ent_ids]
                    labeled_sets.append(LabeledSet(ls.name, ls.setid, 'CELL', elem_list))
                else:
                    # top surface cells, to become side sets
                    elem_list = [eh.col_to_id(c,0) for c in ls.ent_ids]
                    side_list = [0 for c in ls.ent_ids]
                    side_sets.append(SideSet(ls.name, ls.setid, elem_list, side_list))
            elif ls.entity == 'FACE':
                # top surface faces become faces in the subsurface
                # mesh, as they will extract correctly for surface BCs/observations
                outlet_ids_names = []

                # given a 2D edge, find the 2D cell it touches
                col_ids = [mesh2D.edges_to_cells[mesh2D.edge_hash(e[0],e[1])][0]
                           for e in ls.ent_ids]
                if hasattr(ls, 'to_extrude') and ls.to_extrude:
                    elem_list = [eh.col_to_id(c, i) for c in col_ids for i in range(ncells_tall)]
                    face_list = [eh.edge_to_id(added[mesh2D.edge_hash(e[0],e[1])], i)
                                 for e in ls.ent_ids for i in range(ncells_tall)]
                    side_list = [cells[c].index(f) for (f,c) in zip(face_list,elem_list)]
                else:
                    elem_list = [eh.col_to_id(c, 0) for c in col_ids]
                    face_list = [eh.edge_to_id(added[mesh2D.edge_hash(e[0],e[1])], 0)
                                 for e in ls.ent_ids]
                    side_list = [cells[c].index(f) for (f,c) in zip(face_list,elem_list)]
                side_sets.append(SideSet(ls.name, ls.setid, elem_list, side_list))

        # reshape coords
        coords = coords.reshape(nnodes_total, 3)

        # instantiate the mesh
        m3 = cls(coords, faces, cells, material_ids=material_ids, side_sets=side_sets, labeled_sets=labeled_sets, crs=mesh2D.crs)
        return m3


def telescope_factor(ncells, dz, layer_dz):
    """Calculates a telescoping factor to fill a given layer.

    Calculates a constant geometric factor, such that a layer of thickness
    layer_dz is perfectly filled by ncells in the vertical, where the top cell
    is dz in thickness and each successive cell grows by a factor of that
    factor.

    Parameters
    ----------
    ncells : int
       Number of cells (in the vertical) needed.
    dz : float
       Top cell's thickness in the vertical.
    layer_dz : float
       Thickness of the total layer.

    Returns
    -------
    float
       The telescoping factor.
    """
    if ncells * dz > layer_dz:
        raise ValueError(("Cannot telescope {} cells of thickness at least {} "+
                          "and reach a layer of thickness {}"
                         ).format(ncells, dz, layer_dz))

    def seq(r):
        calc_layer_dz = 0
        dz_new = dz
        for i in range(ncells):
            calc_layer_dz += dz_new
            dz_new *= r

        #print('tried: {} got: {}'.format(r, calc_layer_dz))
        return layer_dz - calc_layer_dz
    res = scipy.optimize.root_scalar(seq, method='bisect', bracket=[1.0001,2], maxiter=1000)
    logging.info("Converged?: ratio = {}, layer z (target = {}) = {}".format(res.root, layer_dz, seq(res.root)))
    return res.root


def optimize_dzs(dz_begin, dz_end, thickness, num_cells, p_thickness=1000, p_dz=10000, p_increasing=1000, p_smooth=10, tol=1):
    """Tries to optimize dzs"""
    pad_thickness = thickness + dz_begin + dz_end

    def penalty_thickness(dzs):
        return abs(pad_thickness - dzs.sum())
    def penalty_dz(dzs):
        return dz_end / dz_begin * abs(dzs[0] - dz_begin) + abs(dzs[-1] - dz_end)
    def penalty_increasing(dzs):
        return np.maximum(0., dzs[0:-1] - dzs[1:]).sum()
    def penalty_smooth(dzs):
        growth_factor = dzs[1:] / dzs[0:-1]
        return np.abs(2*growth_factor[1:-1] - growth_factor[0:-2] - growth_factor[2:]).sum()

    def penalty(dzs):
        return p_thickness * penalty_thickness(dzs) \
            + p_dz * penalty_dz(dzs) \
            + p_increasing * penalty_increasing(dzs) \
            + p_smooth * penalty_smooth(dzs)

    x0 = (dz_begin + dz_end) / 2. * np.ones((num_cells,),'d')
    bounds = [(dz_begin, dz_end),]*num_cells
    res = scipy.optimize.minimize(penalty, x0, bounds=bounds)

    dzs = res.x.copy()[1:-1]

    # MUST have increasing
    dzs.sort()

    # MUST have sum
    dzs = dzs / dzs.sum() * thickness

    return dzs, res



def transform_rotation(radians):
    return np.array([[ np.cos(radians), np.sin(radians), 0],
                     [-np.sin(radians), np.cos(radians), 0],
                     [               0,               0, 1]])


def add_nlcd_labeled_sets(m2, nlcd_colors, nlcd_names):
    """Given a 2D mesh and an array of length m2.num_cells of indices, add labeled sets."""
    inds = np.unique(nlcd_colors)
    for ind in inds:
        ent_ids = list(np.where(nlcd_colors == ind)[0])
        ls = LabeledSet(nlcd_names[ind], int(ind), 'CELL', ent_ids)
        m2.labeled_sets.append(ls)


def _get_labels(polygons):
    labels = []
    for i,p in enumerate(polygons):
        label = f'watershed {i}'
        labels.append(label)
    return labels


def add_watershed_regions(m2, polygons, labels=None):
    """Add labeled sets to m2 for each polygon."""
    import shapely
    if labels is None:
        labels = _get_labels(polygons)
    else:
        assert(len(labels) == len(polygons))

    partitions = [list() for p in polygons]
    for c in range(m2.num_cells):
        cc = m2.compute_centroid(c)
        cc = shapely.geometry.Point(cc[0], cc[1])
        try:
            ip = next(i for (i,p) in enumerate(polygons) if p.contains(cc))
        except StopIteration:
            pass
        else:
            partitions[ip].append(c)

    for label, part in zip(labels, partitions):
        if len(part) > 0:
            # add a region, denoting this one as "to extrude".  This
            # will become the volume region
            setid = m2.next_available_labeled_setid()
            ls = LabeledSet(label, setid, 'CELL', part)
            m2.labeled_sets.append(ls)
            ls.to_extrude = True

            # add a second region, denoting this one as the top surface of faces
            setid2 = m2.next_available_labeled_setid()
            ls2 = LabeledSet(label+' surface', setid2, 'CELL', part)
            m2.labeled_sets.append(ls2)

    return partitions

def add_watershed_regions_and_outlets(m2, hucs, outlet_width=None, labels=None):
    """Add four labeled sets to m2 for each polygon:

    - cells in the polygon, to be extruded
    - cells in the polygon, to be kept as faces upon extrusion
    - boundary of the polygon (edges)
    - outlet of the polygon (edges within outlet width of the outlet)

    """
    polygons = list(hucs.polygons())
    if labels is None:
        labels = _get_labels(polygons)
    else:
        assert(len(labels) == len(polygons))

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
        c = (n1 + n2)/2.
        close = watershed_workflow.utils.close(outlet, tuple(c[0:2]), outlet_width)
        return close

    for label, tris, outlet in zip(labels, partitions, hucs.polygon_outlets):
        subdomain_conn = [list(m2.conn[tri]) for tri in tris]
        subdomain_nodes = set([c for e in subdomain_conn for c in e])
        subdomain_coords = np.array([m2.coords[c] for c in subdomain_nodes])
        m2h = Mesh2D(subdomain_coords, subdomain_conn, check_handedness=False, validate=False)

        edges = [(int(e[0]), int(e[1])) for e in m2h.boundary_edges]
        ls = LabeledSet(label+' boundary', m2.next_available_labeled_setid(),
                       'FACE', edges)
        ls.to_extrude = True # this marker tells the extrusion routine
                             # to not limit it to the surface
        m2.labeled_sets.append(ls)

        # every polygon now has an outlet -- find the boundary faces near that outlet
        outlet_faces = [e for e in m2h.boundary_edges if inside_ball(outlet, e)]
        edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
        ls = LabeledSet(label+' outlet', m2.next_available_labeled_setid(),
                        'FACE', edges)
        ls.to_extrude = True # this marker tells the extrusion routine
                             # to not limit it to the surface
        m2.labeled_sets.append(ls)

    # also write one for the full domain
    if hucs.exterior_outlet is not None:
        outlet_faces = [e for e in m2.boundary_edges if inside_ball(hucs.exterior_outlet, e)]
        edges = [(int(e[0]), int(e[1])) for e in outlet_faces]
        ls2 = LabeledSet('surface domain outlet', m2.next_available_labeled_setid(),
                         'FACE', edges)
        ls2.to_extrude = True
        m2.labeled_sets.append(ls2)
