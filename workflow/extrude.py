"""Extrudes a 2D mesh to generate an ExodusII 3D mesh.

Works with and assumes all polyhedra cells (and polygon faces).

Requires building a reasonably recent version of Exodus to get 
the associated exodus.py wrappers.

Note that this is typically done in your standard ATS installation,
assuming you have built your Amanzi TPLs with shared libraries (the
default through bootstrap).

In that case, simply ensure that ${AMANZI_TPLS_DIR}/SEACAS/lib is in
your PYTHONPATH.
"""

import numpy as np
import collections
import logging
import exodus

def _list_or_array(obj):
    return type(obj) == list or type(obj) == np.ndarray
    

class SideSet(object):
    """A collection of faces in elements."""
    def __init__(self, name, setid, elem_list, side_list):
        assert(type(setid) == int)
        assert(_list_or_array(side_list))
        assert(_list_or_array(elem_list))

        self.name = name
        self.setid = setid
        self.elem_list = elem_list
        self.side_list = side_list

class LabeledSet(object):
    """A generic collection of entities."""
    def __init__(self, name, setid, entity, ent_ids):
        assert entity in ['CELL', 'FACE', 'NODE']
        assert(type(setid) == int)
        assert(_list_or_array(ent_ids))

        self.name = name
        self.setid = setid
        self.entity = entity
        self.ent_ids = np.array(ent_ids)

class Mesh2D(object):
    """A surface mesh."""

    def __init__(self, coords, connectivity, labeled_sets=None, check_handedness=True):
        """Creates a 2D mesh from coordinates and a list cell-to-node connectivity lists.

        coords          | numpy array of shape (NCOORDS, NDIMS)
        connectivity    | list of lists of integer indices into coords specifying a
                        | (clockwise OR counterclockwise) ordering of the nodes around
                        | the 2D cell
        labeled_sets    | list of LabeledSet objects

        Note: coords, connectivity is the output provided by a
        workflow.triangulation.triangulate() call.
        """
        assert type(coords) == np.ndarray
        assert len(coords.shape) == 2

        self.dim = coords.shape[1]
        assert self.dim == 2 or self.dim == 3

        self.coords = coords
        self.conn = connectivity
        if labeled_sets is not None:
            self.labeled_sets = labeled_sets
        else:
            self.labeled_sets = []

        self.validate()
        self.edge_counts()
        if check_handedness:
            self.check_handedness()

    def validate(self):
        """Checks the validity of the mesh, or throws an AssertionError."""
        assert self.coords.shape[1] == 2 or self.coords.shape[1] == 3
        assert(_list_or_array(self.conn))
        for f in self.conn:
            assert(_list_or_array(f))
            assert len(set(f)) == len(f)
            for i in f:
                assert i < self.coords.shape[0]

        for ls in self.labeled_sets:
            if ls.entity == "NODE":
                size = len(self.coords)
            elif ls.entity == "CELL":
                size = len(self.conn)

            for i in ls.ent_ids:
                assert i < size
        return True

    def num_cells(self):
        return len(self.conn)

    def num_nodes(self):
        return self.coords.shape[0]

    def num_edges(self):
        return len(self.edges())

    @staticmethod
    def edge_hash(i,j):
        """Hashes edges in a direction-independent way."""
        return tuple(sorted((i,j)))
    
    def edges(self):
        return self.edge_counts().keys()

    def edge_counts(self):
        try:
            return self._edges
        except AttributeError:
            self._edges = collections.Counter(self.edge_hash(f[i], f[(i+1)%len(f)]) for f in self.conn for i in range(len(f)))
        return self._edges

    def boundary_edges(self):
        """Return edges in the boundary of the mesh, ordered around the boundary."""
        be = sorted([k for (k,count) in self.edge_counts().items() if count == 1], key=lambda a : a[0])
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
                new_e = list(reversed(new_e))

            be_ordered.append(new_e)
            done = len(be) == 0
        assert(be_ordered[-1][-1] == be_ordered[0][0])
        return be_ordered

    def boundary_nodes(self):
        return [e[0] for e in self.boundary_edges()]
        
        
    
    def check_handedness(self):
        """Ensures all cells are oriented via the right-hand-rule, i.e. in the +z direction."""
        for conn in self.conn:
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

    def centroids(self):
        """Calculate surface mesh centroids."""
        result = np.zeros((self.num_cells(),3),'d')
        for c, conn in enumerate(self.conn):
            points = np.array([self.coords[c] for c in conn])
            result[c,:] = points.mean(axis=0)
        return result
    
    def plot(self, color=None, ax=None):
        """Plot the flattened 2D mesh."""
        if color is None:
            import colors
            cm = colors.cm_mapper(0,self.num_cells()-1)
            colors = [cm(i) for i in range(self.num_cells())]
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

        Stolen from meshio, https://github.com/nschloe/meshio/blob/master/meshio/vtk_io.py
        """
        from workflow_tpls import vtk_io
        with open(filename,'rb') as fid:
            data = vtk_io.read_buffer(fid)

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
    def from_Transect(cls, x, z, width=1):
        """Creates a 2D surface strip mesh from transect data"""
        # coordinates
        y = np.array([0,width])
        Xc, Yc = np.meshgrid(x, y)
        Xc = Xc.flatten()
        Yc = Yc.flatten()

        Zc = np.concatenate([z,z])

        # connectivity
        nsurf_cells = len(x)-1
        conn = []
        for i in range(nsurf_cells):
            conn.append([i, i+1, nsurf_cells + i + 2, nsurf_cells + i + 1])

        points = np.array([Xc, Yc, Zc])
        return cls(points.transpose(), conn)


class Mesh3D(object):
    """A 3D mesh object."""
    
    def __init__(self, coords, face_to_node_conn, elem_to_face_conn,
                 side_sets=None, labeled_sets=None, material_ids=None):
        """Creates a 3D mesh from coordinates and connectivity lists.

        coords            | numpy array of shape (NCOORDS, 3)
        face_to_node_conn | list of lists of integer indices into coords specifying an
                          | (clockwise OR counterclockwise) ordering of the nodes around
                          | the face
        elem_to_face_conn | list of lists of integer indices into face_to_node_conn
                          | specifying a list of faces that make up the elem
        """
        assert type(coords) == np.ndarray
        assert len(coords.shape) == 2
        assert coords.shape[1] == 3
            
        self.dim = coords.shape[1]

        self.coords = coords
        self.face_to_node_conn = face_to_node_conn
        self.elem_to_face_conn = elem_to_face_conn

        if labeled_sets is not None:
            self.labeled_sets = labeled_sets
        else:
            self.labeled_sets = []

        if side_sets is not None:
            self.side_sets = side_sets
        else:
            self.side_sets = []
            
        if material_ids is not None:
            self.material_id_list = collections.Counter(material_ids).keys()
            self.material_ids = material_ids
        else:
            self.material_id_list = [10000,]
            self.material_ids = [10000,]*len(self.elem_to_face_conn)

        self.validate()

        
    def validate(self):
        """Checks the validity of the mesh, or throws an AssertionError."""
        assert self.coords.shape[1] == 3
        assert type(self.face_to_node_conn) is list
        for f in self.face_to_node_conn:
            assert type(f) is list
            assert len(set(f)) == len(f)
            for i in f:
                assert i < self.coords.shape[0]

        assert type(self.elem_to_face_conn) is list
        for e in self.elem_to_face_conn:
            assert type(e) is list
            assert len(set(e)) == len(e)
            for i in e:
                assert i < len(self.face_to_node_conn)

        for ls in self.labeled_sets:
            if ls.entity == "NODE":
                size = self.num_nodes()
            if ls.entity == "FACE":
                size = self.num_faces()
            elif ls.entity == "CELL":
                size = self.num_cells()

            for i in ls.ent_ids:
                assert i < size

        for ss in self.side_sets:
            for j,i in zip(ss.elem_list, ss.side_list):
                assert j < self.num_cells()
                assert i < len(self.elem_to_face_conn[j])



    def num_cells(self):
        return len(self.elem_to_face_conn)

    def num_faces(self):
        return len(self.face_to_node_conn)

    def num_nodes(self):
        return self.coords.shape[0]


    def write_exodus(self, filename, face_block_mode="one block"):
        """Write the 3D mesh to ExodusII using arbitrary polyhedra spec"""

        # put cells in with blocks, which renumbers the cells, so we have to track sidesets.
        # Therefore we keep a map of old cell to new cell ordering
        #
        # also, though not required by the spec, paraview and visit
        # seem to crash if num_face_blocks != num_elem_blocks.  So
        # make face blocks here too, which requires renumbering the faces.

        # -- first pass, form all elem blocks and make the map from old-to-new
        new_to_old_elems = []
        elem_blks = []
        for i_m,m_id in enumerate(self.material_id_list):
            # split out elems of this material, save new_to_old map
            elems_tuple = [(i,c) for (i,c) in enumerate(self.elem_to_face_conn) if self.material_ids[i] == m_id]
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
            for i_m,m_id in enumerate(self.material_id_list):
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
            for i_m, m_id in enumerate(self.material_id_list):
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

        ep = exodus.ex_init_params(title=filename.encode('ascii'),
                                   num_dim=3,
                                   num_nodes=self.num_nodes(),
                                   num_face=num_faces,
                                   num_face_blk=len(face_blks),
                                   num_elem=num_elems,
                                   num_elem_blk=len(elem_blks),
                                   num_side_sets=len(self.side_sets))
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
        assert len(elem_blks) == len(self.material_id_list)
        for i_blk, (m_id, elem_blk) in enumerate(zip(self.material_id_list, elem_blks)):
            elems_raveled = [f for c in elem_blk for f in c]

            e.put_polyhedra_elem_blk(m_id, len(elem_blk), len(elems_raveled), 0)
            e.put_elem_blk_name(m_id, 'MATERIAL_ID_%d'%m_id)
            e.put_face_count_per_polyhedra(m_id, np.array([len(c) for c in elem_blk]))
            e.put_elem_face_conn(m_id, np.array(elems_raveled)+1)

        # add sidesets
        e.put_side_set_names([ss.name for ss in self.side_sets])
        for ss in self.side_sets:
            for elem in ss.elem_list:
                assert old_to_new_elems[elem][0] == elem
            new_elem_list = [old_to_new_elems[elem][1] for elem in ss.elem_list]                
            e.put_side_set_params(ss.setid, len(ss.elem_list), 0)
            e.put_side_set(ss.setid, np.array(new_elem_list)+1, np.array(ss.side_list)+1)

        # finish and close
        e.close()

    @staticmethod
    def summarize_extrusion(layer_types, layer_data, ncells_per_layer, mat_ids):
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
                mat_id = mat_ids[i][0]
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
        ncells_total = ncells_tall * mesh2D.num_cells()
        nfaces_total = (ncells_tall+1) * mesh2D.num_cells() + ncells_tall * mesh2D.num_edges()
        nnodes_total = (ncells_tall+1) * mesh2D.num_nodes()

        np_mat_ids = np.array(mat_ids, dtype=int)
        if np_mat_ids.size == np.size(np_mat_ids, 0):
            if np_mat_ids.size == 1:
                np_mat_ids = np.full((len(ncells_per_layer), mesh2D.num_cells()), mat_ids[0], dtype=int)
            else:
                np_mat_ids = np.empty((len(ncells_per_layer), mesh2D.num_cells()), dtype=int)
                for ilay in range(len(ncells_per_layer)):
                    np_mat_ids[ilay, :] = np.full(mesh2D.num_cells(), mat_ids[ilay], dtype=int)


        def col_to_id(column, z_cell):
            """Maps 2D cell ID and index in the vertical to a 3D cell ID"""
            return z_cell + column * ncells_tall

        def node_to_id(node, z_node):
            """Maps 2D node ID and index in the vertical to a 3D node ID"""
            return z_node + node * (ncells_tall+1)

        def edge_to_id(edge, z_cell):
            """Maps 2D edge hash and index in the vertical to a 3D face ID of a vertical face"""
            return (ncells_tall + 1) * mesh2D.num_cells() + z_cell + edge * ncells_tall

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
                    centroids = mesh2D.cell_centroids()
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
        for col in range(mesh2D.num_cells()):
            nodes_2 = mesh2D.conn[col]
            surface.append(col_to_id(col,0))
            for z_face in range(ncells_tall + 1):
                i_f = len(faces)
                f = [node_to_id(n, z_face) for n in nodes_2]

                if z_face != ncells_tall:
                    cells[col_to_id(col, z_face)].append(i_f)
                if z_face != 0:
                    cells[col_to_id(col, z_face-1)].append(i_f)

                faces.append(f)
            bottom.append(col_to_id(col,ncells_tall-1))

        # -- loop over the columns, adding the vertical faces
        added = dict()
        vertical_side_cells = []
        vertical_side_indices = []
        for col in range(mesh2D.num_cells()):
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
                        assert i_f == edge_to_id(i_e, z_face)
                        f = [node_to_id(edge[0], z_face),
                             node_to_id(edge[1], z_face),
                             node_to_id(edge[1], z_face+1),
                             node_to_id(edge[0], z_face+1)]
                        faces.append(f)
                        face_cell = col_to_id(col, z_face)
                        cells[face_cell].append(i_f)

                        # check if this is an external
                        if mesh2D._edges[edge] == 1:
                            vertical_side_cells.append(face_cell)
                            vertical_side_indices.append(len(cells[face_cell])-1)
                        
                else:
                    # faces already added from previous column
                    for z_face in range(ncells_tall):
                        i_f = edge_to_id(i_e, z_face)
                        cells[col_to_id(col, z_face)].append(i_f)


        # Do some idiot checking
        # -- check we got the expected number of faces
        assert len(faces) == nfaces_total
        # -- check every cell is at least a tet
        for c in cells:
            assert len(c) > 4
        # -- check surface sideset has the right number of entries
        assert len(surface) == mesh2D.num_cells()
        # -- check bottom sideset has the right number of entries
        assert len(bottom) == mesh2D.num_cells()

        # -- len of vertical sides sideset is number of external edges * number of cells, no pinchouts here
        num_sides = ncells_tall * sum(1 for e,c in mesh2D.edge_counts().items() if c == 1)
        assert num_sides == len(vertical_side_cells)
        assert num_sides == len(vertical_side_indices)

        # make the material ids
        material_ids = np.zeros((len(cells),),'i')
        for col in range(mesh2D.num_cells()):
            z_cell = 0
            for ilay in range(len(ncells_per_layer)):
                ncells = ncells_per_layer[ilay]
                for i in range(z_cell, z_cell+ncells):
                    material_ids[col_to_id(col, i)] = np_mat_ids[ilay, col]
                z_cell = z_cell + ncells

        # make the side sets
        side_sets = []
        side_sets.append(SideSet("bottom", 1, bottom, [1,]*len(bottom)))
        side_sets.append(SideSet("surface", 2, surface, [0,]*len(surface)))
        side_sets.append(SideSet("external_sides", 3, vertical_side_cells, vertical_side_indices))

        # reshape coords
        coords = coords.reshape(nnodes_total, 3)        
        
        for e,s in zip(side_sets[0].elem_list, side_sets[0].side_list):
            face = cells[e][s]
            fz_coords = np.array([coords[n] for n in faces[face]])

        for e,s in zip(side_sets[1].elem_list, side_sets[1].side_list):
            face = cells[e][s]
            fz_coords = np.array([coords[n] for n in faces[face]])
        
        # instantiate the mesh
        return cls(coords, faces, cells, side_sets=side_sets, material_ids=material_ids)



