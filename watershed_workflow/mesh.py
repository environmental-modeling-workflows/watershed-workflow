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
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any, Callable

import os
import numpy as np
import collections
import logging
import attr
import scipy.optimize
import shapely
import warnings
import functools
import pandas
import geopandas as gpd
from matplotlib import collections as mpc
import copy

import watershed_workflow.crs
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
def cache(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """A caching decorator for instance methods.

    This decorator caches the result of a method call in an instance attribute.
    Note: Only works with __dict__ classes, not slotted classes.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function to cache.

    Returns
    -------
    Callable[[Any], Any]
        The wrapped function with caching.
    """
    @functools.wraps(func)
    def cache_func(self):
        cname = '_' + func.__name__
        if not hasattr(self, cname):
            self.__setattr__(cname, func(self))
        return self.__getattribute__(cname)

    return cache_func


@functools.total_ordering
class Edge:
    """A pair of vertex indices forming a mesh edge.

    Edges are order-independent: ``Edge(i, j) == Edge(j, i)``.

    Parameters
    ----------
    *args : int or Tuple[int, int]
        Either two ints ``Edge(i, j)`` or a single tuple ``Edge((i, j))``.
    """

    def __init__(self, *args):
        if len(args) == 1:
            i, j = args[0]
        elif len(args) == 2:
            i, j = args
        else:
            raise ValueError(f"Edge accepts 1 tuple or 2 ints, got {len(args)} arguments")
        self._nodes: Tuple[int, int] = (min(i, j), max(i, j))

    def __repr__(self) -> str:
        return f"Edge({self._nodes[0]}, {self._nodes[1]})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self._nodes == other._nodes

    def __hash__(self) -> int:
        return hash(self._nodes)

    def __lt__(self, other) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self._nodes < other._nodes

    def __contains__(self, v: int) -> bool:
        return v in self._nodes

    def __getitem__(self, i: int) -> int:
        if i not in (0, 1):
            raise IndexError(f"Edge index must be 0 or 1, got {i}")
        return self._nodes[i]

    def __iter__(self):
        yield from self._nodes

    def __len__(self) -> int:
        return 2


@attr.define
class SideSet:
    """A collection of faces in cells."""
    name: str
    setid: int
    cell_list: list[int]
    side_list: list[int]

    def validate(self, cell_faces: List[List[int]]) -> None:
        """Validate the side set against cell faces.

        Parameters
        ----------
        cell_faces : List[List[int]]
            List of cell face connections.

        Raises
        ------
        AssertionError
            If validation fails.
        """
        ncells = len(cell_faces)
        for c, s in zip(self.cell_list, self.side_list):
            assert (0 <= c < ncells)
            assert (0 <= s < len(cell_faces[c]))


@attr.define(slots=False)
class LabeledSet:
    """A generic collection of entities."""
    name: str
    setid: int
    entity: str
    ent_ids: List[Any]
    to_extrude: bool = False

    def validate(self, size: int, is_tuple: bool = False) -> None:
        """Validate the labeled set.

        Parameters
        ----------
        size : int
            Expected size for validation.
        is_tuple : bool, optional
            Whether entities are tuples (edges). Default is False.

        Raises
        ------
        AssertionError
            If validation fails.
        """
        if is_tuple:
            # edges
            self.ent_ids = [(int(e[0]), int(e[1])) for e in self.ent_ids]
            for e in self.ent_ids:
                assert (-1 < e[0] < size)
                assert (-1 < e[1] < size)
        else:
            self.ent_ids = [int(i) for i in self.ent_ids]
            for i in self.ent_ids:
                assert (-1 < i < size)


@attr.define
class _ExtrusionHelper:
    """Helper class for extruding 2D --> 3D"""
    ncells_tall: int
    ncells_2D: int

    def col_to_id(self, column: int, z_cell: int) -> int:
        """Maps 2D cell ID and index in the vertical to a 3D cell ID.

        Parameters
        ----------
        column : int
            2D cell ID.
        z_cell : int
            Index in the vertical direction.

        Returns
        -------
        int
            3D cell ID.
        """
        return z_cell + column * self.ncells_tall

    def vertex_to_id(self, vertex: int, z_vertex: int) -> int:
        """Maps 2D vertex ID and index in the vertical to a 3D vertex ID.

        Parameters
        ----------
        vertex : int
            2D vertex ID.
        z_vertex : int
            Index in the vertical direction.

        Returns
        -------
        int
            3D vertex ID.
        """
        return z_vertex + vertex * (self.ncells_tall + 1)

    def edge_to_id(self, edge: int, z_cell: int) -> int:
        """Maps 2D edge hash and index in the vertical to a 3D face ID of a vertical face.

        Parameters
        ----------
        edge : int
            2D edge hash.
        z_cell : int
            Index in the vertical direction.

        Returns
        -------
        int
            3D face ID.
        """
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
    cell_data : Optional[pandas.DataFrame]
        A DataFrame of length len(conn) including assorted cell-based
        data.
    eps : float, optional=0.01
        A small measure of length between coords.
    check_handedness : bool, optional=True
        If true, ensure all cells are oriented so that right-hand rule
        ordering of the vertices points up.
    validate : bool, optional=False
        If true, validate coordinates and connections
        post-construction.

    Note: (coords, conn) may be output provided by a
    watershed_workflow.triangulation.triangulate() call.

    """
    coords = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    _conn = attr.ib()
    labeled_sets: List[LabeledSet] = attr.ib(factory=list)
    crs: Optional[watershed_workflow.crs.CRS] = attr.ib(default=None)
    _cell_data: Optional[pandas.DataFrame] = attr.ib(default=None)
    eps: float = attr.ib(default=0.001)
    _check_handedness: bool = attr.ib(default=True)
    _validate: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        if self._check_handedness:
            self.checkHandedness()
        if self._validate:
            self.validate()

    @property
    def conn(self):
        """Note that conn is immutable because changing this breaks all the
        other properties, which may be cached.  To change topology, one must construct a new mesh!"""
        return self._conn

    @property
    def dim(self) -> int:
        """Spatial dimension of the mesh.

        Returns
        -------
        int
            Number of spatial dimensions.
        """
        return self.coords.shape[1]

    @property
    def num_cells(self) -> int:
        """Number of cells in the mesh.

        Returns
        -------
        int
            Number of cells.
        """
        return len(self.conn)

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh.

        Returns
        -------
        int
            Number of vertices.
        """
        return self.coords.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of edges in the mesh.

        Returns
        -------
        int
            Number of edges.
        """
        return len(self.edges)

    @property
    def edges(self):
        return self.edge_cells.keys()

    @property
    @cache
    def edge_cells(self):
        """A map from edge to lists of cells that edge touches."""
        e2c = collections.defaultdict(list)
        for i in range(self.num_cells):
            for e in self.cell_edges[i]:
                e2c[e].append(i)
        return e2c

    @property
    @cache
    def cell_edges(self):
        """A map from cell to list of edges."""
        def ce(c):
            conn = self.conn[c]
            return [Edge(conn[i], conn[(i+1)%len(conn)]) for i in range(len(conn))]
        return { c : ce(c) for c in range(self.num_cells) }


    @property
    @cache
    def cell_to_cells(self):
        """A list of length ncells, each entry is a list of neighboring cells."""
        c2c = [list() for c in range(len(self.conn))]
        for e, clist in self.edge_cells.items():
            if len(clist) == 2:
                c2c[clist[0]].append(clist[1])
                c2c[clist[1]].append(clist[0])
        return c2c

    @property
    @cache
    def boundary_edges(self):
        """Return edges in the boundary of the mesh, ordered around the boundary."""
        be = sorted([e for (e, cells) in self.edge_cells.items() if len(cells) == 1])
        be_ordered = [be.pop(0)]
        # tip is the vertex at the forward end of the last ordered edge.
        # Initialise to whichever end is shared with fewer remaining edges,
        # so we walk consistently; a simple heuristic is to start at edge[1].
        tip = be_ordered[0][1]

        while len(be) > 0:
            try:
                next_i = next(i for (i, e) in enumerate(be) if tip in e)
            except StopIteration:
                raise RuntimeError("Invalid set of boundary edges on mesh -- is the mesh domain simply connected?")

            next_e = be.pop(next_i)
            be_ordered.append(next_e)
            # advance tip to the other vertex of the newly appended edge
            tip = next_e[0] if next_e[1] == tip else next_e[1]

        return be_ordered

    @property
    @cache
    def boundary_vertices(self):
        return list(set(v for e in self.boundary_edges for v in e))

    @property
    def cell_data(self):
        if self._cell_data is None:
            self._cell_data = pandas.DataFrame(index=range(len(self.conn)))
        return self._cell_data

    def checkHandedness(self, conn=None) -> List[int] | None:
        """Ensures all cells are oriented via the right-hand-rule, i.e. in the +z direction."""
        if conn is None:
            self._conn = [self.checkHandedness(conn) for conn in self.conn]

        else:
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
            return conn

    def validate(self) -> None:
        # validate coords
        assert (isinstance(self.coords, np.ndarray))
        assert (len(self.coords.shape) == 2)
        assert (self.coords.shape[1] in [2, 3])

        # validate conn
        assert (min(c for conn in self.conn for c in conn) >= 0)
        assert (max(c for conn in self.conn for c in conn) < self.coords.shape[0])

        # validate labeled_sets
        assert (len(set(ls.setid for ls in self.labeled_sets)) == len(self.labeled_sets))

        for ls in self.labeled_sets:
            is_tuple = False
            if ls.entity == 'CELL':
                size = self.num_cells
            elif ls.entity == 'FACE':
                size = self.num_vertices  # note there are no faces, edges are tuples of vertices
                is_tuple = True
            elif ls.entity == 'VERTEX':
                size = self.num_vertices
            ls.validate(size, is_tuple)

    def getNextAvailableLabeledSetID(self) -> int:
        """Returns next available LS id."""
        i = 10000
        while any(i == ls.setid for ls in self.labeled_sets):
            i += 1
        return int(i)

    def computeCentroid(self, c) -> np.ndarray:
        """Computes, based on coords, the centroid of a cell with ID c.

        Note this ALWAYS recomputes the value, not using the cache.
        """
        return watershed_workflow.utils.computeCentroid([self.coords[v]
                                                         for v in self.conn[c]])

    @property
    @cache
    def centroids(self):
        """Cell centroids."""
        return np.array([self.computeCentroid(c) for c in range(self.num_cells)])


    @property
    @cache
    def edge_centroids(self):
        """Edge centroids."""
        return {e: watershed_workflow.utils.computeMidpoint(self.coords[e[0]],
                                                           self.coords[e[1]])
                for e in self.edges}

    
    def clearGeometryCache(self) -> None:
        """If coordinates are changed, any computed, cached geometry must be
        recomputed.  It is the USER's responsibility to call this
        function if any coords are changed!
        """
        # toss geometry cache
        if hasattr(self, '_centroids'):
            del self._centroids
        if hasattr(self, '_edge_centroids'):
            del self._edge_centroids
            

    def to_dataframe(self, include_labeled_sets : bool = False) -> gpd.GeoDataFrame:
        """Convert the mesh to a GeoDataFrame with each cell as a row."""
        vert_sets = [[self.coords[i, 0:2] for i in conn] for conn in self.conn]
        polygons = [shapely.geometry.Polygon(verts) for verts in vert_sets]

        gdf = gpd.GeoDataFrame(self.cell_data, geometry=polygons, crs=self.crs)

        if include_labeled_sets:
            for ls in self.labeled_sets:
                if ls.entity == 'CELL':
                    df[f'{ls.name} : {ls.setid}'] = gdf.index.isin(ls.ent_ids)

        return df


    def plot(self,
             facecolors=None,
             ax=None,
             cmap=None,
             vmin=None,
             vmax=None,
             norm=None,
             colorbar=True,
             **kwargs) -> mpc.PolyCollection:

        """Plot the flattened 2D mesh."""
        from matplotlib import pyplot as plt

        # set default plotting options for mesh
        kwargs.setdefault('edgecolors', 'grey')
        kwargs.setdefault('linewidth', 0.5)
        kwargs.setdefault('linewidth', 0.5)

        if isinstance(facecolors, str) and facecolors == 'elevation':
            facecolors = self.centroids[:, -1]
            if cmap is None:
                cmap = 'terrain'

        # if facecolors is not None and len(facecolors) == len(self.conn):
        #     # convert from an array to a list of colors, using a cmap
        #     array = facecolors
        #     if norm is None:
        #         norm = plt.Normalize(vmin=vmin, vmax=vmax)
        #     if isinstance(cmap, str):
        #         cmap = plt.colormaps[cmap]

        #     facecolors = cmap(norm(facecolors))
        if norm is None:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if isinstance(cmap, str):
            cmap = plt.colormaps[cmap]

        # build the collection of gons
        verts = [[self.coords[i, 0:2] for i in f] for f in self.conn]
        gons = mpc.PolyCollection(verts, array=facecolors, cmap=cmap, norm=norm, **kwargs)

        # put on the axis
        if ax is None:
            fig, ax = plt.subplots()
        ax.add_collection(gons)
        ax.autoscale_view()

        if colorbar:
            gons.set(array=facecolors, cmap=cmap)
            plt.colorbar(gons, ax=ax)

        return gons

    def plotVertices(self,
                     vertex_values,
                     ax=None,
                     cmap=None,
                     vmin=None,
                     vmax=None,
                     norm=None,
                     s=None,
                     colorbar=True,
                     label=None,
                     **kwargs):
        """Plot vertex-based data as a scatter plot.

        Parameters
        ----------
        vertex_values : array-like
            Values at each vertex (length must equal num_vertices)
        ax : matplotlib.Axes, optional
            Axes to plot on. If None, creates new figure/axes.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for mapping values to colors. Default is 'viridis'.
        vmin, vmax : float, optional
            Min and max values for color mapping. If None, uses data range.
        norm : matplotlib.colors.Normalize, optional
            Normalization for color mapping. If None, uses linear normalization.
        s : float, optional
            Marker size. If None, automatically computed based on mesh size.
        colorbar : bool, optional
            If True, adds colorbar. Default is True.
        label : str, optional
            Label for colorbar.
        **kwargs : dict
            Additional keyword arguments passed to scatter()

        Returns
        -------
        PathCollection
            The scatter plot collection
        """
        from matplotlib import pyplot as plt

        # Validate input
        if len(vertex_values) != self.num_vertices:
            raise ValueError(f"vertex_values length ({len(vertex_values)}) must match "
                           f"num_vertices ({self.num_vertices})")

        # Set defaults
        if cmap is None:
            cmap = 'viridis'
        if norm is None:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if isinstance(cmap, str):
            cmap = plt.colormaps[cmap]

        # Determine marker size based on mesh size
        # Make markers bigger for small meshes
        if s is None:
            s = max(10, min(100, 5000 / self.num_vertices))

        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots()

        # Create scatter plot
        scatter = ax.scatter(self.coords[:, 0], self.coords[:, 1],
                            c=vertex_values, cmap=cmap, norm=norm, s=s, **kwargs)

        # ax.set_aspect('equal')
        # ax.autoscale_view()

        # Add colorbar if requested
        if colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            if label:
                cbar.set_label(label)

        return scatter


    def partition(self, nparts : int, reorder : bool = True) -> Mesh2D:
        """Partitions the mesh, adding a cell_data column for partition number.

        Parameters
        ----------
        nparts : int
          Number of parts to partition cells into.
        reorder : bool, optional
          If True, also creates a new mesh in the partition ordering.
          Default is True.

        Returns
        -------
        Mesh2D
          If reorder, the partitioned and reordered mesh; otherwise
          returns self, which includes partition info in cell_data.

        """
        import pymetis

        # convert to adjacency arrays, based on cells
        xadj = [0]
        adjncy = []
        for c2c in self.cell_to_cells:
            xadj.append(xadj[-1] + len(c2c))
            adjncy.extend(c2c)

        adj = pymetis.CSRAdjacency(xadj, adjncy)
        ncuts, parts = pymetis.part_graph(nparts, adjacency=adj)
        self.cell_data['partition'] = parts

        if reorder:
            new_order = sorted(range(len(self.conn)), key=lambda c : parts[c])
            return self.reorder(new_order)
        else:
            return self


    def reorder(self, new_cell_order) -> Mesh2D:
        """Creates a new mesh with reordered cells.

        Parameters
        ----------
        new_order : list[int]
          A list of length num_cells, indicating the new ordering of
          existing cell indices.  The new mesh's cell 0 will be self's
          new_cell_order[0] cell.

        Returns
        -------
        Mesh2D
          The reordered mesh.

        Note this also tries to reorder the vertex coordinates in a
        sane way, by reiterating over cell vertices and renumbering.
        Edges are naturally reordered in the new mesh.

        """
        # check the input -- bad input would result in confusing errors
        if len(new_cell_order) != self.num_cells:
            raise ValueError('New order should be a list of length num_cells')
        if len(set(new_cell_order)) != self.num_cells:
            raise ValueError('New order should be a unique list the cells')
        if max(new_cell_order) != self.num_cells - 1:
            raise ValueError('New order include integers ranging from [0, num_cells)')
        if min(new_cell_order) != 0:
            raise ValueError('New order include integers ranging from [0, num_cells)')

        # compute the new connections, reordering vertices
        new_conns = []

        forward_cell_map = dict()
        forward_vertex_map = collections.defaultdict(lambda : len(forward_vertex_map))
        for i_new, i_old in enumerate(new_cell_order):
            forward_cell_map[i_old] = i_new
            conn = self.conn[i_old]
            new_conns.append([forward_vertex_map[v] for v in conn])

        # reorder the coordinates according to reordered vertices
        new_coords = -np.ones_like(self.coords)
        for v,vn in forward_vertex_map.items():
            new_coords[vn] = self.coords[v]

        # create new labeled sets
        new_labeled_sets = []
        for ls in self.labeled_sets:
            if ls.entity == 'CELL':
                new_ids = list(sorted(forward_cell_map[c] for c in ls.ent_ids))
            elif ls.entity == 'FACE':
                new_ids = list(sorted(
                    Edge(forward_vertex_map[e[0]], forward_vertex_map[e[1]])
                    for e in ls.ent_ids))
            elif ls.entity == 'VERTEX':
                new_ids = list(sorted(forward_vertex_map[v] for v in ls.ent_ids))
            new_labeled_sets.append(LabeledSet(ls.name, ls.setid, ls.entity, new_ids, ls.to_extrude))

        # permute any cell_data
        if self.cell_data is not None:
            new_cell_data = self.cell_data.iloc[new_cell_order].reset_index(drop=True)
        else:
            new_cell_data = None

        return self.__class__(new_coords, new_conns, new_labeled_sets, self.crs,
                              new_cell_data, self.eps, True, True)


    def transform(self,
                  mat: Optional[np.ndarray] = None,
                  shift: Optional[np.ndarray] = None) -> None:

        """Transform a 2D mesh.

        Parameters
        ----------
        mat : np.ndarray, optional
            3x3 transformation matrix. Default is identity.
        shift : np.ndarray, optional
            3-element translation vector. Default is zero.
        """
        if mat is None:
            mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if shift is None:
            shift = np.array([0, 0, 0])

        new_coords = []
        for c in self.coords:
            assert (c.shape == (3, ))
            tc = mat@c + shift
            assert (tc.shape == (3, ))
            new_coords.append(tc)
        new_coords_array = np.array(new_coords)
        assert (new_coords_array.shape == self.coords.shape)
        self.coords = new_coords_array

        # toss geometry cache
        self.clearGeometryCache()

    def writeVTK(self, filename: str) -> None:
        """Writes to VTK.

        Parameters
        ----------
        filename : str
            Output VTK filename.
        """
        import watershed_workflow.vtk_io
        assert (all(len(c) == 3 for c in self.conn))
        watershed_workflow.vtk_io.write(filename, self.coords, { 'triangle': np.array(self.conn) })

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
        with open(filename, 'rb') as fid:
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
                    points = np.fromfile(fid, count=ncoords * 3, sep=' ', dtype='d')
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
                        gon = list(data[idx + 1:idx + 1 + n_in_gon])

                        # check handedness -- need normals to point up!
                        cross = []
                        for i in range(len(gon)):
                            if i == len(gon) - 1:
                                ip = 0
                                ipp = 1
                            elif i == len(gon) - 2:
                                ip = i + 1
                                ipp = 0
                            else:
                                ip = i + 1
                                ipp = i + 2
                            d2 = points[gon[ipp]] - points[gon[ip]]
                            d1 = points[gon[i]] - points[gon[ip]]
                            cross.append(np.cross(d2, d1))
                        if (np.array([c[2] for c in cross]).mean() < 0):
                            gon.reverse()

                        gons.append(gon)

                        idx += n_in_gon + 1
                    assert (idx == n_to_read)
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
        import watershed_workflow.vtk_io
        with open(filename, 'rb') as fid:
            data = watershed_workflow.vtk_io.read_buffer(fid)

        points = data[0]
        if len(data[1]) != 1:
            raise RuntimeError(
                "Simplex VTK file is readable by vtk_io but not by meshing_ats.  Includes: %r"
                % data[1].keys())

        gons = next(v for v in data[1].values())
        gons = gons.tolist()

        # check handedness
        for gon in gons:
            cross = []
            for i in range(len(gon)):
                if i == len(gon) - 1:
                    ip = 0
                    ipp = 1
                elif i == len(gon) - 2:
                    ip = i + 1
                    ipp = 0
                else:
                    ip = i + 1
                    ipp = i + 2
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
            y = np.array([0, 1])
        else:
            variable_width = False
            y = np.array([-width / 2, width / 2])

        Xc, Yc = np.meshgrid(x, y)
        if variable_width:
            assert (Yc.shape[0] == 2)
            assert (len(width) == Yc.shape[1])
            assert (min(width) > 0.)
            Yc[0, :] = -width / 2.
            Yc[1, :] = width / 2.

        Xc = Xc.flatten()
        Yc = Yc.flatten()
        Zc = np.concatenate([z, z])

        # connectivity
        nsurf_cells = len(x) - 1
        conn = []
        for i in range(nsurf_cells):
            conn.append([i, i + 1, nsurf_cells + i + 2, nsurf_cells + i + 1])

        points = np.array([Xc, Yc, Zc])
        return cls(points.transpose(), conn, **kwargs)

    def to_dual(self):
        """Creates a 2D surface mesh from a primal 2D triangular surface mesh.

        Returns
        -------
        dual_vertices : np.ndarray
        dual_conn : list(lists)
            The vertices and cell_to_vertex_conn of a 2D, polygonal, nearly Voronoi
            mesh that is the truncated dual of self.  Here we say nearly because
            the boundary triangles (see note below) are not Voronoi.
        dual_from_primal_mapping : np.array( (len(dual_conn),), 'i')
            Mapping from a dual cell to the primal vertex it is based on.
            Without the boundary, this would be simply
            arange(0,len(dual_conn)), but beacuse we added triangles on the
            boundary, this mapping is useful.

        Note
        ----
        - Vertices of the dual are on circumcenters of the primal triangles.
        - Centers of the dual are numbered by vertices of the primal mesh, modulo
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
            assert (len(c) == 3)  # check all triangles

        def circumcenter(p1, p2, p3):
            d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

            xv = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) *
                  (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
            yv = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) *
                  (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
            return (xv, yv)

        # dual vertices are given by:
        # - primal cell circumcenters
        # - primal vertices on the boundary
        # - primal edge midpoints on the boundary (note this length is the same as primal vertices on the boundary)

        # dual cells are given by:
        # - primal cell vertices (modulo the boundary)

        coords = self.coords[:, 0:2]
        logging.info("-- computing primary boundary edges")
        boundary_edges = self.boundary_edges
        n_dual_vertices = len(self.conn) + 2 * len(boundary_edges)
        logging.info("     n_primal_cell = {}, n_boundary_edges = {}, n_dual_vertices = "
                     "{}".format(len(self.conn), len(boundary_edges), n_dual_vertices))

        # create space to populate dual coordinates
        dual_vertices = np.zeros((n_dual_vertices, 2), 'd')
        dual_from_primal_mapping = list(range(len(coords)))

        # Create a list of lists for dual cells -- at this point the truncated
        # dual (e.g. allow non-convex boundary polygons), so one per primal
        # vertex.  Also create a flag array for whether a given dual cell is on the
        # boundary and so might be non-convex.
        #
        # Note that both of these will be indexed by the index of the
        # corresponding primal vertex.
        dual_cells = [list() for i in range(len(coords))]
        is_boundary = np.zeros(len(dual_cells), 'i')

        # Loop over all primal cells (triangles), adding the circumcenter
        # as a dual vertex and sticking that vertex in three dual cells rooted
        # at the three primal vertices.
        logging.info("-- computing dual vertices")
        i_dual_vertex = 0
        for j, c in enumerate(self.conn):
            dual_vertices[i_dual_vertex][:] = circumcenter(coords[c[0]], coords[c[1]], coords[c[2]])
            dual_cells[c[0]].append(i_dual_vertex)
            dual_cells[c[1]].append(i_dual_vertex)
            dual_cells[c[2]].append(i_dual_vertex)
            i_dual_vertex += 1

        logging.info("    added {} tri centroid vertices".format(i_dual_vertex))

        # Loop over the boundary, adding both the primal vertices and the edge
        # midpoints as dual vertices.
        #
        # Add the primal vertex and two midpoints on either side to the list
        # of dual vertices in the cell "rooted at" the primal vertex.
        for i, e in enumerate(boundary_edges):
            # add the primal vertex always
            my_primal_vertex_dual_vertex = i_dual_vertex
            dual_vertices[i_dual_vertex][:] = coords[e[0]]
            i_dual_vertex += 1

            my_cell = list()

            # stick in the previous midpoint vertex, add to my_cell
            if i == 0:
                # reserve a spot for the last midpoint
                first_cell_added = e[0]
                my_cell.append(-1)
            else:
                my_cell.append(prev_midp_n)

            # stick in the next midpoint vertex, add to my_cell
            next_midp_n = i_dual_vertex
            next_midp = (coords[e[0]][:] + coords[e[1]][:]) / 2.
            dual_vertices[i_dual_vertex][:] = next_midp
            i_dual_vertex += 1
            my_cell.append(next_midp_n)

            # add the primal vertex to my_cell
            my_cell.append(my_primal_vertex_dual_vertex)
            dual_cells[e[0]].extend(my_cell)

            is_boundary[e[0]] = 1
            prev_midp_n = next_midp_n

        # patch up the first cell added
        assert (dual_cells[first_cell_added][-3] == -1)
        dual_cells[first_cell_added][-3] = prev_midp_n

        logging.info("    added {} boundary vertices".format(len(boundary_edges)))

        #
        # Now every dual cell has a list of vertices (in no particular order).
        # But some of these vertices may be duplicated -- either the circumcenter
        # of two adjacent triangles may both be on the boundary, allowing for
        # coincident points at the midpoint of the coincident faces, or (more
        # likely) there was a circumcenter on the boundary, meaning it is
        # coincident with the boundary edge's midpoint.
        #
        # Order those vertices, and collect a list of coincident vertices for removal.
        logging.info("-- Finding duplicates and ordering conn_cell_to_vertex")
        vertices_to_kill = dict(
        )  # coincident vertices (key = vertex to remove, val = coincident vertex)
        for i in range(len(dual_cells)):
            c = dual_cells[i]

            if is_boundary[i]:
                # check for duplicate vertices
                to_pop = []
                for k in range(len(c)):
                    for j in range(k + 1, len(c)):
                        if (np.linalg.norm(dual_vertices[c[k]] - dual_vertices[c[j]]) < self.eps):
                            logging.info("    found dup on boundary cell {} = {}".format(i, c))
                            logging.info("        indices = {}, {}".format(k, j))
                            logging.info("        vertices = {}, {}".format(c[k], c[j]))
                            logging.info("        coords = {}, {}".format(
                                dual_vertices[c[k]], dual_vertices[c[j]]))

                            if c[k] in vertices_to_kill:
                                assert c[j] == vertices_to_kill[c[k]]
                                assert k < len(c) - 3
                                to_pop.append(k)
                            elif c[j] in vertices_to_kill:
                                assert c[k] == vertices_to_kill[c[j]]
                                assert j < len(c) - 3
                                to_pop.append(j)
                            else:
                                assert k < len(c) - 3
                                vertices_to_kill[c[k]] = c[j]
                                to_pop.append(k)

                # remove the duplicated vertices from the cell_to_vertex_conn
                for j in reversed(sorted(to_pop)):
                    c.pop(j)

                # may not be convex -- triangulate
                c_orig = c[:]
                c0 = c[-1]  # the primal vertex
                cup = c[-2]  # boundary midpoint, one direction
                cdn = c[-3]  # boundary midpoint, the other direction

                # order the vertices (minus the primal vertex) clockwise around the primal vertex
                cell_coords = np.array([dual_vertices[cj] for cj in c[:-1]]) - dual_vertices[c0]
                angle = np.array([
                    np.arctan2(cell_coords[j, 1], cell_coords[j, 0])
                    for j in range(len(cell_coords))
                ])
                order = np.argsort(angle)
                c = [c[j] for j in order]

                # now find what is "inside" and what is "outside" the domain by
                # finding up and dn, making one the 0th point and the other the
                # last point
                up_i = c.index(cup)
                dn_i = c.index(cdn)

                if dn_i == (up_i+1) % len(c):
                    cn = c[dn_i:] + c[0:dn_i]
                elif up_i == (dn_i+1) % len(c):
                    cn = c[up_i:] + c[0:up_i]
                else:
                    # I screwed up... debug me!
                    logging.info("Uh oh bad geom: up_i = {}, dn_i = {}, c = {}".format(
                        up_i, dn_i, c))
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    cc_sorted = np.array([cell_coords[k] for k in order]) + dual_vertices[c0]
                    cb_sorted = np.array([dual_vertices[cdn], dual_vertices[c], dual_vertices[cup]])
                    ax.plot(cc_sorted[:, 0], cc_sorted[:, 1], 'k-x')
                    ax.scatter(dual_vertices[c0, 0], dual_vertices[c0, 1], color='r')
                    ax.scatter(dual_vertices[cup, 0], dual_vertices[cup, 1], color='m')
                    ax.scatter(dual_vertices[cdn, 0], dual_vertices[cdn, 1], color='b')
                    plt.show()

                    fig = plt.figure(figsize=figsize)
                    ax = watershed_workflow.plot.get_ax(crs, fig)

                    mp = watershed_workflow.plot.triangulation(mesh_points3,
                                                               mesh_tris,
                                                               crs,
                                                               ax=ax,
                                                               color='elevation',
                                                               edgecolor='white',
                                                               linewidth=0.4)
                    cbar = fig.colorbar(mp, orientation="horizontal", pad=0.05)
                    #watershed_workflow.plot.hucs(shapes, crs, ax=ax, color='k', linewidth=1)
                    watershed_workflow.plot.shply([shapely.geometry.LineString(cc_sorted), ],
                                                  crs,
                                                  ax=ax,
                                                  color='red',
                                                  linewidth=1)
                    watershed_workflow.plot.shply([shapely.geometry.LineString(cb_sorted), ],
                                                  crs,
                                                  ax=ax,
                                                  color='blue',
                                                  linewidth=1)
                    ax.set_aspect('equal', 'datalim')

                    raise RuntimeError('uh oh borked geom')

                # triangulate the truncated dual polygon, always including the
                # primal vertex to guarantee all triangles exist and partition
                # the polygon.
                for k in range(len(cn) - 1):
                    if k == 0:
                        dual_cells[i] = [c0, cn[k + 1], cn[k]]
                    else:
                        dual_cells.append([c0, cn[k + 1], cn[k]])
                        dual_from_primal_mapping.append(i)

            else:
                # NOT a boundary polygon.  Simply order and check for duplicate vertices.
                to_pop = []
                for k in range(len(c)):
                    for j in range(k + 1, len(c)):
                        if (np.linalg.norm(dual_vertices[c[k]] - dual_vertices[c[j]]) < self.eps):
                            logging.info("    found dup on interior cell {} = {}".format(i, c))
                            logging.info("        indices = {}, {}".format(k, j))
                            logging.info("        vertices = {}, {}".format(c[k], c[j]))
                            logging.info("        coords = {}, {}".format(
                                dual_vertices[c[k]], dual_vertices[c[j]]))

                            if c[k] in vertices_to_kill:
                                assert c[j] == vertices_to_kill[c[k]]
                                to_pop.append(k)
                            elif c[j] in vertices_to_kill:
                                assert c[k] == vertices_to_kill[c[j]]
                                to_pop.append(j)
                            else:
                                vertices_to_kill[c[k]] = c[j]
                                to_pop.append(k)

                # remove the duplicated vertices from the cell_to_vertex_conn
                for j in reversed(sorted(to_pop)):
                    c.pop(j)

                # order around the primal vertex (now dual cell centroid)
                cell_coords = np.array([dual_vertices[j] for j in c]) - coords[i]
                angle = np.array([
                    np.arctan2(cell_coords[j, 1], cell_coords[j, 0])
                    for j in range(len(cell_coords))
                ])
                order = np.argsort(angle)
                dual_cells[i] = [c[j] for j in order]

        logging.info("-- removing duplicate vertices")
        # note this requires both removing the duplicates from the coordinate
        # list, but also remapping to new numbers that range in [0,
        # num_not_removed_vertices).  To do the latter, we create a map from old
        # indices to new indices, where removed indices are mapped to their
        # coincident, new index.  These old vertices may have appeared in cells
        # that did not actually include its duplicate, so must get remapped to
        # the coincident vertex.
        i = 0
        compression_map = -np.ones(len(dual_vertices), 'i')
        for j in range(len(dual_vertices)):
            if j not in vertices_to_kill:
                compression_map[j] = i
                i += 1
        for j in vertices_to_kill.keys():
            compression_map[j] = compression_map[vertices_to_kill[j]]
        assert (compression_map.min() >= 0)

        # -- delete the vertices
        to_kill = sorted(list(vertices_to_kill.keys()))
        dual_vertices = np.delete(dual_vertices, to_kill, axis=0)
        # -- remap the conn
        for c in dual_cells:
            for j in range(len(c)):
                c[j] = compression_map[c[j]]
            assert (min(c) >= 0)

        return dual_vertices, dual_cells, np.array(dual_from_primal_mapping)


@attr.define(slots=False)
class Mesh3D:
    """A 3D mesh class.

    Parameters
    ----------
    coords : np.ndarray(NCOORDS,3)
        Array of coordinates of the 3D mesh.
    face_to_vertex_conn : list(lists)
        List of lists of indices into coords that form the faces.
    cell_to_face_conn : list(lists)
        List of lists of indices into face_to_vertex_conn that form the
        cells.
    side_sets : list(SideSet), optional
        List of side sets to add to the mesh.
    labeled_sets : list(LabeledSet), optional
        List of labeled sets to add to the mesh.
    material_ids : np.array((len(cell_to_face_conn),),'i'), optional
        Array of length num_cells that specifies material IDs
    crs : CRS
        Keep the coordinate system for reference.
    cell_data : dict | pandas.DataFrame
        Extra cell-based data, stored as a dataframe.
    eps : float, optional=0.01
        A small measure of length between coords.

    Note that (coords, conn) may be output provided by a
    watershed_workflow.triangulation.triangulate() call.

    """
    coords = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    _face_to_vertex_conn = attr.ib()
    _cell_to_face_conn = attr.ib()

    labeled_sets: List[LabeledSet] = attr.ib(factory=list)
    side_sets: List[SideSet] = attr.ib(factory=list)
    material_ids: np.ndarray = attr.ib(default=None)
    crs: Optional[watershed_workflow.crs.CRS] = attr.ib(default=None)
    _cell_data: Optional[pandas.DataFrame] = attr.ib(default=None)
    eps: float = attr.ib(default=0.001)
    _validate: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        if self._validate:
            self.validate()

    @property
    def face_to_vertex_conn(self):
        return self._face_to_vertex_conn

    @property
    def cell_to_face_conn(self):
        return self._cell_to_face_conn

    @property
    def unique_material_ids(self):
        return list(np.unique(self.material_ids))

    @property
    def dim(self):
        return self.coords.shape[1]

    @property
    def num_cells(self):
        return len(self.cell_to_face_conn)

    @property
    def num_vertices(self):
        return self.coords.shape[0]

    @property
    def num_faces(self):
        return len(self.face_to_vertex_conn)


    @property
    @cache
    def barycentric_centroids(self):

        def _bary_centroid(m3, c):
            cverts = set(v for f in m3.cell_to_face_conn[c] for v in m3.face_to_vertex_conn[f])
            ccoords = np.array([m3.coords[v] for v in cverts])

            bary_c = ccoords.mean(axis=0)
            return bary_c

        return np.array([_bary_centroid(self, c) for c in range(self.num_cells)])

    @property
    @cache
    def mstk_centroids(self):
        def _mstk_centroid(m3, c):
            cverts = set(v for f in m3.cell_to_face_conn[c] for v in m3.face_to_vertex_conn[f])
            ccoords = np.array([m3.coords[v] for v in cverts])

            bary_c = ccoords.mean(axis=0)

            volume = 0
            ccentroid = np.array([0.,0.,0.])
    
            for f in m3.cell_to_face_conn[c]:
                f2v = m3.face_to_vertex_conn[f]
                fcoords = np.array([m3.coords[v] for v in f2v])
                bary_f = fcoords.mean(axis=0)

                # for each edge of the face, form the tet corresponding to the two edge points, the bary center of f, and the bary center of c.  Weight the centroid sum by the volume of that tet.
                for i in range(len(f2v)):
                    tet_centroid = (bary_c + bary_f + fcoords[i] + fcoords[(i+1)%len(f2v)]) / 4
                    v1 = fcoords[i] - bary_c
                    v2 = fcoords[(i+1)%len(f2v)] - bary_c
                    v3 = bary_f - bary_c
                    tet_vol = np.dot(v3, np.cross(v1,v2))

                    volume += tet_vol
                    ccentroid += tet_centroid * tet_vol

            ccentroid = ccentroid / volume
            return ccentroid
        
        return np.array([_mstk_centroid(self, c) for c in range(self.num_cells)])
        
    
    @property
    def cell_data(self):
        if self._cell_data is None:
            self._cell_data = pandas.DataFrame(index=range(len(self.conn)))
        return self._cell_data

    def validate(self) -> None:
        """Checks the validity of the mesh, or throws an AssertionError."""
        # validate coords
        assert (isinstance(self.coords, np.ndarray))
        assert (len(self.coords.shape) == 2)
        assert (self.coords.shape[1] == 3)

        # validate conn
        assert (min(c for conn in self.face_to_vertex_conn for c in conn) >= 0)
        assert (max(c for conn in self.face_to_vertex_conn for c in conn) < len(self.coords))

        assert (min(c for conn in self.cell_to_face_conn for c in conn) >= 0)
        assert (max(c for conn in self.cell_to_face_conn for c in conn)
                < len(self.face_to_vertex_conn))

        # validate uniqueness of labeled_set and side set IDs
        ls_ss_ids = [ls.setid for ls in self.labeled_sets] + \
            [ss.setid for ss in self.side_sets]
        assert (len(set(ls_ss_ids)) == len(ls_ss_ids))

        for ls in self.labeled_sets:
            if ls.entity == 'CELL':
                size = self.num_cells
            elif ls.entity == 'VERTEX':
                size = self.num_vertices
            else:
                raise ValueError("Mesh3D.validate: only 'CELL' or 'VERTEX' sets are supported -- face sets are supported as side sets")
            ls.validate(size, False)

        for ss in self.side_sets:
            ss.validate(self.cell_to_face_conn)

        # validate material ids
        assert self.material_ids is not None
        assert len(self.material_ids) == len(self.cell_to_face_conn)

    def getNextAvailableLabeledSetID(self) -> int:
        """Returns next available LS id."""
        i = 10000
        while any(i == ls.setid for ls in self.labeled_sets) or \
              any(i == ls.setid for ls in self.side_sets):
            i += 1
        return i

    def writeVTK(self, filename: str) -> None:
        """Writes to VTK.

        Note, this just writes the topology/geometry information, for
        WEDGE type meshes (extruded triangles).  No labeled sets are
        written.  Prefer to use writeExodus() for a fully featured
        mesh.

        Parameters
        ----------
        filename : str
            Output VTK filename.
        """
        import watershed_workflow.vtk_io
        assert (all(len(c) == 5 for c in self.cell_to_face_conn))
        wedges = []
        for c2f in self.cell_to_face_conn:
            fup = c2f[0]
            fdn = c2f[1]
            assert (len(self.face_to_vertex_conn[fup]) == 3
                    and len(self.face_to_vertex_conn[fdn]) == 3)
            wedges.append(self.face_to_vertex_conn[fup] + self.face_to_vertex_conn[fdn])
        watershed_workflow.vtk_io.write(filename, self.coords, { 'wedge': np.array(wedges) })

    def writeExodus(self, filename: str,
                    element_block_mode : str) -> None:
        """Write the 3D mesh to ExodusII using arbitrary polyhedra spec.

        Parameters
        ----------
        filename : str
            Output Exodus filename.
        element_block_mode : str, optional

            - "one block" (default): preserves sequential columnar
              ordering of elements by writing all elements to a single
              element block, and treating material IDs as element
              sets.
            - "material id" (old method): one element block per
              material ID -- reorders cells by material ID.

        """
        if exodus is None:
            raise ImportError(
                "The python ExodusII wrappers were not found, please see the installation documentation to install Exodus"
            )

        # Note exodus uses the term element instead of cell, so we
        # swap to that term in this method.

        # put elems in with blocks, which renumbers the elems, so we
        # have to track sidesets.  Therefore we keep a map of old elem
        # to new elem ordering

        # -- first pass, form all element blocks and make the map from old-to-new
        if element_block_mode == "one block":
            mat_ids_as_sets = True
            old_to_new_elems = list(enumerate(range(len(self.cell_to_face_conn))))
            elem_blks = [self.cell_to_face_conn,]
            elem_blk_ids = [9,]

        elif element_block_mode.lower() == "material id":
            mat_ids_as_sets = False
            new_to_old_elems = []
            elem_blks = []
            elem_blk_ids = self.unique_material_ids
            for i_m, m_id in enumerate(self.unique_material_ids):
                # split out elems of this material, save new_to_old map
                elems_tuple = [(i, c) for (i, c) in enumerate(self.cell_to_face_conn)
                               if self.material_ids[i] == m_id]
                new_to_old_elems.extend([i for (i, c) in elems_tuple])
                elems = [c for (i, c) in elems_tuple]
                elem_blks.append(elems)

            old_to_new_elems = sorted([(old, i) for (i, old) in enumerate(new_to_old_elems)],
                                      key=lambda a: a[0])

        else:
            raise ValueError(f'Invalid element_block_mode "{element_block_mode}", valid are "one block" and "material id"')

        # -- deal with faces, form all face blocks and make the map from old-to-new
        face_blks = [self.face_to_vertex_conn,]

        # add material IDs as labeled element sets
        if mat_ids_as_sets:
            for mat_id in self.unique_material_ids:
                mat_id_cells = np.where(self.material_ids == mat_id)[0]
                self.labeled_sets.append(LabeledSet(f"material ID {mat_id}", mat_id, 'CELL', mat_id_cells))

        # open the mesh file
        num_elems = sum(len(elem_blk) for elem_blk in elem_blks)
        num_faces = sum(len(face_blk) for face_blk in face_blks)

        filename_base = os.path.split(filename)[-1]
        ep = exodus.ex_init_params(title=filename_base.encode('ascii'),
                                   num_dim=3,
                                   num_nodes=self.num_vertices,
                                   num_face=num_faces,
                                   num_face_blk=len(face_blks),
                                   num_elem=num_elems,
                                   num_elem_blk=len(elem_blks),
                                   num_side_sets=len(self.side_sets),
                                   num_elem_sets=sum(1 for ls in self.labeled_sets
                                                     if ls.entity == 'CELL'),
                                   )
        e = exodus.exodus(filename, mode='w', array_type='numpy', init_params=ep)

        # put the coordinates
        e.put_coord_names(['coordX', 'coordY', 'coordZ'])
        e.put_coords(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2])

        # put the face blocks
        for i_blk, face_blk in enumerate(face_blks):
            face_raveled = [n for f in face_blk for n in f]
            e.put_polyhedra_face_blk(i_blk + 1, len(face_blk), len(face_raveled), 0)
            e.put_node_count_per_face(i_blk + 1, np.array([len(f) for f in face_blk]))
            e.put_face_node_conn(i_blk + 1, np.array(face_raveled) + 1)

        # set up variables
        if hasattr(self, 'cell_data') and 'partition' in self.cell_data:
            e.set_element_variable_number(1)
            e.put_element_variable_name('partition', 1)
            e.put_time(1, 0.0)

        # put the elem blocks
        for i_blk, (elem_blk_id, elem_blk) in enumerate(zip(elem_blk_ids, elem_blks)):
            elems_raveled = [f for c in elem_blk for f in c]

            # put the block
            e.put_polyhedra_elem_blk(elem_blk_id, len(elem_blk), len(elems_raveled), 0)
            e.put_elem_blk_name(elem_blk_id, f'ELEM_BLK_{elem_blk_id}')
            e.put_face_count_per_polyhedra(elem_blk_id, np.array([len(c) for c in elem_blk]))
            e.put_elem_face_conn(elem_blk_id, np.array(elems_raveled) + 1)

            # add cell_data as variables
            if hasattr(self, 'cell_data') and 'partition' in self.cell_data:
                logging.info(f'writing partition data as variable')
                values = list(self.cell_data['partition'].astype(float))
                e.put_element_variable_values(elem_blk_id, 'partition', 1, values)

        # add sidesets
        for ss in self.side_sets:
            logging.info(f'adding side set: {ss.setid}')
            for elem in ss.cell_list:
                assert old_to_new_elems[elem][0] == elem
            new_elem_list = sorted([(old_to_new_elems[elem][1], side)
                                    for (elem, side) in zip(ss.cell_list, ss.side_list)])
            e.put_side_set_params(ss.setid, len(ss.cell_list), 0)
            e.put_side_set_name(ss.setid, ss.name)
            e.put_side_set(ss.setid,
                           np.array([e_id for (e_id, side) in new_elem_list]) + 1,
                           np.array([side for (e_id, side) in new_elem_list]) + 1)

        # add labeled sets
        for ls in self.labeled_sets:
            if ls.entity == 'CELL':
                if hasattr(e, 'put_set_indices'):
                    logging.info(f'adding elem set: {ls.setid}')
                    new_elem_list = sorted([old_to_new_elems[elem][1] for elem in ls.ent_ids])
                    e.put_set_params('EX_ELEM_SET', ls.setid, len(new_elem_list), None)
                    e.put_set_name('EX_ELEM_SET', ls.setid, ls.name)
                    e.put_set_indices('EX_ELEM_SET', ls.setid, np.array(new_elem_list) + 1)
                else:
                    logging.warning(
                        f'not writing elem_set: {ls.setid} because exodus installation at {exodus.__file__} does not write element sets'
                    )
            else:
                warnings.warn(f'Cannot write labeled set of type {ls.entity}')

        # finish and close
        e.close()

    @staticmethod
    def summarizeExtrusion(layer_types,
                           layer_data,
                           ncells_per_layer,
                           mat_ids,
                           surface_cell_id=0) -> None:
        """
        Summarizes extruded data by printing info to log file.

        This is useful in rapidly debugging and understanding the layering before
        you do the extrusion process.
        """
        count = 0
        logging.info("Cell summary:")
        logging.info("-" * 60)
        logging.info("l_id\t| c_id\t|mat_id\t| dz\t\t| z_top")
        logging.info("-" * 60)
        rep_z = 0.
        for i, thick in enumerate(layer_data):
            for j in range(ncells_per_layer[i]):
                try:
                    mat_id = mat_ids[i][surface_cell_id]
                except TypeError:
                    mat_id = mat_ids[i]
                logging.info(" %02i \t| %02i \t| %4i \t| %10.6f \t| %10.6f" %
                             (i, count, mat_id, thick / ncells_per_layer[i], rep_z))
                count += 1
                rep_z += thick / ncells_per_layer[i]

    @classmethod
    def extruded_Mesh2D(cls, mesh2D, layer_types, layer_data, ncells_per_layer, mat_ids):
        """Uniformly extrude a 2D mesh to make a 3D mesh.

        Layers of potentially multiple sets of cells are extruded
        downward in the vertical.  The cell dz is uniform horizontally
        and vertically through the layer in all but the 'snapped'
        case, where it is uniform vertically but not horizontally.

        Parameters
        ----------
        mesh2D : Mesh2D
          The 2D mesh to extrude
        layer_types : str, list[str]
          One of ['snapped', 'function', 'vertex', 'cell',] or a list of
          these strings, describing the extrusion method for each
          layer.  If only a single string is supplied, all layers are
          of this type.  See layer_data below.
        layer_data : list[data]
          Data required, one for each layer.  The data is specific to
          the layer type:

          - 'constant' : float, layer thickness
          - 'snapped' : float, the bottom z-coordinate of the layer.
            Cells are extruded uniformly to whatever thickness
            required to match this bottom coordinate.
          - 'function' : function, layer_thickness = func(x,y)
          - 'vertex' : np.array((mesh2D.num_vertices,), float) layer
            thickness for each vertex
          - 'cell' : np.array((mesh2D.num_cells,), float)
            interpolates nodal layer thickness from neighboring cell
            thicknesses
        ncells_per_layer : int, list[int]
          Either a single integer (same number of cells in all layers)
          or a list of number of cells in each layer
        mat_ids : int, list[int], np.ndarray((num_layers, mesh2D.num_cells), dtype=int)
          Material ID for each cell in each layer.  If an int, all
          cells in all layers share thesame material ID.  If a list or
          1D array, one material ID per layer.  If a 2D array,
          provides all material IDs explicitly.

        Returns
        -------
        Mesh3D
          The extruded, 3D mesh.

        """

        # make the data all lists
        # ---------------------------------
        def is_list(data):
            if type(data) is str:
                return False
            try:
                iter(data)
            except TypeError:
                return False
            else:
                return True

        if is_list(layer_types):
            if not is_list(layer_data):
                layer_data = [layer_data, ] * len(layer_types)
            else:
                assert len(layer_data) == len(layer_types)

            if not is_list(ncells_per_layer):
                ncells_per_layer = [ncells_per_layer, ] * len(layer_types)
            else:
                assert len(ncells_per_layer) == len(layer_types)

        elif is_list(layer_data):
            layer_types = [layer_types, ] * len(layer_data)

            if not is_list(ncells_per_layer):
                ncells_per_layer = [ncells_per_layer, ] * len(layer_data)
            else:
                assert len(ncells_per_layer) == len(layer_data)

        elif is_list(ncells_per_layer):
            layer_type = [layer_type, ] * len(ncells_per_layer)
            layer_data = [layer_data, ] * len(ncells_per_layer)
        else:
            layer_type = [layer_type, ]
            layer_data = [layer_data, ]
            ncells_per_layer = [ncells_per_layer, ]

        # helper data and functions for mapping indices from 2D to 3D
        # ------------------------------------------------------------------
        if min(ncells_per_layer) < 0:
            raise RuntimeError("Invalid number of cells, negative value provided.")
        ncells_tall = sum(ncells_per_layer)
        ncells_total = ncells_tall * mesh2D.num_cells
        nfaces_total = (ncells_tall+1) * mesh2D.num_cells + ncells_tall * mesh2D.num_edges
        nvertices_total = (ncells_tall+1) * mesh2D.num_vertices

        eh = _ExtrusionHelper(ncells_tall, mesh2D.num_cells)

        np_mat_ids = np.array(mat_ids, dtype=int)
        if np_mat_ids.size == np.size(np_mat_ids, 0):
            if np_mat_ids.size == 1:
                np_mat_ids = np.full((len(ncells_per_layer), mesh2D.num_cells),
                                     mat_ids[0],
                                     dtype=int)
            else:
                np_mat_ids = np.empty((len(ncells_per_layer), mesh2D.num_cells), dtype=int)
                for ilay in range(len(ncells_per_layer)):
                    np_mat_ids[ilay, :] = np.full(mesh2D.num_cells, mat_ids[ilay], dtype=int)

        # create coordinates
        # ---------------------------------
        coords = np.zeros((mesh2D.coords.shape[0], ncells_tall + 1, 3), 'd')
        coords[:, :, 0:2] = np.expand_dims(mesh2D.coords[:, 0:2], 1)

        if mesh2D.dim == 3:
            coords[:, 0, 2] = mesh2D.coords[:, 2]
        # else the surface is at 0 depth

        cell_layer_start = 0
        for layer_type, layer_datum, ncells in zip(layer_types, layer_data, ncells_per_layer):
            if layer_type.lower() == 'constant':
                dz = float(layer_datum) / ncells
                for i in range(1, ncells + 1):
                    coords[:, cell_layer_start + i, 2] = coords[:, cell_layer_start, 2] - i*dz

            else:
                # allocate an array of coordinates for the bottom of the layer
                layer_bottom = np.zeros((mesh2D.coords.shape[0], ), 'd')

                if layer_type.lower() == 'snapped':
                    # layer bottom is uniform
                    layer_bottom[:] = layer_datum

                elif layer_type.lower() == 'function':
                    # layer thickness is given by a function evaluation of x,y
                    for vertex_col in range(mesh2D.coords.shape[0]):
                        layer_bottom[vertex_col] = coords[vertex_col, cell_layer_start,
                                                          2] - layer_datum(
                                                              coords[vertex_col, 0, 0],
                                                              coords[vertex_col, 0, 1])

                elif layer_type.lower() == 'vertex':
                    # layer bottom specifically provided through thickness
                    layer_bottom[:] = coords[:, cell_layer_start, 2] - layer_datum

                elif layer_type.lower() == 'cell':
                    # interpolate cell thicknesses to vertex thicknesses
                    import scipy.interpolate
                    centroids = mesh2D.centroids
                    interp = scipy.interpolate.interp2d(centroids[:, 0],
                                                        centroids[:, 1],
                                                        layer_datum,
                                                        kind='linear')
                    layer_bottom[:] = coords[:, cell_layer_start, 2] - interp(
                        mesh2D.coords[:, 0], mesh2D.coords[:, 1])

                else:
                    raise RuntimeError("Unrecognized layer_type '%s'" % layer_type)

                # linspace from bottom of previous layer to bottom of this layer
                for vertex_col in range(mesh2D.coords.shape[0]):
                    coords[vertex_col, cell_layer_start:cell_layer_start + ncells + 1,
                           2] = np.linspace(coords[vertex_col, cell_layer_start, 2],
                                            layer_bottom[vertex_col], ncells + 1)

            cell_layer_start = cell_layer_start + ncells

        # create faces, face sets, cells
        bottom = []
        surface = []
        faces = []
        cells = [list() for c in range(ncells_total)]

        # -- loop over the columns, adding the horizontal faces
        for col in range(mesh2D.num_cells):
            vertices_2 = mesh2D.conn[col]
            surface.append(eh.col_to_id(col, 0))
            for z_face in range(ncells_tall + 1):
                i_f = len(faces)
                f = [eh.vertex_to_id(n, z_face) for n in vertices_2]

                if z_face != ncells_tall:
                    cells[eh.col_to_id(col, z_face)].append(i_f)
                if z_face != 0:
                    cells[eh.col_to_id(col, z_face - 1)].append(i_f)

                faces.append(f)
            bottom.append(eh.col_to_id(col, ncells_tall - 1))

        # -- loop over the columns, adding the vertical faces
        added = dict()
        vertical_side_cells = []
        vertical_side_indices = []
        for col in range(mesh2D.num_cells):
            vertices_2 = mesh2D.conn[col]
            for i in range(len(vertices_2)):
                edge = Edge(vertices_2[i], vertices_2[(i+1) % len(vertices_2)])
                try:
                    i_e = added[edge]
                except KeyError:
                    # faces not yet added to facelist
                    i_e = len(added.keys())
                    added[edge] = i_e

                    for z_face in range(ncells_tall):
                        i_f = len(faces)
                        assert i_f == eh.edge_to_id(i_e, z_face)
                        f = [
                            eh.vertex_to_id(edge[0], z_face),
                            eh.vertex_to_id(edge[1], z_face),
                            eh.vertex_to_id(edge[1], z_face + 1),
                            eh.vertex_to_id(edge[0], z_face + 1)
                        ]
                        faces.append(f)
                        face_cell = eh.col_to_id(col, z_face)
                        cells[face_cell].append(i_f)

                        # check if this is an external
                        if len(mesh2D.edge_cells[edge]) == 1:
                            vertical_side_cells.append(face_cell)
                            vertical_side_indices.append(len(cells[face_cell]) - 1)

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
        num_sides = ncells_tall * sum(1 for e, c in mesh2D.edge_cells.items() if len(c) == 1)
        assert num_sides == len(vertical_side_cells)
        assert num_sides == len(vertical_side_indices)

        # make the material ids
        material_ids = np.zeros((len(cells), ), 'i')
        for col in range(mesh2D.num_cells):
            z_cell = 0
            for ilay in range(len(ncells_per_layer)):
                ncells = ncells_per_layer[ilay]
                for i in range(z_cell, z_cell + ncells):
                    material_ids[eh.col_to_id(col, i)] = np_mat_ids[ilay, col]
                z_cell = z_cell + ncells

        # make the side sets
        side_sets = []
        side_sets.append(SideSet("bottom", 1, bottom, [1, ] * len(bottom)))
        side_sets.append(SideSet("surface", 2, surface, [0, ] * len(surface)))
        side_sets.append(SideSet("external sides", 3, vertical_side_cells, vertical_side_indices))

        labeled_sets = []
        for ls in mesh2D.labeled_sets:
            if ls.entity == 'CELL':
                if hasattr(ls, 'to_extrude') and ls.to_extrude:
                    # top surface cells, to become columns of cells
                    elem_list = [eh.col_to_id(c, j) for j in range(ncells_tall) for c in ls.ent_ids]
                    labeled_sets.append(LabeledSet(ls.name, ls.setid, 'CELL', elem_list))
                else:
                    # top surface cells, to become side sets
                    elem_list = [eh.col_to_id(c, 0) for c in ls.ent_ids]
                    side_list = [0 for c in ls.ent_ids]
                    side_sets.append(SideSet(ls.name, ls.setid, elem_list, side_list))
            elif ls.entity == 'FACE':
                # top surface faces become faces in the subsurface
                # mesh, as they will extract correctly for surface BCs/observations
                outlet_ids_names = []

                # given a 2D edge, find a 2D cell it touches
                ls_edges = [Edge(e) for e in ls.ent_ids]
                col_ids = [mesh2D.edge_cells[e][0] for e in ls_edges]
                if hasattr(ls, 'to_extrude') and ls.to_extrude:
                    elem_list = [eh.col_to_id(c, i) for c in col_ids for i in range(ncells_tall)]
                    face_list = [
                        eh.edge_to_id(added[e], i) for e in ls_edges
                        for i in range(ncells_tall)
                    ]
                    side_list = [cells[c].index(f) for (f, c) in zip(face_list, elem_list)]
                else:
                    elem_list = [eh.col_to_id(c, 0) for c in col_ids]
                    face_list = [
                        eh.edge_to_id(added[e], 0) for e in ls_edges
                    ]
                    side_list = [cells[c].index(f) for (f, c) in zip(face_list, elem_list)]
                side_sets.append(SideSet(ls.name, ls.setid, elem_list, side_list))

        # make the cell data
        cell_data = None
        if hasattr(mesh2D, 'cell_data') and mesh2D.cell_data is not None:
            if 'partition' in mesh2D.cell_data:
                cell_data = pandas.DataFrame({'partition' : np.concatenate([mesh2D.cell_data.loc[c, 'partition'] * np.ones((ncells_tall,), 'i') for c in range(mesh2D.num_cells)])})

        # reshape coords
        coords = coords.reshape(nvertices_total, 3)

        # instantiate the mesh
        m3 = cls(coords,
                 faces,
                 cells,
                 material_ids=material_ids,
                 side_sets=side_sets,
                 labeled_sets=labeled_sets,
                 crs=mesh2D.crs,
                 cell_data=cell_data
                 )
        return m3


def computeTelescopeFactor(ncells: int, dz: float, layer_dz: float) -> float:
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
        raise ValueError(("Cannot telescope {} cells of thickness at least {} "
                          + "and reach a layer of thickness {}").format(ncells, dz, layer_dz))

    def seq(r):
        calc_layer_dz = 0
        dz_new = dz
        for i in range(ncells):
            calc_layer_dz += dz_new
            dz_new *= r

        #logging.debug('tried: {} got: {}'.format(r, calc_layer_dz))
        return layer_dz - calc_layer_dz

    res = scipy.optimize.root_scalar(seq, method='bisect', bracket=[1.0001, 2], maxiter=1000)
    logging.info("Converged?: ratio = {}, layer z (target = {}) = {}".format(
        res.root, layer_dz, seq(res.root)))
    return res.root


def optimizeDzs(dz_begin: float,
                dz_end: float,
                thickness: float,
                num_cells: int,
                p_thickness: float = 1000,
                p_dz: float = 10000,
                p_increasing: float = 1000,
                p_smooth: float = 10,
                tol: float = 1) -> Tuple[np.ndarray, float]:
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
        return np.abs(2 * growth_factor[1:-1] - growth_factor[0:-2] - growth_factor[2:]).sum()

    def penalty(dzs):
        return p_thickness * penalty_thickness(dzs) \
            + p_dz * penalty_dz(dzs) \
            + p_increasing * penalty_increasing(dzs) \
            + p_smooth * penalty_smooth(dzs)

    x0 = (dz_begin+dz_end) / 2. * np.ones((num_cells, ), 'd')
    bounds = [(dz_begin, dz_end), ] * num_cells
    res = scipy.optimize.minimize(penalty, x0, bounds=bounds)

    dzs = res.x.copy()[1:-1]

    # MUST have increasing
    dzs.sort()

    # MUST have sum
    dzs = dzs / dzs.sum() * thickness

    return dzs, res


def transformRotation(radians: float) -> np.ndarray:
    return np.array([[np.cos(radians), np.sin(radians), 0], [-np.sin(radians),
                                                             np.cos(radians), 0], [0, 0, 1]])


def createSubmesh(m2 : Mesh2D,
                  shp : shapely.geometry.base.BaseGeometry) \
        -> Tuple[Dict[int,int], Dict[int, int], Mesh2D]:
    """Given a shape that contains some cells of m2, create the submesh."""
    # create the new coordinates and a map
    new_coords_i: List[int] = []
    new_coords_map: Dict[int, int] = dict()
    for i, c in enumerate(m2.coords):
        if shp.intersects(shapely.geometry.Point(c[0], c[1])):
            new_coords_map[i] = len(new_coords_i)
            new_coords_i.append(i)
    new_coords = np.array([m2.coords[i] for i in new_coords_i])

    # create the new conn and map
    new_conns: List[List[int]] = []
    new_conn_map: Dict[int, int] = dict()
    for j, conn in enumerate(m2.conn):
        if all(i in new_coords_i for i in conn):
            new_conn = [new_coords_map[i] for i in conn]
            new_conn_map[j] = len(new_conns)
            new_conns.append(new_conn)

    # new labeled sets
    new_labeled_sets: List[LabeledSet] = []
    for ls in m2.labeled_sets:
        if (ls.entity == 'CELL'):
            new_ent_ids = [new_conn_map[e] for e in ls.ent_ids if e in new_conn_map.keys()]
            if len(new_ent_ids) > 0:
                new_ls = LabeledSet(ls.name, ls.setid, ls.entity, new_ent_ids, ls.to_extrude)
                new_labeled_sets.append(new_ls)
        elif (ls.entity == 'FACE'):
            new_edges = [(new_coords_map[e[0]], new_coords_map[e[1]]) for e in ls.ent_ids
                         if (e[0] in new_coords_map and e[1] in new_coords_map)]
            if len(new_edges) > 0:
                new_ls = LabeledSet(ls.name, ls.setid, ls.entity, new_edges, ls.to_extrude)
                new_labeled_sets.append(new_ls)

    # create the new mesh
    new_mesh = Mesh2D(new_coords, new_conns, new_labeled_sets, m2.crs, m2.eps, False, True)
    return new_coords_map, new_conn_map, new_mesh


def mergeMeshes(meshes: List[Mesh2D]) -> Mesh2D:
    """ Combines multiple 2D meshes into a single mesh.

    It is assumed that the meshes to be combined have common vertices
    on the shared edge (no steiner points). labeledsets should be added
    after merging the mesh. Option of merging pre-existing labeledsets
    in the to-be-merged meshes will be added soon

    Parameters
    ----------
    meshes : list(mesh.Mesh2D)
      The list of meshes to be merged

    Returns
    -------
    mesh.Mesh2D
      combined mesh
    """
    # ## identify base mesh
    # meshes = list(sorted(meshes, key=lambda m : -m.num_cells)) ##FIX ME##

    m2_combined = meshes[0]
    for mesh in meshes[1:]:
        m2_combined = mergeTwoMeshes(m2_combined, mesh)

    return m2_combined


def mergeTwoMeshes(mesh1: Mesh2D, mesh2: Mesh2D) -> Mesh2D:
    """merge two meshes (mesh.Mesh2D objects)"""
    # --THIS OPTION TO BE ADDED LATER-- #transfer_labeled_sets=True
    assert len(mesh1.labeled_sets) + len(mesh2.labeled_sets) == 0, \
        'to-be-merged meshes should not have labeled sets, they must be added after merging'

    combined_coords = mesh1.coords.tolist()
    # mapping for adjusting coord indices
    mapping = { i: i for i in range(mesh2.num_vertices) }

    # adjust connection indices for the second mesh using the mapping
    def map_conn(conn, mapping):
        return [mapping[idx] for idx in conn]

    for idx, coord in enumerate(mesh2.coords):
        # check if the point is close enough to any point in combined_coords
        found = False
        for i, existing_coord in enumerate(combined_coords):
            if watershed_workflow.utils.isClose(coord[:2], existing_coord[:2], 1e-3):
                mapping[idx] = i
                found = True
                break
        # if the point is not close enough to any existing point, add to combined_coords
        if not found:
            combined_coords.append(coord.tolist())
            mapping[idx] = len(combined_coords) - 1

    adjusted_conn2 = [map_conn(conn, mapping) for conn in mesh2.conn]

    # merge conn lists
    combined_conn = mesh1.conn + adjusted_conn2

    # merged mesh creation
    m2_combined = watershed_workflow.mesh.Mesh2D(coords=np.array(combined_coords),
                                                 conn=combined_conn)

    # # trasferring labeled sets ##--FIX ME--##
    # if transfer_labeled_sets:
    #     m2_combined.labeled_sets = m2_combined.labeled_sets + mesh1.labeled_sets

    #     for ls in mesh2.labeled_sets:
    #         if ls.entity == 'CELL':
    #             new_ent_ids = [mesh1.num_cells + c for c in ls.ent_ids]
    #             ls.ent_ids = new_ent_ids
    #         elif ls.entity == 'FACE':
    #             new_ent_ids = [(mapping(e[0]), mapping(e[1])) for e in ls.ent_ids]
    #             ls.ent_ids = new_ent_ids

    #     m2_combined.labeled_sets = m2_combined.labeled_sets + mesh2.labeled_sets

    return m2_combined


def refineTriangle(m2 : Mesh2D, c : int) -> Mesh2D:
    """Make a new mesh by refining a triangular cell.

    Note that cell c must be:
    - a triangle
    - that is not on the boundary
    - whose neighboring cells are also triangles

    """
    if len(m2.labeled_sets) != 0:
        raise ValueError("This algorithm does not correctly deal with labeled sets.")
    if len(m2.conn[c]) != 3:
        raise ValueError("Only triangles may be refined.")
    if any(e in m2.boundary_edges for e in m2.cell_edges(m2.conn[c])):
        raise ValueError("Only non-boundary triangles may be refined.")
    if not all(len(m2.conn[n]) == 3 for n in m2.cell_to_cells[c]):
        raise ValueError("Only triangles whose neighbors are also triangles may be refined.")

    old_edges = list(m2.cell_edges(m2.conn[c]))
    
    new_coords = np.array([(m2.coords[e[0]] + m2.coords[e[1]])/ 2. for e in old_edges])
    assert len(new_coords) == 3 # only triangles please!
    all_coords = np.concatenate([m2.coords, new_coords])
   
    # this triangle gets split into 4
    new_conn = []
    # the center triangle...
    new_conn.append(list(range(len(m2.coords), len(m2.coords)+3)))
    
    # the three external ones, one per vertex
    for i,e1 in enumerate(old_edges):
        e2 = old_edges[(i+1)%3]
        v1 = len(m2.coords) + i
        v2 = len(m2.coords) + (i + 1)%3
        v3 = e1[0] if e1[0] in e2 else e1[1]
        assert v3 in e2
        new_conn.append([v1, v2, v3])        

    # refine neighboring cells -- each a triangle that gets split in 2
    neighbors = m2.cell_to_cells[c]
    for i, e in enumerate(old_edges):
        # find the neighbor through edge e
        ecells = m2.edge_cells[e]
        assert len(ecells) == 2
        n = ecells[0] if ecells[1] == c else ecells[1]

        # find the node NOT in e
        for v in m2.conn[n]:
            if v not in e:
                # add the two new conn
                new_conn.append([e[0], len(m2.coords) + i, v])
                new_conn.append([e[1], len(m2.coords) + i, v])

    # remove the old cells
    all_conn = copy.copy(m2.conn)
    removed_cells = [c,] + neighbors
    for cc in reversed(sorted(removed_cells)):
        all_conn.pop(cc)
    all_conn.extend(new_conn)

    logging.debug(f' ... removing {len(removed_cells)}, adding {len(new_conn)} cells')
    logging.debug(f' ... adding {len(new_coords)} vertices')
    return watershed_workflow.mesh.Mesh2D(all_coords, all_conn, crs=m2.crs)
        

def refineTriangles(m2 : Mesh2D, to_refine : List[int]) -> Mesh2D:
    """Refine a set of triangles, making a new mesh.

    Note that cell in to_refine must be:
    - a triangle
    - that is not on the boundary
    - whose neighboring cells are also triangles
    - distinct -- for each pair of cells c1,c2 in to_refine, the set
      of c1 and its neighbors must not overlap with the set of c2 and
      its neighbors.

    """
    logging.info(f'Refining {len(to_refine)} triangles -- topology will change!')
    # make sure they are nonoverlapping
    affected = []
    for c in to_refine:
        affected.append(c)
        affected.extend(list(m2.cell_to_cells[c]))
        
    if len(set(affected)) != 4 * len(to_refine):
        raise ValueError('Cells in to_refine plus their neighbors are not distinct.')
    
    to_refine_conn = [m2.conn[c] for c in to_refine]
    m2r = m2
    for conn in to_refine_conn:
        c = next(i for (i, test_conn) in enumerate(m2r.conn) if len(conn) == len(test_conn) and all( j == k for (j,k) in zip(conn, test_conn)))   
        m2r = refineTriangle(m2r, c)

    return m2r, affected


def refineCorridorTriangles(m2 : Mesh2D,
                            river_corrs : List[shapely.geometry.Polygon]
                            ) -> Mesh2D:
    """Given a mesh, refine all triangles where all three vertices are
    on the river corridor.

    This deals with both interior junction triangles and sharp-angle
    reach triangles.
    """
    if len(m2.labeled_sets) != 0:
        raise ValueError("This algorithm does not correctly deal with labeled sets.")

    river_buffer = shapely.ops.unary_union(river_corrs).buffer(0.1)

    def _isStreamPoint(coord):
        return river_buffer.contains(shapely.geometry.Point(*coord))


    new_coords = list(m2.coords)
    new_conns = copy.deepcopy(m2.conn)
    def _splitTri(c,i,p):
        """Split tri c on edge conn[i],conn[i+1] at point p"""
        old_conn = [v for v in m2.conn[c]]
        new_conns[c] = [old_conn[(i-1)%3], old_conn[i], p]
        new_conns.append([p, old_conn[(i+1)%3], old_conn[(i+2)%3]])
    
    count = 0
    for c in range(m2.num_cells):
        # is a triangle, all 3 points on the corridor
        if len(m2.conn[c]) == 3 and all(_isStreamPoint(m2.coords[v]) for v in m2.conn[c]):
            # find a midpoint NOT on the corridor
            midps = []
            for (i,e) in enumerate(m2.cell_edges[c]):
                midp = m2.edge_centroids[e]
                if not _isStreamPoint(midp):
                    midps.append((i,e,midp))

            if len(midps) == 0:
                # likely this is a beginning-of-the-stream triangle
                continue
            elif len(midps) == 1:
                count = count + 1
                
                # split c along this edge
                i,e,new_p = midps[0]

                # add the new coordinate
                new_v = len(new_coords)
                new_coords.append(new_p)

                # replace and add a new triangle
                _splitTri(c, i, new_v)

                # find the neighbor and split it too
                try:
                    other_c = next(cc for cc in m2.edge_cells[e] if cc != c)
                except StopIteration:
                    pass
                else:
                    other_i = m2.cell_edges[other_c].index(e)
                    _splitTri(other_c, other_i, new_v)
                    

            else:
                raise RuntimeError('This should not be possible, something went wrong!')

    if count == 0:
        return m2
    else:
        return watershed_workflow.mesh.Mesh2D(new_coords, new_conns, crs=m2.crs)

    
