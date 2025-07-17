"""Module for working with tree data structures, built on watershed_workflow.tinytree

Note that this class knows how to work with the following properties,
which are expected to be spelled this way if they exist.  Only index
and geometry MUST exist.

index
  Must be the index of the DataFrame
geometry : shapely.LineString
  the river reach line
catchment : shapely.Polygon
  the local contributing area to this reach
area : double
  area [m^2] of catchment, the local contributing area

hydroseq : int
  See documentation for NHDPlus
dnhydroseq : int
  See documentation for NHDPlus

"""
from __future__ import annotations
from typing import List, Optional, Any, Tuple, Callable, Literal, Generator

import logging
import numpy as np
import copy
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import itertools
import sortedcontainers
import folium
import folium.plugins

import shapely.geometry
import shapely.ops
import geopandas as gpd
import pandas as pd
import pandas.api.types

import watershed_workflow.utils
import watershed_workflow.tinytree
from watershed_workflow.crs import CRS
import watershed_workflow.plot
import watershed_workflow.sources.standard_names as names

_tol = 1.e-7

class _RowView:
    """A helper class, this is what is returned in a call to node.properties.

    This behaves like a dictionary and represents a reference/view to
    the row of the pandas DataFrame that stores the underlying data in
    the tree.  We can't just return a row is because of pandas
    Copy-on-Write mechanism -- we really want a reference, not a copy.

    """
    def __init__(self,
                 df : gpd.GeoDataFrame,
                 index : int | str):
        self._df = df
        self._index = index
        self._df_reset : bool = False

    def __len__(self) -> int:
        return len(self._df.keys())
        
    def __getitem__(self, k : str) -> Any:
        if k == 'index':
            return self._index
        else:
            return self._df.at[self._index, k]

    def __setitem__(self, k : str, v : Any) -> None:
        self._df.loc[self._index, k] = v

    def __iter__(self) -> Any:
        return self._df.keys()

    def __contains__(self, k : str) -> bool:
        return k in self._df.keys()

    def keys(self) -> Any:
        return self._df.keys()

    def __repr__(self) ->str:
        return repr(self._df.loc[self._index])


class _MySortedList(sortedcontainers.SortedKeyList):
    def __init__(self, iterable=None):
        # note, SortedKeyList requires a total ordering.  Since angle
        # can be equal for two different linestrings (if they share
        # the same first segment angle), we also order (secondarily)
        # by id to make an arbitrary but now total ordering of nodes.
        return super(_MySortedList, self).__init__(iterable, key=lambda node : (node.angle,id(node)))
    
class River(watershed_workflow.tinytree.Tree):
    """A tree structure whose node data is stored in a pandas DataFrame, accessed by an index."""
    ListType = _MySortedList

    def __init__(self,
                 index : int | str,
                 df : gpd.GeoDataFrame,
                 children : Optional[List[River]] = None):
        """Do not call me.  Instead use the class factory methods, one of:

        - construct_rivers_by_geometry() # generic data
        - construct_rivers_by_hydroseq() # NHDPlus data

        This method initializes a single node in the River,
        representing one reach and its upstream children.

        """
        self.index = index
        self.df = df
        super(River, self).__init__(children)

    def __iter__(self):
        return self.preOrder()
            
    def addChild(self, child_or_index : River | str | int) -> River:
        """Append a child (upstream) reach to this reach."""
        if isinstance(child_or_index, River):
            super(River, self).addChild(child_or_index)
        else:
            super(River, self).addChild(type(self)(child_or_index, self.df))
        return self.children[-1]

    @property
    def linestring(self) -> shapely.geometry.LineString:
        """Returns the linestring geometry."""
        return self.df.at[self.index, 'geometry']

    @linestring.setter
    def linestring(self, value : shapely.geometry.LineString):
        self.df.loc[self.index, 'geometry'] = value

    @property
    def crs(self) -> CRS:
        return self.df.crs
        
    @property
    def properties(self):
        return _RowView(self.df, self.index)

    @properties.setter
    def properties(self, value):
        self.df.loc[self.index, value.keys()] = tuple(value.values())

    def __getitem__(self, name : str) -> Any:
        """Faster/preferred getter for properties"""
        if name == 'index':
            return self.index
        else:
            return self.df.at[self.index, name]

    def __setitem__(self, name : str, value : Any) -> None:
        """Faster/preferred setter for properties"""
        self.df.loc[self.index, name] = value

    def __contains__(self, name : str) -> bool:
        return self.df.__contains__(name)

    @property
    def angle(self):
        """Returns the angle, in radians, from node.parent.linestring to node.linestring, in a clockwise sense."""
        if self.parent is None:
            return 0.0
        else:
            return watershed_workflow.utils.computeAngle(self.parent.linestring, self.linestring)

    def plot(self, *args, **kwargs):
        """Plot the rivers."""
        inds = [r.index for r in self]
        return watershed_workflow.plot.linestringsWithCoords(self.df.loc[inds], *args, **kwargs)
    
    def explore(self, column=names.ID, m=None, marker=None, name=None, **kwargs):
        """Open a map!"""
        # get a name
        if column == names.ID and names.ID not in self.df:
            self.df[names.ID+"_as_column"] = self.df.index.astype('string')
            column = names.ID+"_as_column"

        if name is None:
            try:
                name = self[names.NAME]
            except KeyError:
                try:
                    name = self[names.ID]
                except KeyError:
                    name = self.index
                
        kwargs.setdefault('tooltip', False)

        default_props = [pname for pname in [names.NAME,
                                             'gnis_name',
                                             'ftype',
                                             names.ORDER,
                                             names.DIVERGENCE,
                                             names.HYDROSEQ,
                                             names.DOWNSTREAM_HYDROSEQ,
                                             names.UPSTREAM_HYDROSEQ,
                                             names.LENGTH,
                                             names.CATCHMENT_AREA, ] if pname in self.df]
        for p in self.df.keys():
            if len(default_props) >= 8:
                break
            if p not in default_props and p != 'geometry':
                default_props.append(p)
        kwargs.setdefault('popup', [names.ID,]+default_props)

        if 'color' not in kwargs:
            kwargs.setdefault('cmap', watershed_workflow.colors.xkcd_bolds)
        kwargs.setdefault('legend', False)

        # style
        style_kwds = kwargs.setdefault('style_kwds', dict())
        style_kwds.setdefault('weight', 5)

        # default explore
        m = self.df.explore(column=column, m=m, name=name, **kwargs)

        if marker:
            # don't reuse -- some versions keep the various *_kwds
            # dictionaries by reference
            kwargs2 = copy.deepcopy(kwargs)

            # explore the coordinates too!
            marker_kwds = kwargs2.setdefault('marker_kwds', dict())
            marker_kwds.setdefault('radius', 8)
            kwargs2['style_kwds']['fillOpacity'] = 1

            marker_df = self.df.set_geometry([shapely.geometry.MultiPoint(ls.coords) for ls in self.df.geometry]) \
                .explode(index_parts=True).reset_index(names=[names.ID, 'coord'])

            for disp_mode in ['tooltip', 'popup']:
                if disp_mode in kwargs2 and isinstance(kwargs2[disp_mode], list):
                    if names.ID in kwargs2[disp_mode]:
                        kwargs2[disp_mode].remove(names.ID)
                    kwargs2[disp_mode].insert(0,'coord')
                    kwargs2[disp_mode].insert(0,names.ID)

            m = marker_df.explore(column=column, m=m, name=self[names.NAME]+' coordinates', **kwargs2)
        return m
    

    #
    # methods that act on topology and geometry -- high level API
    #
    def split(self, i : int) -> Tuple[River, River]:
        """Split the reach at the ith coordinate of the linestring.

        Note that this does not split the catchment!

        self becomes the downstream node, and is modified in-place to
        preserve the full tree if the trunk is the one being split.
        
        Returns upstream_node, downstream_node.

        """
        if i < 0:
            i = len(self.linestring.coords) + i
        assert (i > 0 and i < len(self.linestring.coords) - 1)

        linestring = self.linestring
        upstream_linestring = shapely.geometry.LineString(list(linestring.coords)[0:i + 1])
        downstream_linestring = shapely.geometry.LineString(list(linestring.coords)[i:])
        upstream_area_frac = upstream_linestring.length / linestring.length

        # fix properties
        upstream_props = dict(self.properties)
        if names.CATCHMENT_AREA in upstream_props:
            upstream_props[names.CATCHMENT_AREA] = self[names.CATCHMENT_AREA] * upstream_area_frac
            self[names.CATCHMENT_AREA] = self[names.CATCHMENT_AREA] * (1-upstream_area_frac)

        if names.DRAINAGE_AREA in upstream_props:
            if names.CATCHMENT_AREA in self.properties:
                upstream_props[names.DRAINAGE_AREA] = self[names.DRAINAGE_AREA] - self[names.CATCHMENT_AREA]

        if names.HYDROSEQ in upstream_props:
            upstream_props[names.HYDROSEQ] = (self[names.HYDROSEQ] + self.parent[names.HYDROSEQ]) / 2.0
            if names.UPSTREAM_HYDROSEQ in upstream_props:
                self[names.UPSTREAM_HYDROSEQ] = upstream_props[names.HYDROSEQ]
            if names.DOWNSTREAM_HYDROSEQ in upstream_props:
                upstream_props[names.DOWNSTREAM_HYDROSEQ] = self[names.HYDROSEQ]

        if names.ID in self.properties:
            ID = self[names.ID]
            upstream_props[names.ID] = str(ID) + 'a'
            self[names.ID] = str(ID) + 'b'

        if names.DIVERGENCE in upstream_props:
            self[names.DIVERGENCE] = 0

        if names.CATCHMENT in upstream_props:
            upstream_props[names.CATCHMENT] = shapely.geometry.Point()

        upstream_props['geometry'] = upstream_linestring

        # new node
        # -- create a valid index and add the row to the dataframe
        if pandas.api.types.is_integer_dtype(self.df.index.dtype):
            new_index = pd.Series(max(self.df.index)+1).astype(self.df.index.dtype)[0]
        elif isinstance(self.index, str):
            if self.index[-1].isalpha():
                new_index = self.index[:-1] + chr(ord(self.index[-1])+1)
            else:
                new_index = str(self.index) + 'a'
        else:
            new_index = str(self.index) + 'a'

        assert new_index not in self.df.index

        crs = self.df.crs
        self.df.loc[new_index] = upstream_props

        # clean up -- see geopandas/geopandas#3119
        # -- geometry seems to be correct, but catchment is not, and the crs gets dropped
        if names.CATCHMENT in self.df:
            self.df[names.CATCHMENT] = self.df[names.CATCHMENT].astype(gpd.array.GeometryDtype())

        if crs is not None:
            self.df.set_crs(crs, inplace=True)
        # end clean up -- hopefully this gets fixed sometime in upstream (pandas or geopandas)

        # -- construct the new upstream node inject it into the tree
        new_node = self.__class__(new_index, self.df)
        self.giveChildren(new_node)
        self.linestring = downstream_linestring
        self.addChild(new_node)
        return new_node, self

    def splitAtArclen(self, s : float):
        """Inserts a coordinate at arclen s, then splits at that coordinate."""
        i = self.insertCoordinateByArclen(s)
        return self.split(i)

    def merge(self, merge_reach : bool = True) -> None:
        """Merges this with its parent."""
        parent = self.parent

        if merge_reach:
            assert (len(list(self.siblings)) == 0)
            new_seg = shapely.geometry.LineString(list(self.linestring.coords)[0:-1]+
                                                  list(parent.linestring.coords))
            parent.linestring = new_seg

        # fix properties
        if names.CATCHMENT_AREA in self:
            parent[names.CATCHMENT_AREA] += self[names.CATCHMENT_AREA]
        if names.CATCHMENT in self and self[names.CATCHMENT] is not None:
            if parent[names.CATCHMENT] is None:
                parent[names.CATCHMENT] = self[names.CATCHMENT]
            else:
                parent[names.CATCHMENT] = shapely.ops.unary_union(
                    [self[names.CATCHMENT], parent[names.CATCHMENT]])

        if names.DIVERGENCE in self:
            parent[names.DIVERGENCE] = self[names.DIVERGENCE]

        # set topology
        self.remove()
        self.linestring = shapely.geometry.LineString()
        for child in self.children:
            parent.addChild(child)

            
    def prune(self) -> None:
        """Removes this node and all below it, merging properties."""
        if self.parent is None:
            raise ValueError("Cannot prune a branch with no parent.")

        for node in self.postOrder():
            node.merge(False)

            
    #
    # methods that act on coordinates only
    #
    def moveCoordinate(self,
                       i : int,
                       xy : Tuple[float,float] | Tuple[float,float,float]) -> None:
        """Moves the ith coordinate of self.linestring to a new location."""
        if i < 0:
            i = len(self.linestring.coords) + i
        coords = list(self.linestring.coords)
        coords[i] = xy
        self.linestring = shapely.geometry.LineString(coords)

    def insertCoordinate(self, i : int, xy : Tuple[float,float]) -> int:
        """If it doesn't already exist, inserts a new coordinate before the ith coordinate.

        Returns the index of the new (or preexisting) coordinate.
        """
        if i < 0:
            i = len(self.linestring.coords) + i
        coords = list(self.linestring.coords)

        if watershed_workflow.utils.isClose(xy, coords[i-1]):
            # don't insert and existing point
            return i-1
        elif i < len(coords) and watershed_workflow.utils.isClose(xy, coords[i]):
            return i
        else:
            coords.insert(i, xy)
            self.linestring = shapely.geometry.LineString(coords)
            return i

    def insertCoordinateByArclen(self, s : float) -> int:
        """Inserts a new coordinate at a given arclen, returning the index of that coordinate.

        Note that arclen is measured from the downstream end!
        """
        sp = self.linestring.length - s
        
        coords = np.array(self.linestring.coords)
        dcoords = coords[1:] - coords[:-1]
        ds = np.linalg.norm(dcoords, axis=1)
        point_arclens = np.cumsum(ds)
        i = np.where(point_arclens > sp)[0][0]
        p = self.linestring.interpolate(sp)
        return self.insertCoordinate(i+1, p)

    def appendCoordinate(self, xy : Tuple[float,float]) -> None:
        """Appends a coordinate at the end (downstream) of the linestring."""
        coords = list(self.linestring.coords) + [xy, ]
        self.linestring = shapely.geometry.LineString(coords)

    def extendCoordinates(self, xys : List[Tuple[float,float]]) -> None:
        """Appends multiple coordinates at the end (downstream) of the linestring."""
        coords = list(self.linestring.coords) + xys
        self.linestring = shapely.geometry.LineString(coords)

    def prependCoordinates(self, xys : List[Tuple[float,float]]) -> None:
        """Prepends multiple coordinates at the beginning (upstream) of the linestring."""
        coords = xys + list(self.linestring.coords)
        self.linestring = shapely.geometry.LineString(coords)

    def popCoordinate(self, i : int) -> Tuple[float,float]:
        """Removes the ith coordinate and returns its value."""
        coords = list(self.linestring.coords)
        c = coords.pop(i)
        self.linestring = shapely.geometry.LineString(coords)
        return c


    #
    # Methods that act on the network and its properties
    #
    def accumulate(self,
                   to_accumulate : str,
                   to_save : Optional[str] = None,
                   op : Callable = sum):
        """Accumulates a property across the river tree."""
        val = op(child.accumulate(to_accumulate, to_save, op) for child in self.children)
        val = op([val, self[to_accumulate]])
        if to_save is not None:
            self[to_save] = val
        return val
    
    def getNode(self, index : int | str) -> River | None:
        """return node for a given index"""
        try:
            node = next(node for node in self if node.index == index)
        except StopIteration:
            node = None
        return node
    
    def findNode(self, lambd : Callable) -> River | None:
        """Find a node, returning the first whose lambda application is true, or None"""
        try:
            return next(n for n in self.preOrder() if lambd(n))
        except StopIteration:
            return None
        
    def assignOrder(self):
        """Working from leave to trunk, assign stream order property"""
        self.df[self.ORDER] = -1
        for leaf in self.leaf_nodes:
            leaf[self.ORDER] = 1

            node = leaf
            while node.parent[self.ORDER] == -1 and all(c[self.ORDER] > 0 for c in node.siblings):
                node = node.parent
                order = max(c[self.ORDER] for c in node.children)
                if len(node.children) > 1:
                    order += 1
                node[self.ORDER] = order

    def _isContinuous(self, child, tol : float = _tol) -> bool:
        """Is a given child continuous with self?"""
        return watershed_workflow.utils.isClose(child.linestring.coords[-1], self.linestring.coords[0], tol)

    def isLocallyContinuous(self, tol : float = _tol) -> bool:
        """Is this node continuous with its parent and children?"""
        res = all(self._isContinuous(child, tol=_tol) for child in self.children)
        if self.parent is not None:
            res = res and self.parent._isContinuous(self, tol=_tol)
        return res

    def isContinuous(self, tol : float = _tol) -> bool:
        """Checks geometric continuity of the river.

        Confirms that all upstream children's downstream coordinate
        coincides with self's upstream coordinate.
        """
        return all(self._isContinuous(child, tol) for child in self.children) and \
            all(child.isContinuous(tol) for child in self.children)

    def _makeContinuous(self, child : River) -> None:
        child_coords = list(child.linestring.coords)
        child_coords[-1] = list(self.linestring.coords)[0]
        child.linestring = shapely.geometry.LineString(child_coords)

    def makeContinuous(self, tol : float = _tol) -> None:
        """Sometimes there can be small gaps between linestrings of river
        tree if river is constructed using hydroseq and Snap
        option is not used. Here we make them consistent.

        """
        for node in self:
            for child in node.children:
                if not node._isContinuous(child, tol):
                    node._makeContinuous(child)
        assert (self.isContinuous())

    def isLocallyMonotonic(self) -> bool:
        """Checks for monotonically decreasing elevation as we march downstream in this reach."""
        coords = np.array(self.linestring.coords)
        if max(coords[1:,2] - coords[:-1,2]) > 0:
            return False
            
        for child in self.children:
            if self.linestring.coords[0][2] > child.linestring.coords[-1][2]:
                return False
        return True
        
    def isMonotonic(self, known_depressions = None) -> bool:
        if known_depressions is None:
            known_depressions = []
        return all(reach.isLocallyMonotonic() for reach in self if reach.index not in known_depressions)

    def isHydroseqConsistent(self) -> bool:
        """Confirms that hydrosequence is valid."""
        if len(self.children) == 0:
            return True

        self.children = sorted(self.children, key=lambda c: c[names.HYDROSEQ])
        return self[names.HYDROSEQ] < self.children[0][names.HYDROSEQ] and \
            all(child.isHydroseqConsistent() for child in self.children)

    def isConsistent(self, tol : float = _tol) -> bool:
        """Validity checking of the tree."""
        good = self.isContinuous(tol)
        if names.HYDROSEQ in self:
            good &= self.isHydroseqConsistent()
        return good

    def pathToRoot(self) -> Generator:
        """A generator for the nodes on the path to root, including this."""
        yield self
        if self.parent is not None:
            for n in self.parent.pathToRoot():
                yield n

    def resetDataFrame(self, force=False) -> None:
        """Resets the data frame for the river rooted at self, and
        reindexes the tree to a simple integer-based, preOrdered
        indexing.

        This restricts the (shared) DataFrame to a subset of rows that
        are all in the river rooted at self.

        """
        # this should only be called on a root!
        if self.parent is not None:
            raise ValueError("Only call resetDataFrame on a root of the river tree!")

        if getattr(self, '_df_reset', False) and not force:
            # already reset and not forced, all good
            return 
        
        # collect the indices in this river
        ids = [reach.index for reach in self]

        # subset the dataframe to just this river
        new_df = self.df.loc[ids]

        # reindex to a preOrdered listing
        # -- save the old index as ID
        if names.ID not in new_df.columns:
            new_df[names.ID] = new_df.index

        # -- create a new index
        assert 'new_preorder_index' not in new_df.columns
        new_df['new_preorder_index'] = -np.ones((len(new_df),), 'i')
        new_preorder_indices = dict((n.index, i) for (i,n) in enumerate(self))
        new_df.loc[list(new_preorder_indices.keys()), 'new_preorder_index'] = pd.Series(new_preorder_indices)

        # -- assign the new index as the index, then sort by this index
        new_df = new_df.set_index('new_preorder_index', drop=True).sort_index()

        # -- pass out to all reaches to make sure all have a reference
        #    to the same dataframe, and update the index of the reach
        for i,reach in enumerate(self):
            reach.index = i
            reach.df = new_df

        # mark a flag so that we don't have to do this repeatedly
        self._df_reset = True

    #
    # methods that convert this to another object
    #
    def to_crs(self, crs : CRS) -> None:

        """Warp the coordinate system."""
        self.df.to_crs(crs, inplace=True)

    def to_dataframe(self) -> gpd.GeoDataFrame:
        """Represent as GeoDataFrame, useful for pickling."""
        # reset the dataframe to be tight on this tree, and we can
        # rely on rows being in preOrder
        self.resetDataFrame()
        
        # move the parent into the dataframe
        def _parent_index(n):
            if n.parent is None:
                return None
            else:
                return n.parent[names.ID]

        self.df[names.PARENT] = [_parent_index(n) for n in self]
        self.df[names.PARENT] = self.df[names.PARENT].convert_dtypes()

        # move the children into the dataframe
        self.df[names.CHILDREN] = [[c.index for c in n.children] for n in self]
        self.df[names.CHILDREN] = self.df[names.CHILDREN].convert_dtypes()
        return self.df

    def to_mls(self) -> shapely.geometry.MultiLineString:
        """Represent this as a shapely.geometry.MultiLineString"""
        return shapely.geometry.MultiLineString([r.linestring for r in self])

    def to_file(self, filename : str, **kwargs) -> None:
        """Save the network for this river only to a geopandas file.

        Note this file can be reloaded via:

        $> watershed_workflow.river_tree.River.constructRiversByDataFrame(gpd.read_file(filename))

        """
        self.to_dataframe().to_file(filename, **kwargs)
    
    def copy(self, df : gpd.GeoDataFrame) -> River:
        """Shallow copy using a provided DataFrame"""
        if df is None:
            df = self.df
        copy_children = [child.copy(df) for child in self.children]
        return self.__class__(self.index, df, copy_children)

    def deepcopy(self) -> River:
        """Creates a deep copy of self"""
        df_copy = self.df.copy()
        return self.copy(df_copy)

    def copySubtree(self) -> River:
        """Returns a deep copy rooted at self."""
        inds = [r.index for r in self.preOrder()]
        df_copy = self.df.loc[inds]
        return self.copy(df_copy)

    #
    # Factory functions
    #
    @classmethod
    def constructRiversByGeometry(cls, df, tol : float = _tol):
        """Forms a list of River trees from a list of reaches by looking for
        close endpoints of those reaches.

        Note that this expects that endpoints of a reach coincide with
        beginpoints of their downstream reach, and does not work for
        cases where the junction is at a midpoint of a reach.
        """
        logging.debug("Generating Rivers")

        if len(df) == 0:
            return list()

        # make a kdtree of beginpoints
        coords = np.array([r.coords[0] for r in df['geometry']])
        kdtree = cKDTree(coords)

        # make a node for each linestring
        nodes = [cls(i,df) for i in df.index]

        # match nodes to their parent through the kdtree
        rivers = []
        divergence = []
        divergence_matches = []
        for j, n in enumerate(nodes):
            # find the closest beginpoint the this node's endpoint
            closest = kdtree.query_ball_point(n.linestring.coords[-1], tol)
            if len(closest) > 1:
                logging.debug("Bad multi linestring:")
                logging.debug(" connected to %d: %r" % (j, list(n.linestring.coords[-1])))
                divergence.append(j)
                divergence_matches.append(closest)

                # end at the same point, pick the min angle deviation
                my_tan = np.array(n.linestring.coords[-1]) - np.array(n.linestring.coords[-2])
                my_tan = my_tan / np.linalg.norm(my_tan)

                other_tans = [
                    np.array(df.geometry[c].coords[1]) - np.array(df.geometry[c].coords[0]) for c in closest
                ]
                other_tans = [ot / np.linalg.norm(ot) for ot in other_tans]
                dots = [np.inner(ot, my_tan) for ot in other_tans]
                for i, c in enumerate(closest):
                    logging.debug("  %d: %r --> %r with dot product = %g" %
                                  (c, coords[c], df.geometry[c].coords[-1], dots[i]))
                c = closest[np.argmax(dots)]
                nodes[c].addChild(n)

            elif len(closest) == 0:
                rivers.append(n)
            else:
                assert (len(closest) == 1)
                nodes[closest[0]].addChild(n)

        assert (len(rivers) > 0)
        return rivers

    
    @classmethod
    def constructRiversByHydroseq(cls, df):
        """Given a list of linestrings, create a list of rivers using the
        HydroSeq maps provided in NHDPlus datasets.
        """
        # create a map from hydroseq to node
        hydro_seq_ids = dict(zip(df[names.HYDROSEQ],
                                 (cls(i,df) for i in df.index)))

        roots = []
        for hs_id, node in hydro_seq_ids.items():
            down_hs_id = node[names.DOWNSTREAM_HYDROSEQ]
            try:
                hydro_seq_ids[down_hs_id].addChild(node)
            except KeyError:
                roots.append(node)
        return roots

    @classmethod
    def constructRiversByDataFrame(cls, df):
        """Create a list of rivers from a dataframe that includes a 'parent' column.
        """
        assert names.PARENT in df
        assert names.ID in df

        # create a dictionary from ID --> node (NOT index --> node!)
        nodes = dict((df.at[index, names.ID], cls(index, df)) for index in df.index)

        roots = []
        for ID, node in nodes.items():
            parent = df.at[node.index, names.PARENT]
            if pd.isna(parent):
                roots.append(node)
            else:
                nodes[parent].addChild(node)
        return roots

        
#
# Helper functions
#
def getNode(rivers, index):
    """Finds the node, by index, in a list of rivers"""
    for river in rivers:
        n = river.getNode(index)
        if n is not None:
            return n
    return None


def combineSiblings(n1 : River,
                    n2 : River,
                    new_ls : Optional[shapely.geometry.LineString] = None,
                    ds : Optional[float] = None) -> River:
    """Combines two sibling nodes, merging catchments and metadata.

    Note the resulting reach is either provided (by new_ls) or is
    computed by interpolating discrete nodes every ds.

    """
    assert (n1.isSiblingOf(n2))

    if new_ls is None:
        assert ds is not None
        avg_length = (n1.linestring.length + n2.linestring.length) / 2
        npoints = int(avg_length // ds) + 2
        ds1 = np.linspace(0, n1.linestring.length, npoints)
        points1 = n1.linestring.interpolate(ds1)
    
        ds2 = np.linspace(0, n2.linestring.length, npoints)
        points2 = n2.linestring.interpolate(ds2)

        points = [watershed_workflow.utils.computeMidpoint(p1.coords[0], p2.coords[0]) for (p1,p2) in zip(points1, points2)]
        new_ls = shapely.geometry.LineString(points)

    n1.linestring = new_ls
    
    if names.CATCHMENT_AREA in n1:
        n1[names.CATCHMENT_AREA] += n2[names.CATCHMENT_AREA]

    if 'catchment' in n2 and n2['catchment'] is not None:
        if n1['catchment'] is None:
            n1['catchment'] = n2['catchment']
        else:
            n1['catchment'] = shapely.ops.unary_union([n1['catchment'], n2['catchment']])

    for child in n1.children:
        if not watershed_workflow.utils.isClose(child.linestring.coords[-1], new_ls.coords[0]):
            child.appendCoordinate(new_ls.coords[0])
            
    for child in n2.children:
        if not watershed_workflow.utils.isClose(child.linestring.coords[-1], new_ls.coords[0]):
            child.appendCoordinate(new_ls.coords[0])
        if names.DOWNSTREAM_HYDROSEQ in child:
            child[names.DOWNSTREAM_HYDROSEQ] = n1[names.HYDROSEQ]
        n1.addChild(child)
    n2.children = []
    n2.remove()
    n2.linestring = shapely.geometry.LineString()
    return n1


#
# Construction method
#
def createRivers(reaches : gpd.GeoDataFrame,
                 method : Literal['geometry','hydroseq','native'] = 'geometry',
                 tol : float = _tol) -> List[River]:
    """Constructs River objects from a list of reaches.

    Parameters
    ----------
    :reaches: The reaches to turn into rivers.
    :method: Provide the method for constructing rivers.  Valid are:

        * 'geometry' looks at coincident coordinates
        * 'hydroseq' Valid only for NHDPlus data, this uses the
          NHDPlus VAA tables Hydrologic Sequence.  If using this
          method, get_reaches() must have been called with both
          'hydroseq' and 'dnhydroseq'
          properties requested (or properties=True).
        * 'native' Reads a natively dumped list of rivers.
    :tol: Defines what close is in the case of method == 'geometry'

    """
    if method == 'hydroseq':
        rivers = watershed_workflow.river_tree.River.constructRiversByHydroseq(reaches)
    elif method == 'geometry':
        rivers = watershed_workflow.river_tree.River.constructRiversByGeometry(reaches, tol)
    elif method == 'native':
        rivers = watershed_workflow.river_tree.River.constructRiversByDataFrame(reaches)
    else:
        raise ValueError(
            "Invalid method for making Rivers, must be one of 'hydroseq' or 'geometry'")

    # reset the data frame to be unique for each river
    for river in rivers:
        river.resetDataFrame()

    return rivers


#
# Helper functions on lists of rivers
#
def accumulateCatchments(rivers, outlet_indices, names=None):
    """Given a list of outlet_indices, find the reach in rivers and form its contributing area.
    
    Parameters:
    -----------
    rivers: list[River]
      Rivers from which outlet reaches are potentially from 
    outlet_indices: list[str]
      List of indices of the outlet reaches
    names: list[str], optional
      Names for the catchments

    Returns
    -------
    geopandas.GeoDataFrame
      DataFrame of the contributing areas to each of the outlet indices.
    """
    if names is None:
        names = outlet_indices

    roots = [river.getNode(index) for index in outlet_indices]
    assert (all(root is not None for root in roots))

    indices = ['CA_'+index for index in outlet_indices]
    outlet_points = [root.linestring.coords[-1] for root in roots]
    contributing_areas = [root.accumulateCatchments() for root in roots]
    return gpd.GeoDataFrame({names.INDEX : indices,
                             'outlet_ID' : outlet_indices,
                             'name' : names,
                             'outlet_point' : outlet_points,
                             'geometry' : contributing_areas},
                            crs=rivers[0].crs)
    

def accumulateIncrementalCatchments(rivers, outlet_indices, names=None):
    """Given a list of outlet_indices, form the incremental contributing areas.
    
    Parameters:
    -----------
    rivers: list[River]
      Rivers from which outlet reaches are potentially from 
    outlet_indices: list[str]
      List of indices of the outlet reaches
    names: list[str], optional
      Names for the catchments

    Returns
    -------
    gpd.GeoDataFrame
      DataFrame of the incremental contributing areas to each of the outlet indices.
    
    """
    if names is None:
        names = outlet_indices

    roots = [getNode(rivers, out_id) for out_id in outlet_indices]
    assert (all(root is not None for root in roots))

    sorted_ids = sorted(outlet_indices)

    def truncated_tree_iter(n):
        yield n
        for c in n.children:
            if c.index not in sorted_ids:
                for nn in truncated_tree_iter(c):
                    yield nn

    incremental_cas = [
        shapely.ops.unary_union([
            n['catchment'] for n in truncated_tree_iter(root)
            if n['catchment'] is not None
        ]) for root in roots
    ]

    indices = ['CA_'+index for index in outlet_indices]
    outlet_points = [root.linestring.coords[-1] for root in roots]

    return gpd.GeoDataFrame({names.ID : indices,
                                   'outlet_ID' : outlet_indices,
                                   'name' : names,
                                   'outlet_point' : outlet_points,
                                   'geometry' : incremental_cas},
                                  crs=rivers[0].crs)


#
# Cleanup methods -- merge and prune
#
def mergeShortReaches(river : River,
                      tol : float) -> None:
    """Remove inner branches that are short, combining branchpoints as needed.

    This function merges the "short" linestring into the child
    linestring if it is a junction tributary with one child or into
    the parent linestring otherwise.

    Note if tol is None, the tol is taken from the reach property TARGET_SEGMENT_LENGTH.

    """
    for node in list(river):
        if tol is None:
            ltol = node[names.TARGET_SEGMENT_LENGTH]
        else:
            ltol = tol
        if node.linestring.length < ltol and node.parent is not None:
            nname = node[names.ID] if names.ID in node.properties else node.index
            logging.info(
                "  ...cleaned inner linestring of length %g at centroid %r with id %r" %
                (node.linestring.length, node.linestring.centroid.coords[0], nname))

            if len(list(node.siblings)) > 0 and len(node.children) == 1:
                # junction tributary with one child
                node.children[0].merge()
            elif len(node.children) == 0:
                # if the leaf node is too small
                node.remove()
                node.linestring = shapely.geometry.LineString()
            else:
                for sibling in list(node.siblings):
                    sibling.moveCoordinate(-1, node.linestring.coords[0])
                    sibling.remove()
                    node.addChild(sibling)

                assert (len(list(node.siblings)) == 0)
                node.merge()
            

def pruneByLineStringLength(river : River,
                            prune_tol : Optional[float] = None) -> int:
    """Removes any leaf linestrings that are shorter than prune_tol"""
    count = 0
    iter_count = 1
    while iter_count > 0:
        iter_count = 0
        for leaf in river.leaf_nodes:
            if prune_tol is None:
                lprune_tol = leaf[names.TARGET_SEGMENT_LENGTH]
            else:
                lprune_tol = prune_tol

            if leaf.linestring.length < lprune_tol:
                count += 1
                logging.info("  ...cleaned leaf linestring of length: %g at centroid %r" %
                             (leaf.linestring.length, leaf.linestring.centroid.coords[0]))
                leaf.prune()
        count += iter_count
    return count


def pruneByArea(river : River,
                area : float,
                prop : str = names.DRAINAGE_AREA):
    """Removes, IN PLACE, reaches whose total contributing area is less than area km^2.

    Note this requires NHDPlus data to have been used and the
    'DivergenceRoutedDrainAreaSqKm' property (or whatever is selected) to have been set.
    """
    count = 0
    for node in river.preOrder():
        # note, we only ever prune children, to avoid unneeded recursive pruning
        #
        # make a copy of the children, as this list will be modified by potential prune calls
        children = node.children[:]
        for child in children:
            if child[prop] < area:
                logging.debug(f"... removing trib with {len(child)}"
                              f" reaches of area: {child[prop]}")
                count += len(child)
                child.prune()
    return count


def pruneRiversByArea(rivers : List[River],
                      area : float,
                      prop : str = names.DRAINAGE_AREA) -> List[River]:
    """Both prunes reaches and filters rivers whose contributing area is less than area."""
    num_reaches = sum(len(river) for river in rivers)
    count = 0
    sufficiently_big_rivers = []
    for river in rivers:
        if river[prop] >= area:
            count += pruneByArea(river, area, prop)
            sufficiently_big_rivers.append(river)
        else:
            count += len(river)
    logging.info(f"... pruned {count} of {num_reaches}")
    return sufficiently_big_rivers
    

def filterDiversions(rivers : List[River]) -> List[River]:
    """Filteres diversions, but not braids."""
    logging.info("Remove diversions...")
    non_diversions = []
    for river in rivers:
        keep_river = True
        count_tribs = 0
        count_reaches = 0
        for leaf in river.leaf_nodes:
            if leaf[names.DIVERGENCE] == 2:
                # is a braid or a diversion
                if river.getNode(leaf[names.UPSTREAM_HYDROSEQ]) is None:
                    # diversion!
                    try:
                        joiner = next(n for n in leaf.pathToRoot()
                                      if n.parent is not None and len(n.parent.children) > 1)
                    except StopIteration:
                        # no joiner means kill the whole river
                        logging.info(f'  ... remove diversion river with {len(river)} reaches.')
                        keep_river = False
                        break
                    else:
                        count_tribs += 1
                        count_reaches += len(joiner)
                        joiner.prune()

        if keep_river:
            logging.info(
                f'  ... removed {count_tribs} diversion tributaries with {count_reaches} total reaches.'
            )
            non_diversions.append(river)

    return non_diversions

    
def removeBraids(rivers : List[River]) -> None:
    """Remove braids, but not diversions."""
    logging.debug("Removing braided sections...")
    for river in rivers:
        count_tribs = 0
        count_reaches = 0

        for leaf in river.leaf_nodes:
            if leaf[names.DIVERGENCE] == 2:
                # is a braid or a diversion?
                logging.info(f"  Found a braid with upstream = {leaf[names.UPSTREAM_HYDROSEQ]}")
                upstream_hydroseq = leaf[names.UPSTREAM_HYDROSEQ]
                if river.findNode(lambda n : n[names.HYDROSEQ] == upstream_hydroseq) is not None:
                    # braid!
                    try:
                        joiner = next(n for n in leaf.pathToRoot()
                                      if n.parent is not None and len(n.parent.children) > 1)
                    except StopIteration:
                        assert (False)
                        # this should not be possible, because our braid must come back somewhere
                    else:
                        count_tribs += 1
                        count_reaches += len(joiner)
                        joiner.prune()

        logging.debug(f'... removed {count_tribs} braids with {count_reaches} reaches'
                      f' from a river of length {len(river)}')


def filterDivergences(rivers : List[River]) -> List[River]:
    """Removes both diversions and braids.

    Braids are divergences that return to the river network, and so
    look like branches of a river tree whose upstream entity is in the
    river (in another branch).

    Diversions are divergences that do not return to the stream
    network, and so their upstream entity is in another river.

    """
    logging.info("Removing divergent sections...")
    non_divergences = []

    for river in rivers:
        keep_river = True
        count_tribs = 0
        count_reaches = 0
        for leaf in river.leaf_nodes:
            if leaf[names.DIVERGENCE] == 2:
                # diversion!
                try:
                    joiner = next(n for n in leaf.pathToRoot()
                                  if n.parent is not None and len(n.parent.children) > 1)
                except StopIteration:
                    # no joiner means kill the whole river
                    logging.info(f'  ... remove divergence river with {len(river)} reaches.')
                    keep_river = False
                    break
                else:
                    count_tribs += 1
                    count_reaches += len(joiner)
                    joiner.prune()

        if keep_river:
            logging.info(
                f'  ... removed {count_tribs} divergence tributaries with {count_reaches} total reaches.'
            )
            non_divergences.append(river)

    return non_divergences

    
def filterSmallRivers(rivers : List[River],
                      count : int) -> List[River]:
    """Remove any rivers with fewer than count reaches."""
    logging.info(f"Removing rivers with fewer than {count} reaches.")
    new_rivers = []
    for river in rivers:
        ltree = len(river)
        if ltree < count:
            logging.debug(f"  ...removing river with {ltree} reaches")
        else:
            new_rivers.append(river)
            logging.debug(f"  ...keeping river with {ltree} reaches")
    logging.info(f'... removed {len(rivers) - len(new_rivers)} rivers')
    return new_rivers


def simplify(rivers : List[River],
             tol : float) -> None:
    """Simplify, IN PLACE, all reaches."""
    if len(rivers) == 0:
        return

    rivers[0].df.simplify(tol)
    for river in rivers[1:]:
        if river.df is not rivers[0].df:
            river.df.simplify(tol)


def isClose(river1 : River,
            river2 : River,
            tol : float):
    """Equivalence of rivers."""
    return all((watershed_workflow.utils.isClose(r1.linestring, r2.linestring, tol) and
                    len(r1.children) == len(r2.children))
                   for (r1,r2) in zip(river1.preOrder(), river2.preOrder()))
    
