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

HydrologicSequence : int
  See documentation for NHDPlus
DownstreamMainpathHydroSeq : int
  ee documentation for NHDPlus

"""
import logging
import numpy as np
import copy
from scipy.spatial import cKDTree

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree

_tol = 1.e-7

class _RowView:
    """A helper class, this is what is returned in a call to node.properties.

    This behaves like a dictionary and represents a reference/view to
    the row of the pandas DataFrame that stores the underlying data in
    the tree.  Why we can't just return a row is because of pandas
    Copy-on-Write mechanism.

    """
    def __init__(self, df, index):
        self._df = df
        self._index = index

    def __len__(self):
        return len(self._df.keys())
        
    def __getitem__(self, k):
        return self._df.at[self._index, k]

    def __setitem__(self, k, v):
        df.loc[self._index, k] = v

    def __iter__(self):
        return self._df.keys()

    def keys(self):
        return self._df.keys()

    
class River(watershed_workflow.tinytree.Tree):
    """A tree structure whose node data is stored in a pandas DataFrame, accessed by an index."""

    def __init__(self, index, df, children=None):
        """Do not call me.  Instead use the class factory methods, one of:

        - construct_rivers_by_geometry() # generic data
        - construct_rivers_by_hydroseq() # NHDPlus data

        This method initializes a single node in the River,
        representing one reach and its upstream children.

        """
        super(River, self).__init__(children)
        self.index = index
        self.df = df

    def __iter__(self):
        return self.preOrder()
            
    def addChild(self, child_or_index):
        """Append a child (upstream) reach to this reach."""
        if isinstance(child_or_index, River):
            super(River, self).addChild(child_or_index)
        else:
            super(River, self).addChild(type(self)(child_or_index, self.df))
        return self.children[-1]

    @property
    def segment(self):
        """Returns the segment geometry."""
        return self.df.at[self.index, 'geometry']

    @segment.setter
    def segment(self, value):
        assert (isinstance(value, shapely.geometry.LineString))
        self.df.loc[self.index, 'geometry'] = value

    @property
    def crs(self):
        return self.df.crs
        
    @property
    def properties(self):
        return _RowView(self.df, self.index)

    @properties.setter
    def properties(self, value):
        self.df.loc[self.index, value.keys()] = tuple(value.values())

    def __getitem__(self, name):
        """Faster/preferred getter for properties"""
        return self.df.at[self.index, name]

    def __setitem__(self, name, value):
        """Faster/preferred setter for properties"""
        self.df.loc[self.index, name] = value

    def __contains__(self, name):
        return self.df.__contains__(name)
        
    def split(self, i):
        """Split the reach at the ith coordinate of the segment.

        Note that this does not split the catchment!

        Returns upstream_node, downstream_node
        """
        if i < 0:
            i = len(self.segment.coords) + i
        assert (i > 0 and i < len(self.segment.coords) - 1)

        segment = self.segment
        upstream_segment = shapely.geometry.LineString(list(segment.coords)[0:i + 1])
        downstream_segment = shapely.geometry.LineString(list(segment.coords)[i:])
        downstream_area_frac = downstream_segment.length / segment.length

        # fix properties
        downstream_props = copy.deepcopy(self.properties)
        if 'area' in downstream_props:
            downstream_props['area'] = self['area'] * downstream_area_frac
            self['area'] = self['area'] * (1-downstream_area_frac)

        if 'HydrologicSequence' in downstream_props:
            downstream_props['HydrologicSequence'] -= 0.5
            self['DownstreamMainPathHydroSeq'] = downstream_props['HydrologicSequence']

        index = self['index']
        downstream_props['index'] = index + 'a'
        self['index'] = index + 'b'

        if 'DivergenceCode' in downstream_props:
            downstream_props['DivergenceCode'] = 0

        # detach self
        self.segment = upstream_segment
        parent = self.parent
        self.remove()

        # new node
        # -- add the row to the dataframe
        self.df.loc[downstream_props['index']] = downstream_props
        # -- construct the node and add it to the original parent
        new_node = self.__class__(River(downstream_props['index'], self.df, [self,]))
        parent.addChild(new_node)
        return self, new_node

    def merge(self, merge_reach=True):
        """Merges this with its parent."""
        parent = self.parent

        if merge_reach:
            assert (len(list(self.siblings)) == 0)
            new_seg = shapely.geometry.LineString(list(self.segment.coords)[0:-1]+
                                                  list(parent.segment.coords))
            parent.segment = new_seg

        # fix properties
        if 'area' in self:
            self.parent['area'] += self['area']
        if 'catchment' in self and self['catchment'] is not None:
            if self.parent['catchment'] is None:
                self.parent['catchment'] = self['catchment']
            else:
                self.parent['catchment'] = shapely.ops.unary_union(
                    [self['catchment'], self.parent['catchment']])

        if 'DivergenceCode' in self:
            self.parent['DivergenceCode'] = self['DivergenceCode']

        # set topology
        self.remove()
        for child in self.children:
            parent.addChild(child)

    def prune(self):
        """Removes this node and all below it, merging properties."""
        if self.parent is None:
            raise ValueError("Cannot prune a branch with no parent.")

        for node in self.postOrder():
            node.merge(False)

    def moveCoordinate(self, i, xy):
        """Moves the ith coordinate of self.segment to a new location."""
        if i < 0:
            i = len(self.segment.coords) + i
        coords = list(self.segment.coords)
        coords[i] = xy
        self.segment = shapely.geometry.LineString(coords)

    def insertCoordinate(self, i, xy):
        """Inserts a new coordinate before the ith coordinate."""
        if i < 0:
            i = len(self.segment.coords) + i
        coords = list(self.segment.coords)
        coords.insert(i, xy)
        self.segment = shapely.geometry.LineString(coords)

    def appendCoordinate(self, xy):
        """Appends a coordinate at the end (downstream) of the segment."""
        coords = list(self.segment.coords) + [xy, ]
        self.segment = shapely.geometry.LineString(coords)

    def extendCoordinates(self, xys):
        """Appends multiple coordinates at the end (downstream) of the segment."""
        coords = list(self.segment.coords) + xys
        self.segment = shapely.geometry.LineString(coords)

    def prependCoordinates(self, xys):
        """Prepends multiple coordinates at the beginning (upstream) of the segment."""
        coords = xys + list(self.segment.coords)
        self.segment = shapely.geometry.LineString(coords)

    def popCoordinate(self, i):
        """Removes the ith coordinate and returns its value."""
        coords = list(self.segment.coords)
        coords.pop(i)
        self.segment = shapely.geometry.LineString(coords)

    def accumulate(self, to_accumulate, to_save=None, op=sum):
        """Accumulates a property across the river tree."""
        val = op(child.accumulate(to_accumulate, to_save, op) for child in self.children)
        val = op([val, self.properties[to_accumulate]])
        if to_save is not None:
            self.properties[to_save] = val
        return val

    def getNode(self, index):
        """return node for a given index"""
        try:
            node = next(node for node in self.preOrder() if node.index == index)
        except StopIteration:
            node = None
        return node

    def findNode(self, lambd):
        """Find a node, returning the first whose lambda application is true, or None"""
        try:
            return next(n for n in self.preOrder() if lambd(n))
        except StopIteration:
            return None

    def _isContinuous(self, child, tol=_tol):
        """Is a given child continuous with self?"""
        return watershed_workflow.utils.isClose(child.segment.coords[-1], self.segment.coords[0], tol)

    def isLocallyContinuous(self, tol=_tol):
        """Is this node continuous with its parent and children?"""
        res = all(self._isContinuous(child, tol=_tol) for child in self.children)
        if self.parent is not None:
            res = res and self.parent._isContinuous(self, tol=_tol)
        return res

    def isContinuous(self, tol=_tol):
        """Checks geometric continuity of the river.

        Confirms that all upstream children's downstream coordinate
        coincides with self's upstream coordinate.
        """
        return all(self._isContinuous(child, tol) for child in self.children) and \
            all(child.isContinuous(tol) for child in self.children)

    def _makeContinuous(self, child):
        child_coords = list(child.segment.coords)
        child_coords[-1] = list(self.segment.coords)[0]
        child.segment = shapely.geometry.LineString(child_coords)

    def makeContinuous(self, tol=_tol):
        """Sometimes there can be small gaps between segments of river
        tree if river is constructed using HydrologicSequence and Snap
        option is not used. Here we make them consistent.

        """
        for node in self.preOrder():
            for child in node.children:
                if not node._isContinuous(child, tol):
                    node._makeContinuous(child)
        assert (self.isContinuous())

    def isHydroseqConsistent(self):
        """Confirms that hydrosequence is valid."""
        if len(self.children) == 0:
            return True

        self.children = sorted(self.children, key=lambda c: c.HydrologicSequence)
        return self.properties['HydrologicSequence'] < self.children[0].properties['HydrologicSequence'] and \
            all(child.isHydroseqConsistent() for child in self.children)

    def isConsistent(self, tol=_tol):
        """Validity checking of the tree."""
        good = self.isContinuous(tol)
        if 'HydrologicSequence' in self:
            good &= self.isHydroseqConsistent()
        return good

    def pathToRoot(self):
        if self.parent is not None:
            yield self.parent
            for n in self.parent.pathToRoot():
                yield n
    
    def to_crs(self, crs):
        """Warp the coordinate system."""
        self.df.to_crs(crs, inplace=True)

    def to_dataframe(self):
        """Represent as GeoDataFrame"""
        # move the parent into the dataframe
        def _parent_index(n):
            if n.parent is None:
                return None
            else:
                return n.parent.index

        self.df['parent'] = dict([(n.index, _parent_index(n)) for n in self.preOrder()])
        self.df['parent'] = self.df['parent'].convert_dtypes()

        # move the children into the dataframe
        self.df['children'] = dict([(n.index, [c.index for c in n.children]) for n in self.preOrder()])
        self.df['children'] = self.df['children'].convert_dtypes()
        return self.df

    def to_mls(self):
        """Represent this as a shapely.geometry.MultiLineString"""
        return shapely.geometry.MultiLineString([r.segment for r in self.preOrder()])
    
    def _copy(self, df):
        """Shallow copy using a provided DataFrame"""
        if df is None:
            df = self.df
        copy_children = [child.copy(df) for child in self.children]
        return self.__class__(self.index, df, copy_children)
    
    def deepcopy(self):
        """Creates a deep copy of self"""
        df_copy = self.df.copy()
        return self._copy(df_copy)
    
    #
    # Factory functions
    #
    @classmethod
    def constructRiversByGeometry(cls, df, tol=_tol):
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

        # make a node for each segment
        nodes = [cls(i,df) for i in df.index]

        # match nodes to their parent through the kdtree
        rivers = []
        divergence = []
        divergence_matches = []
        for j, n in enumerate(nodes):
            # find the closest beginpoint the this node's endpoint
            closest = kdtree.query_ball_point(n.segment.coords[-1], tol)
            if len(closest) > 1:
                logging.debug("Bad multi segment:")
                logging.debug(" connected to %d: %r" % (j, list(n.segment.coords[-1])))
                divergence.append(j)
                divergence_matches.append(closest)

                # end at the same point, pick the min angle deviation
                my_tan = np.array(n.segment.coords[-1]) - np.array(n.segment.coords[-2])
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
        """Given a list of segments, create a list of rivers using the
        HydroSeq maps provided in NHDPlus datasets.
        """
        # create a map from HydrologicSequence to node
        hydro_seq_ids = dict(zip(df['HydrologicSequence'],
                                 (cls(i,df) for i in df.index)))

        roots = []
        for hs_id, node in hydro_seq_ids.items():
            down_hs_id = node['DownstreamMainPathHydroSeq']
            try:
                hydro_seq_ids[down_hs_id].addChild(node)
            except KeyError:
                roots.append(node)
        return roots

    @classmethod
    def constructRiversByDataFrame(cls, df):
        """Create a list of rivers from a dataframe that includes a 'parent' column.
        """
        assert 'parent' in df
        nodes = dict([(index,cls(index, df)) for index in df.index])

        roots = []
        for index, node in nodes.items():
            parent = df.at[index, 'parent']
            if pandas.isna(parent):
                roots.append(node)
            else:
                nodes[parent].addChild(node)
        return roots

        
def getNode(rivers, index):
    """Finds the node, by index, in a list of rivers"""
    for river in rivers:
        n = river.getNode(index)
        if n is not None:
            return n
    return None


def combineSiblings(n1, n2):
    """Combines two sibling nodes, merging catchments and metadata.

    Note the resulting reach segment only merges the beginpoint and endpoint.
    """
    assert (n1.isSiblingOf(n2))
    beginpoint = (np.array(n1.segment.coords[0]) + np.array(n2.segment.coords[0])) / 2
    endpoint = (np.array(n1.segment.coords[-1]) + np.array(n2.segment.coords[-1])) / 2
    new_seg = shapely.geometry.LineString([beginpoint, endpoint])
    n1.segment = new_seg

    if 'area' in n1:
        n1['area'] += n2['area']

    if 'catchment' in n2 and n2['catchment'] is not None:
        if n1['catchment'] is None:
            n1['catchment'] = n2['catchment']
        else:
            n1['catchment'] = shapely.ops.unary_union([n1['catchment'], n2['catchment']])

    for child in n2.children:
        if 'DownstreamMainPathHydroSeq' in child:
            child['DownstreamMainPathHydroSeq'] = n1['HydrologicSequence']
        n1.addChild(child)
    n2.children = []
    n2.remove()

    for child in n1.children:
        child.moveCoordinate(-1, n1.segment.coords[0])

    return n1


def accumulateCatchments(rivers, outlet_indices, names=None):
    """Given a list of outlet_indices, find the reach in rivers and form its contributing area.
    
    Parameters:
    -----------
    rivers: list[RiverTree]
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
    outlet_points = [root.segment.coords[-1] for root in roots]
    contributing_areas = [root.accumulateCatchments() for root in roots]
    return geopandas.GeoDataFrame({'index' : indices,
                                   'outlet_index' : outlet_indices,
                                   'name' : names,
                                   'outlet_point' : outlet_points,
                                   'geometry' : contributing_areas},
                                  crs=rivers[0].crs)
    

def accumulateIncrementalCatchments(rivers, outlet_indices, names=None):
    """Given a list of outlet_indices, form the incremental contributing areas.
    
    Parameters:
    -----------
    rivers: list[RiverTree]
      Rivers from which outlet reaches are potentially from 
    outlet_indices: list[str]
      List of indices of the outlet reaches
    names: list[str], optional
      Names for the catchments

    Returns
    -------
    geopandas.GeoDataFrame
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
    outlet_points = [root.segment.coords[-1] for root in roots]

    return geopandas.GeoDataFrame({'index' : indices,
                                   'outlet_index' : outlet_indices,
                                   'name' : names,
                                   'outlet_point' : outlet_points,
                                   'geometry' : incremental_cas},
                                  crs=rivers[0].crs)


def createRiverTrees(reaches, method='geometry', tol=_tol):
    """Constructs River objects from a list of reaches.

    Parameters
    ----------
    reaches : list[LineString]
      List of reaches
    method : str, optional='geometry'
        Provide the method for constructing rivers.  Valid are:

        * 'geometry' looks at coincident coordinates
        * 'hydroseq' Valid only for NHDPlus data, this uses the
          NHDPlus VAA tables Hydrologic Sequence.  If using this
          method, get_reaches() must have been called with both
          'HydrologicSequence' and 'DownstreamMainPathHydroSeq'
          properties requested (or properties=True).
    tol : float, optional=0.1
        Defines what close is in the case of method == 'geometry'
    
    """
    if method == 'hydroseq':
        return watershed_workflow.river_tree.River.constructRiversByHydroseq(reaches)
    elif method == 'geometry':
        return watershed_workflow.river_tree.River.constructRiversByGeometry(reaches, tol)
    else:
        raise ValueError(
            "Invalid method for making Rivers, must be one of 'hydroseq' or 'geometry'")


