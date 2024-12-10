"""A module for working with multi-polys, a MultiLine that together forms a Polygon"""

import logging
import numpy as np
import collections
import copy
import itertools
from matplotlib import pyplot as plt

import shapely.geometry
import shapely.ops
import shapely.errors

import watershed_workflow.utils

_abs_tol = 1
_rel_tol = 1.e-5


class HandledCollection:
    """A collection of of objects and handles for those objects.

    Semantics of this are a bit odd -- it is somewhat like a list and
    somewhat like a dict.
    """
    def __init__(self, *args):
        """Create the HandledCollection

        May be called with 0, 1, or 2 args:

        - If 0, an empty HandledCollection is created.
        - If 1, the argument is the handled objects, and handles are
          generated on the fly.
        - If 2, both handles (which must be unique) and objects are
          provided, in that order.
        """        
        self._store = dict()
        self._key = 0

        if len(args) == 2:
            handles = args[0]
            objs = args[1]
            for h,o in zip(handles, objs):
                self[h] = o

        elif len(args) == 1:
            self.extend(args[0])

        elif len(args) > 2:
            raise RuntimeError("HandledCollection takes 0, 1 or 2 arguments")

    def __getitem__(self, key):

        """Get an object"""
        return self._store[key]

    def __setitem__(self, key, val):
        """Set an object"""
        self._store[key] = val

    def append(self, value):
        """Adds a object, returning a handle to that object"""
        self._store[self._key] = value
        ret = self._key
        self._key += 1
        return ret

    def extend(self, values):
        """Add many objects, returning a list of handles."""
        return [self.append(v) for v in values]

    def pop(self, key):
        """Removes a handle and its object."""
        return self._store.pop(key)

    def __iter__(self):
        """Generator for the collection."""
        for v in self._store.values():
            yield v

    def __len__(self):
        return len(self._store)

    def handles(self):
        """Generator for handles"""
        for k in self._store.keys():
            yield k

    def items(self):
        for it in self._store.items():
            yield it


class SplitHUCs:
    """Class for dealing with the multiple interacting views of HUCs

    Parameters
    ----------
    shapes : list[Polygon]
      The shapes to be split, one per subcatchment to be delineated.
    abs_tol : float
      Distance used in defining small holes at intersections to
      ignore.
    rel_tol : float
      Relative to the shapes area, a tolerance for defining small
      holes on intersections.
    exterior_outlet : np.array((2,))
      Location of the outlet of the entire domain.
    polygon_outlets : np.array((len(shapes), 2))
      Location of the outlets of each polygon.

    
    The resulting class instance includes the following views into
    data:

    linestrings : HandledCollection[LineString]
      unique list of all linestrings, a HandledCollection of LineStrings
    boundaries : HandledCollection[int]
      A HandledCollection of handles into linestrings, identifying those
      linestrings on the outer boundary of the collection.
    intersections : HandledCollection[int]
      A HandledCollection of handles into linestrings, identifying those
      linestrings on the shared, inner boundaries.
    gons : list[HandledCollection[int], HandledCollection[int]]
      One per polygon provided, a pair of HandledCollections,
      identifying the collection of handles into intersetctions and
      boudaries that make up thos polygon.

    """
    def __init__(self,
                 df,
                 abs_tol=_abs_tol,
                 rel_tol=_rel_tol,
                 exterior_outlet=None):
        self.df = df

        # all shapes are stored as a collection of collections of linestrings
        self.linestrings = HandledCollection()  # stores linestrings

        # Intersect and split the HUCs into unique linestrings.  Every
        # linestring in linestrings is referenced exactly once in either boundaries
        # or intersections.
        self.boundaries = HandledCollection()  # stores handles into linestrings
        self.intersections = HandledCollection()  # stores handles into linestrings

        # save the exterior outlet
        if exterior_outlet is not None:
            assert type(exterior_outlet) is shapely.geometry.Point
        self.exterior_outlet = exterior_outlet

        # initialize
        shapes = df['geometry']
        assert (all(isinstance(poly, shapely.geometry.Polygon) for poly in shapes))
        shapes = partition(shapes, abs_tol, rel_tol)
        assert (all(isinstance(poly, shapely.geometry.Polygon) for poly in shapes))
        uniques, intersections = intersectAndSplit(shapes)

        boundary_gon = [HandledCollection() for i in range(len(shapes))]
        for i, u in enumerate(uniques):
            if watershed_workflow.utils.isEmpty(u):
                pass
            elif type(u) is shapely.geometry.LineString:
                handle = self.linestrings.append(u)
                bhandle = self.boundaries.append(HandledCollection([handle, ]))
                boundary_gon[i].append(bhandle)
            elif type(u) is shapely.geometry.MultiLineString:
                handles = self.linestrings.extend(u.geoms)
                bhandles = self.boundaries.extend([HandledCollection([h, ]) for h in handles])
                boundary_gon[i].extend(bhandles)
            else:
                raise RuntimeError(
                    "Uniques from intersectAndSplit is not None, LineString, or MultiLineString?")

        intersection_gon = [HandledCollection() for i in range(len(shapes))]
        for i in range(len(shapes)):
            for j in range(len(shapes)):
                inter = intersections[i][j]
                if watershed_workflow.utils.isEmpty(inter):
                    pass
                elif type(inter) is shapely.geometry.LineString:
                    #print("Adding linestring intersection")
                    handle = self.linestrings.append(inter)
                    ihandle = self.intersections.append(HandledCollection([handle, ]))
                    intersection_gon[i].append(ihandle)
                    intersection_gon[j].append(ihandle)
                elif type(inter) is shapely.geometry.MultiLineString:
                    handles = self.linestrings.extend(list(inter))
                    ihandles = self.intersections.extend(
                        [HandledCollection([h, ]) for h in handles])
                    intersection_gon[i].extend(ihandles)
                    intersection_gon[j].extend(ihandles)
                else:
                    raise RuntimeError(
                        "Intersections from intersectAndSplit is not None, LineString, or MultiLineString?"
                    )

        # the list of shapes, each entry in the list is a tuple
        self.gons = [(u, i) for u, i in zip(boundary_gon, intersection_gon)]

    @property
    def crs(self):
        return self.df.crs

    def to_crs(self, crs):
        self.df.to_crs(crs)
        obj_handles = list(self.linestrings.handles())
        shapes = list(self.linestrings)
        tmp_df = geopandas.GeoDataFrame({'geometry':shapes}, crs=self.df.crs)
        tmp_df.to_crs(crs, inplace=True)
        self.linestrings = HandledCollection(obj_handles, tmp_df['geometry'])
        self.update()
        
    def update(self):
        """Recomputes all polygons"""
        geom = [self.computePolygon(i) for i in range(len(self))]
        self.df['geometry'] = geom

    def computePolygon(self, i):
        """Construct polygon i and return a copy."""
        linestrings = []
        boundary, inter = self.gons[i]
        for h_boundary in boundary:
            for s in self.boundaries[h_boundary]:
                linestrings.append(self.linestrings[s])

        for h_intersection in inter:
            for s in self.intersections[h_intersection]:
                linestrings.append(self.linestrings[s])

        ml = shapely.ops.linemerge(linestrings)
        try:
            assert (type(ml) is shapely.geometry.LineString)
        except AssertionError:
            for ls in linestrings:
                logging.info(ls)
            raise AssertionError
        poly = shapely.geometry.Polygon(ml)
        return poly

    def polygons(self):
        """Iterate over the polygons."""
        self.update()
        for g in self.df['geometry']:
            yield g

    def to_dataframe(self):
        self.update()
        return self.df

    def plot(self, column=None, fill=False, **kwargs):
        # get marker arguments, popping them from kwargs
        markers = False
        if 'marker' in kwargs:
            markers = True
            markerargs = {'marker' : kwargs.pop('marker')}
            if 'markersize' in kwargs:
                markerargs['s'] = kwargs.pop('markersize')

        # force cycled colors as default, not all blue as default
        if column is None and 'color' not in kwargs:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color = [c for (ind,c) in zip(self.df.index, itertools.cycle(color_cycle))]
            kwargs['color'] = color

        # call the default plotter, which, because HUCs are all
        # Polygons, will always add exactly one collection.
        if fill:
            ax = self.df.plot(**kwargs)
            lc = ax.collections[-1]
            colors = lc.get_facecolors()
        else:
            ax = self.df.boundary.plot(**kwargs)
            lc = ax.collections[-1]
            colors = lc.get_colors()

        if markers:
            # scatter the markers
            for i, linestring in enumerate(self.df.boundary.geometry):
                if len(colors) == 1:
                    color = colors[0]
                else:
                    color = colors[i]
                ax.scatter(linestring.xy[0], linestring.xy[1], color=color, **markerargs)

        return ax
        

            
        
    
    def spines(self):
        """Iterate over spines."""
        for b in self.boundaries:
            yield b
        for i in self.intersections:
            yield i

    @property
    def exterior(self):
        """Construct boundary polygon and return a copy."""
        linestrings = []
        for b in self.boundaries:
            for s in b:
                linestrings.append(self.linestrings[s])
        ml = shapely.ops.linemerge(linestrings)
        if type(ml) is shapely.geometry.LineString:
            return shapely.geometry.Polygon(ml)
        else:
            return shapely.geometry.MultiPolygon([shapely.geometry.Polygon(l) for l in ml])
        
    def deepcopy(self):
        """Return a deep copy"""
        cp = copy.deepcopy(self)
        return cp

    def __len__(self):
        return len(self.gons)


def simplify(hucs, tol=0.1):
    """Simplify, IN PLACE, all linestrings in the polygon representation."""
    for i, linestring in hucs.linestrings.items():
        hucs.linestrings[i] = linestring.simplify(tol)


def removeHoles(polygons, abs_tol=_abs_tol, rel_tol=_rel_tol, remove_all_interior=True):
    """Removes interior small holes between the boundaries of polygons.

    Note this assumes the polygons are mostly disjoint.
    
    """
    polygons = polygons[:]
    logging.info(f'Removing holes on {len(polygons)} polygons')
    assert (all(isinstance(p, shapely.geometry.Polygon) for p in polygons))

    # first remove interior holes
    if remove_all_interior:
        polygons2 = [shapely.geometry.Polygon(p.exterior) for p in polygons]
        for p1, p2 in zip(polygons, polygons2):
            if hasattr(p1, 'properties'):
                p2.properties = p1.properties
        polygons = polygons2
    logging.info(f'  -- removed interior')

    union = shapely.ops.unary_union(polygons)
    logging.info(f'  -- union')

    # -- deal with disjoint sections separately
    if isinstance(union, shapely.geometry.Polygon):
        union = [union, ]
    else:
        # MultiPolygon --> list of polygons
        union = [p for p in union]

    logging.info(f'Parsing {len(union)} components for holes')
    big_holes = []
    for part in union:
        # find all holes
        for hole in part.interiors:
            hole = shapely.geometry.Polygon(hole)
            if hole.area > 0:
                if hole.area < (abs_tol**2) or hole.area < rel_tol * part.area:
                    # give it to someone, anyone, doesn't matter who
                    logging.info(f'Found a little hole: area = {hole.area} at {hole.centroid}')
                    try:
                        i, poly = next(
                            (i, poly) for (i, poly) in enumerate(polygons)
                            if watershed_workflow.utils.non_point_intersection(poly, hole))
                        logging.debug(f'      placing in shape {i}')
                        polygons[i] = poly.union(hole)
                        if hasattr(poly, 'properties'):
                            polygons[i].properties = poly.properties
                    except StopIteration:
                        pass

                else:
                    logging.info(f'Found a big hole: area = {hole.area}, leaving it alone...')
                    big_holes.append(hole)

    logging.info(f'  -- complete')
    return polygons, big_holes


def partition(list_of_shapes, abs_tol=_abs_tol, rel_tol=_rel_tol):
    """Given a list of shapes which mostly share boundaries, make sure
    they partition the space.  Often HUC boundaries have minor
    overlaps and underlaps -- here we try to account for wiggles.

    Modifies the list.

    """
    # deal with overlaps
    for i in range(len(list_of_shapes)):
        s1 = list_of_shapes[i]
        for j in range(i + 1, len(list_of_shapes)):
            s2 = list_of_shapes[j]

            try:
                if watershed_workflow.utils.isVolumetricIntersection(s1, s2):
                    props = s2.properties
                    s2 = s2.difference(s1)
                    if isinstance(s2, shapely.geometry.base.BaseMultipartGeometry):
                        s2 = biggest(s2)
                    list_of_shapes[j] = s2
                    list_of_shapes[j].properties = props
            except shapely.errors.TopologicalError as err:
                raise shapely.errors.TopologicalError(f'When intersection polygons {i} and {j}: '
                                                      + str(err))

    # remove holes
    list_of_shapes, holes = removeHoles(list_of_shapes, abs_tol, rel_tol)
    return list_of_shapes


def intersectAndSplit(list_of_shapes):
    """Given a list of shapes which share boundaries (i.e. they partition
    some space), return a compilation of their linestrings.

    Parameters
    ----------
    list_of_shapes : list[shapely.geometry.Polygon]
      The polygons to intersect and split, of length N.

    Returns
    -------
    uniques : list[None | shapely.geometry.LineString | shapely.geometry.MultiLineString]
      An N-length-list of the entities describing the exterior boundary.
    intersections : list[list[None | shapely.geometry.LineString | shapely.geometry.MultiLineString]]
      An NxN list of lists of the entities describing the interior boundary.

    """
    intersections = [[None for i in range(len(list_of_shapes))] for j in range(len(list_of_shapes))]
    uniques = [shapely.geometry.LineString(list(sh.exterior.coords)) for sh in list_of_shapes]

    for i, s1 in enumerate(list_of_shapes):
        for j, s2 in enumerate(list_of_shapes):
            if i != j and watershed_workflow.utils.isNonPointIntersection(s1, s2):
                inter = s1.intersection(s2)

                if type(inter) is shapely.geometry.MultiLineString:
                    inter = shapely.ops.linemerge(inter)

                if type(inter) is shapely.geometry.GeometryCollection:

                    parts_lines = [l for l in inter if isinstance(l, shapely.geometry.LineString)]
                    parts_points = [p for p in inter if isinstance(p, shapely.geometry.Point)]
                    parts_polys = [p for p in inter if isinstance(p, shapely.geometry.Polygon)]

                    mls = shapely.geometry.MultiLineString(parts_lines)
                    left_over_polys = []
                    for poly in parts_polys:
                        mps = poly.intersection(mls)

                        # print(mps)
                        assert (isinstance(mps, shapely.geometry.MultiPoint))
                        assert (len(mps) == 2)
                        parts_lines.append(
                            shapely.geometry.LineString([mps[0].coords[0], mps[1].coords[0]]))
                    inter = shapely.ops.linemerge(parts_lines)
                    # if (isinstance(mps, shapely.geometry.MultiPoint)) and (len(mps) == 2):
                    #     parts_lines.append(shapely.geometry.LineString([mps[0].coords[0], mps[1].coords[0]]))
                    # else:
                    #     left_over_polys.append(poly)
                    # mls_poly = shapely.geometry.MultiLineString([poly.exterior for poly in left_over_polys])
                    # inter_with_polyrings =  shapely.geometry.MultiLineString(parts_lines+list(mls_poly))
                    # inter = shapely.ops.linemerge(inter_with_polyrings)

                if type(inter) is shapely.geometry.GeometryCollection or \
                   type(inter) is shapely.geometry.MultiLineString:

                    logging.info(
                        f'HUC intersection yielded collection of odd types: {set(type(i) for i in inter)}'
                    )
                    err = RuntimeError('HUC intersection yielded collection of odd types')
                    err.polys = list_of_shapes
                    err.i_p1 = i
                    err.p1 = s1
                    err.i_p2 = j
                    err.p2 = s2
                    err.inter = inter
                    raise err

                if type(inter) is not shapely.geometry.LineString:

                    logging.info('Hopefully hole in HUC intersection: ({},{}) = {}'.format(
                        i, j, type(inter)))

                if type(inter) is not shapely.geometry.LineString and \
                   type(inter) is not shapely.geometry.MultiLineString:
                    raise RuntimeError('things are breaking...')

                diff = uniques[i].difference(s2)
                if type(diff) is shapely.geometry.MultiLineString:
                    diff = shapely.ops.linemerge(diff)
                uniques[i] = diff

                # only save once!
                if i > j:
                    intersections[i][j] = inter

    # merge uniques, as we have a bunch of linestrings.
    for i, u in enumerate(uniques):
        if type(u) is shapely.geometry.MultiLineString:
            uniques[i] = shapely.ops.linemerge(uniques[i])

    uniques_r = [None, ] * len(uniques)
    for i, u in enumerate(uniques):
        if not watershed_workflow.utils.isEmpty(u):
            uniques_r[i] = u
    return uniques_r, intersections


def computeNonOverlappingPolygons(polys,
                                  abs_tol=_abs_tol,
                                  rel_tol=_rel_tol,
                                  remove_all_interior=True):
    """Given a list of overlapping contributing area polygons, compute a nonoverlapping set.

    Often we wish to use SplitHucs constructed from the "delta
    contributing area", e.g. the full domain is defined by a single
    polygon, then multiple gages on the river network CAs are used to
    form the subdomains.

    This splits the overlapping contributing areas into the delta
    contributing area for each gage, returning a non-overlapping set
    of polygons, one per original polygon.

    Properties are presereved.

    Parameters
    ----------
    polys : list[shapely.geometry.Polygon]
      The overlapping CAs

    Returns
    -------
    partition : list[shapely.geometry.Polygon]
      The non-overlapping delta CAs.
    holes : list[shapely.geometry.Polygon]
      Holes in the partition that are bigger than the tolerance.

    """
    assert (all(isinstance(p, shapely.geometry.Polygon) for p in polys))
    assert (all(hasattr(p, 'properties') for p in polys))

    sorted_polys = sorted(polys, key=lambda a: a.area, reverse=True)

    # form the tree

    logging.info(f'Create {len(roots)} roots')

    def print(r, ntabs):
        logging.info(' '*ntabs + f'node {r.properties["ID"]}')
        for n in r.children:
            print(n, ntabs + 1)

    for i, root in enumerate(roots):
        logging.info(f'Root {i}:')
        print(root, 1)

    assert (all(hasattr(node, 'properties') for root in roots for node in root.preOrder()))

    # now, at each level, subtract all the containing children.  note
    # we want to work down the tree here
    big_holes = []
    for root in roots:
        for node in root.preOrder():
            if len(node.children) > 0:
                logging.info('First Remove Holes')
                child_polys, holes = removeHoles([c.poly for c in node.children], abs_tol, rel_tol,
                                                 remove_all_interior)
                big_holes.extend(holes)
                upstream = shapely.ops.unary_union(child_polys + holes)
                assert (isinstance(node.poly, shapely.geometry.Polygon))
                node.poly = node.poly.difference(upstream)
                if isinstance(node.poly, shapely.geometry.MultiPolygon):
                    node.poly = biggest(list(node.poly))
                assert (isinstance(node.poly, shapely.geometry.Polygon))

    def getPoly(node):
        poly = node.poly
        poly.properties = node.properties
        return poly

    partition = [getPoly(n) for n in root.preOrder()]
    #return partition, big_holes

    logging.info('Second Remove Holes')
    partition, holes = removeHoles(partition, abs_tol, rel_tol, remove_all_interior)
    assert (all(hasattr(p, 'properties') for p in partition))
    return partition, holes


def findBiggest(list_of_shapes):
    return next(reversed(sorted(list_of_shapes, key=lambda a: a.area)))
