"""A module for working with multi-polys, a MultiLine that together forms a Polygon"""

import logging

import shapely.geometry
import shapely.ops

import watershed_workflow.utils

class HandledCollection:
    """A collection of of objects and handles for those objects."""
    def __init__(self, objs=None):
        """Create"""
        self._store = dict()
        self._key = 0

        if objs is not None:
            self.add_many(objs)

    def __getitem__(self, key):
        """Get an object"""
        return self._store[key]

    def __setitem__(self, key, val):
        """Set an object"""
        self._store[key] = val
    
    def add(self, value):
        """Adds a object, returning a handle to that object"""
        self._store[self._key] = value
        ret = self._key
        self._key += 1
        return ret

    def add_many(self, values):
        """Add many objects, returning a list of handles."""
        return [self.add(v) for v in values]
    
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

    def keys(self):
        for it in self._store.keys():
            yield it
        
    def items(self):
        for it in self._store.items():
            yield it
            
class SplitHUCs:
    """Class for dealing with the multiple interacting views of HUCs

    Includes the following views into data:

    segments            | Unique list of segments -- the only actual data, 
                        | a HandledCollection of LineStrings
    boundaries          | A HandledCollection of handles into segments 
                        | describing the outer boundary of the collection.
    intersections       | A HandledCollection of handles into segments 
                        | describing the internal boundaries of the 
                        | collection.
    gons                | A HandledCollection of two-tuples of handles
                        | the first into boundaries and the second into 
                        | intersections -- which together form the polygon.

    """
    def __init__(self, shapes, abs_tol=0., rel_tol=1.e-5):
        # all shapes are stored as a collection of collections of segments
        self.segments = HandledCollection() # stores segments

        # Intersect and split the HUCs into unique segments.  Every
        # segment in segments is referenced exactly once in either boundaries
        # or intersections.
        self.boundaries = HandledCollection() # stores handles into segments
        self.intersections = HandledCollection() # stores handles into segments

        # save the property dictionaries to give back upon request
        self.properties = []
        for s in shapes:
            try:
                self.properties.append(s.properties)
            except AttributeError:
                self.properties.append(None)

        # initialize
        shapes = partition(shapes, abs_tol, rel_tol)
        uniques, intersections = intersect_and_split(shapes)

        boundary_gon = [HandledCollection() for i in range(len(shapes))]
        for i,u in enumerate(uniques):
            if watershed_workflow.utils.empty_shapely(u):
                pass
            elif type(u) is shapely.geometry.LineString:
                handle = self.segments.add(u)
                bhandle = self.boundaries.add(HandledCollection([handle,]))
                boundary_gon[i].add(bhandle)
            elif type(u) is shapely.geometry.MultiLineString:
                handles = self.segments.add_many(u)
                bhandles = self.boundaries.add_many([HandledCollection([h,]) for h in handles])
                boundary_gon[i].add_many(bhandles)
            else:
                raise RuntimeError("Uniques from intersect_and_split is not None, LineString, or MultiLineString?")

        intersection_gon = [HandledCollection() for i in range(len(shapes))]
        for i in range(len(shapes)):
            for j in range(len(shapes)):
                inter = intersections[i][j]
                if watershed_workflow.utils.empty_shapely(inter):
                    pass
                elif type(inter) is shapely.geometry.LineString:
                    #print("Adding linestring intersection")
                    handle = self.segments.add(inter)
                    ihandle = self.intersections.add(HandledCollection([handle,]))
                    intersection_gon[i].add(ihandle)
                    intersection_gon[j].add(ihandle)
                elif type(inter) is shapely.geometry.MultiLineString:
                    handles = self.segments.add_many(list(inter))
                    ihandles = self.intersections.add_many([HandledCollection([h,]) for h in handles])
                    intersection_gon[i].add_many(ihandles)
                    intersection_gon[j].add_many(ihandles)
                else:
                    raise RuntimeError("Intersections from intersect_and_split is not None, LineString, or MultiLineString?")

        # the list of shapes, each entry in the list is a tuple
        self.gons = [(u,i) for u,i in zip(boundary_gon, intersection_gon)]

        

    def polygon(self, i):
        """Construct polygon i and return a copy."""
        segs = []
        boundary, inter = self.gons[i]
        for h_boundary in boundary:
            for s in self.boundaries[h_boundary]:
                segs.append(self.segments[s])

        for h_intersection in inter:
            for s in self.intersections[h_intersection]:
                segs.append(self.segments[s])

        ml = shapely.ops.linemerge(segs)
        assert(type(ml) is shapely.geometry.LineString)
        poly = shapely.geometry.Polygon(ml)
        poly.properties = self.properties[i]
        return poly

    def polygons(self):
        """Iterate over the polygons."""
        for i in range(len(self.gons)):
            yield self.polygon(i)

    def spines(self):
        """Iterate over spines."""
        for b in self.boundaries:
            yield b
        for i in self.intersections:
            yield i

    def exterior(self):
        """Construct boundary polygon and return a copy."""
        segs = []
        for b in self.boundaries:
            for s in b:
                segs.append(self.segments[s])
        ml = shapely.ops.linemerge(segs)
        if type(ml) is shapely.geometry.LineString:
            return shapely.geometry.Polygon(ml)
        else:
            return shapely.geometry.MultiPolygon([shapely.geometry.Polygon(l) for l in ml])

    def __len__(self):
        return len(self.gons)


def simplify(hucs, tol=0.1):
    """Simplify, IN PLACE, all segments in the polygon representation."""
    for i,seg in hucs.segments.items():
        hucs.segments[i] = seg.simplify(tol)


def partition(list_of_shapes, abs_tol=1.0, rel_tol=1.e-3):
    """Given a list of shapes which mostly share boundaries, make sure they
    partition the space.  Often HUC boundaries have minor overlaps and
    underlaps -- here we try to account for wiggles."""
    # deal with overlaps
    for i in range(len(list_of_shapes)):
        for j in range(i+1,len(list_of_shapes)):
            s1 = list_of_shapes[i]
            s2 = list_of_shapes[j]

            s2 = s2.buffer(abs_tol)
            if watershed_workflow.utils.intersects(s1, s2):
                s2 = s2.difference(s1)
                list_of_shapes[j] = s2.difference(s1)

    # remove holes
    union = shapely.ops.cascaded_union(list_of_shapes)
    assert(type(union) is shapely.geometry.Polygon)

    # -- deal with disjoint sections separately
    if type(union) is shapely.geometry.Polygon:
        union = [union,]

    for part in union:
        # find all holes
        for hole in part.interiors:
            hole = shapely.geometry.Polygon(hole)
            if hole.area < abs_tol or hole.area < rel_tol * part.area:
                # give it to someone, anyone, doesn't matter who
                logging.info("Found a little hole: area = {}".format(hole.area))
                for i,poly in enumerate(list_of_shapes):
                    if watershed_workflow.utils.non_point_intersection(poly, hole):
                        logging.info('touches {}'.format(i))
                        poly = poly.union(hole)
                        list_of_shapes[i] = poly
                        break
            else:
                logging.info("Found a big hole: area = {}".format(hole.area))
                
    return list_of_shapes
        
def intersect_and_split(list_of_shapes):
    """Given a list of shapes which share boundaries (i.e. they partition
    some space), return a compilation of their segments.

    Given a list of shapes of length N, returns:

    uniques             | An N-length-list of either None, LineString,
                        |  or MultiLineString, describing the exterior 
                        |  boundary
    intersections       | A NxN list of lists of either None, LineString, 
                        |  or MultiLineString, describing the interior
                        |  boundary.
    """
    intersections = [[None for i in range(len(list_of_shapes))] for j in range(len(list_of_shapes))]
    uniques = [shapely.geometry.LineString(list(sh.exterior.coords)) for sh in list_of_shapes]

    for i, s1 in enumerate(list_of_shapes):
        for j, s2 in enumerate(list_of_shapes):
            if i != j and watershed_workflow.utils.non_point_intersection(s1,s2):
                inter = s1.intersection(s2)

                if type(inter) is shapely.geometry.MultiLineString:
                    inter = shapely.ops.linemerge(inter)

                if type(inter) is not shapely.geometry.LineString:
                    logging.info('Hopefully hole in HUC intersection: ({},{}) = {}'.format(i,j,type(inter)))

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

    # merge uniques, as we have a bunch of segments.
    for i,u in enumerate(uniques):
        if type(u) is shapely.geometry.MultiLineString:
            uniques[i] = shapely.ops.linemerge(uniques[i])

    uniques_r = [None,]*len(uniques)
    for i,u in enumerate(uniques):
        if not watershed_workflow.utils.empty_shapely(u):
            uniques_r[i] = u
    return uniques_r, intersections
