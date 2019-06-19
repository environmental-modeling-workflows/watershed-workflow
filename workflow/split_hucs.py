"""A module for working with multi-polys, a MultiLine that together forms a Polygon"""

import shapely.geometry
import shapely.ops

import workflow.utils

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
    def __init__(self, shapes):
        # all shapes are stored as a collection of collections of segments
        self.segments = HandledCollection() # stores segments

        # Intersect and split the HUCs into unique segments.  Every
        # segment in segments is referenced exactly once in either boundaries
        # or intersections.
        self.boundaries = HandledCollection() # stores handles into segments
        self.intersections = HandledCollection() # stores handles into segments

        # initialize
        uniques, intersections = intersect_and_split(shapes)

        boundary_gon = [HandledCollection() for i in range(len(shapes))]
        for i,u in enumerate(uniques):
            if u is None:
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
                if inter is None:
                    pass
                elif type(inter) is shapely.geometry.LineString:
                    #print("Adding linestring intersection")
                    handle = self.segments.add(inter)
                    ihandle = self.intersections.add(HandledCollection([handle,]))
                    intersection_gon[i].add(ihandle)
                    intersection_gon[j].add(ihandle)
                elif type(inter) is shapely.geometry.MultiLineString:
                    assert(False)
                    handles = self.segments.add_many(inter)
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
        return shapely.geometry.Polygon(ml)

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

    def __len__(self):
        return len(self.gons)


def simplify(hucs, tol=0.1):
    """Simplify, IN PLACE, all segments in the polygon representation."""
    for i,seg in hucs.segments.items():
        hucs.segments[i] = seg.simplify(tol)

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
    uniques = [shapely.geometry.LineString(list(sh.boundary.coords)) for sh in list_of_shapes]

    for i, s1 in enumerate(list_of_shapes):
        for j, s2 in enumerate(list_of_shapes):
            if i != j and s1.intersects(s2):
                inter = s1.intersection(s2)
                if type(inter) is shapely.geometry.MultiLineString:
                    inter = shapely.ops.linemerge(inter)
                elif type(inter) is shapely.geometry.LineString:
                    pass
                elif type(inter) is shapely.geometry.Point:
                    continue
                else:
                    raise RuntimeError("Invalid type of intersection: %r"%type(inter))
                
                uniques[i] = uniques[i].difference(inter)

                # only save once!
                if i > j:
                    intersections[i][j] = inter

    # merge uniques, as we have a bunch of segments.
    for i,u in enumerate(uniques):
        if type(u) is shapely.geometry.MultiLineString:
            uniques[i] = shapely.ops.linemerge(uniques[i])

    uniques_r = [None,]*len(uniques)
    for i,u in enumerate(uniques):
        if type(u) is not shapely.geometry.GeometryCollection:
            uniques_r[i] = u
    return uniques_r, intersections
