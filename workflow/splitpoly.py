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
        """Get a object"""
        return self._store[key]

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
        self._store.pop(key)

    def __iter__(self):
        """Generator for the collection."""
        for v in self._store.values():
            yield v
        

            
class HUCs:
    """Class for dealing with the multiple interacting views of HUCs

    Includes the following views into data:

    segs                | Unique list of segments -- the only actual data, 
                        | a HandledCollection of LineStrings
    boundaries          | A HandledCollection of handles into segs 
                        | describing the outer boundary of the collection.
    intersections       | A HandledCollection of handles into segs 
                        | describing the internal boundaries of the 
                        | collection.
    gons                | A HandledCollection of two-tuples of handles
                        | the first into boundaries and the second into 
                        | intersections -- which together form the polygon.

    """
    def __init__(self, shapes):
        # all shapes are stored as a collection of collections of segments
        self.segs = HandledCollection() # stores segments

        # intersect and split the HUCs into unique segments
        self.boudaries = HandledCollection() # stores handles into segs
        self.intersections = HandledCollection() # stores handles into segs

        # initialize
        boundary_gon = []
        for u in uniques:
            if u is None:
                boundary_gon.append(HandledCollection())
            elif type(u) is shapely.geometry.LineString:
                handle = self.segs.add(u)
                bhandle = self.boundaries.add(HandledCollection([handle,]))
                boundary_gon.append(HandledCollection([bhandle,]))
            elif type(u) is shapely.geometry.MultiLineString:
                handles = self.segs.add_many(u)
                bhandles = self.boundaries.add_many([HandledCollection([h,]) for h in handles])
                boundary_gon.append(HandledCollection([bhandles,]))
            else:
                raise RuntimeError("Uniques from intersect_and_split is not None, LineString, or MultiLineString?")

        intersection_gon = [HandledCollection() for i in range(len(shapes))]
        for i in range(len(shapes)):
            for j in range(len(shapes)):
                inter = intersections[i][j]
                if inter is None:
                    pass
                elif type(inter) is shapely.geometry.LineString:
                    handle = self.segs.add(inter)
                    ihandle = self.intersections.add(HandledCollection[handle,])
                    intersection_gon[i].add(HandledCollection([ihandle,]))
                    intersection_gon[j].add(HandledCollection([ihandle,]))
                elif type(inter) is shapely.geometry.MultiLineString:
                    handles = self.segs.add_many(inter)
                    ihandles = self.intersections.add_many([HandledCollection([h,]) for h in handles])
                    intersection_gon[i].add_many([ihandles,])
                    intersection_gon[j].add_many([ihandles,])
                else:
                    raise RuntimeError("Intersections from intersect_and_split is not None, LineString, or MultiLineString?")

        # the list of shapes, each entry in the list is a tuple
        self.gons = [(u,i) for u,i in zip(boundary_gon, intersection_gon)]

        
        

def get_polygon(ml):
    """Generates a polygon from the multiline"""
    assert(type(ml) is shapely.geometry.MultiLineString)
    line = shapely.ops.linemerge(ml)
    assert(type(line) is shapely.geometry.LineString)
    return shapely.geometry.Polygon(line)

def split_multiline(ml, seg):
    """Splits a multiline at an intersection"""
    topop = []
    topush = []
    for i, line in enumerate(ml):
        if seg.intersects(line):
            newlines = workflow.utils.split(line, seg)
            topop.append(i)
            topush.extend(newlines)

    new_ml = list(ml)
    for i in reversed(topop):
        new_ml.pop(i)
    new_ml.extend(topush)
    new_ml = shapely.geometry.MultiLineString(new_ml)

    # check
    poly = get_polygon(new_ml)
    return new_ml
