"""A module for working with multi-polys, a MultiLine that together forms a Polygon"""

import logging
import numpy as np
import collections
import copy

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
    def __init__(self,
                 shapes,
                 abs_tol=0.,
                 rel_tol=1.e-5,
                 exterior_outlet=None,
                 polygon_outlets=None):
        # all shapes are stored as a collection of collections of segments
        self.segments = HandledCollection()  # stores segments

        # Intersect and split the HUCs into unique segments.  Every
        # segment in segments is referenced exactly once in either boundaries
        # or intersections.
        self.boundaries = HandledCollection()  # stores handles into segments
        self.intersections = HandledCollection()  # stores handles into segments

        # save the property dictionaries to give back upon request
        self.properties = []
        for s in shapes:
            try:
                self.properties.append(s.properties)
            except AttributeError:
                self.properties.append(None)

        if polygon_outlets is not None:
            assert len(shapes) == len(polygon_outlets)
            for out in polygon_outlets:
                assert type(out) is shapely.geometry.Point
        self.polygon_outlets = polygon_outlets

        if exterior_outlet is not None:
            assert type(exterior_outlet) is shapely.geometry.Point
        self.exterior_outlet = exterior_outlet

        # initialize
        shapes = partition(shapes, abs_tol, rel_tol)
        uniques, intersections = intersect_and_split(shapes)

        boundary_gon = [HandledCollection() for i in range(len(shapes))]
        for i, u in enumerate(uniques):
            if watershed_workflow.utils.is_empty_shapely(u):
                pass
            elif type(u) is shapely.geometry.LineString:
                handle = self.segments.add(u)
                bhandle = self.boundaries.add(HandledCollection([handle, ]))
                boundary_gon[i].add(bhandle)
            elif type(u) is shapely.geometry.MultiLineString:
                handles = self.segments.add_many(u)
                bhandles = self.boundaries.add_many([HandledCollection([h, ]) for h in handles])
                boundary_gon[i].add_many(bhandles)
            else:
                raise RuntimeError(
                    "Uniques from intersect_and_split is not None, LineString, or MultiLineString?")

        intersection_gon = [HandledCollection() for i in range(len(shapes))]
        for i in range(len(shapes)):
            for j in range(len(shapes)):
                inter = intersections[i][j]
                if watershed_workflow.utils.is_empty_shapely(inter):
                    pass
                elif type(inter) is shapely.geometry.LineString:
                    #print("Adding linestring intersection")
                    handle = self.segments.add(inter)
                    ihandle = self.intersections.add(HandledCollection([handle, ]))
                    intersection_gon[i].add(ihandle)
                    intersection_gon[j].add(ihandle)
                elif type(inter) is shapely.geometry.MultiLineString:
                    handles = self.segments.add_many(list(inter))
                    ihandles = self.intersections.add_many(
                        [HandledCollection([h, ]) for h in handles])
                    intersection_gon[i].add_many(ihandles)
                    intersection_gon[j].add_many(ihandles)
                else:
                    raise RuntimeError(
                        "Intersections from intersect_and_split is not None, LineString, or MultiLineString?"
                    )

        # the list of shapes, each entry in the list is a tuple
        self.gons = [(u, i) for u, i in zip(boundary_gon, intersection_gon)]

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
        assert (type(ml) is shapely.geometry.LineString)
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

    def deep_copy(self):
        """Return a deep copy"""
        cp = copy.deepcopy(self)
        return cp

    def __len__(self):
        return len(self.gons)


def simplify(hucs, tol=0.1):
    """Simplify, IN PLACE, all segments in the polygon representation."""
    for i, seg in hucs.segments.items():
        hucs.segments[i] = seg.simplify(tol)


def partition(list_of_shapes, abs_tol=1.0, rel_tol=1.e-3):
    """Given a list of shapes which mostly share boundaries, make sure they
    partition the space.  Often HUC boundaries have minor overlaps and
    underlaps -- here we try to account for wiggles."""
    # deal with overlaps
    for i in range(len(list_of_shapes)):
        for j in range(i + 1, len(list_of_shapes)):
            s1 = list_of_shapes[i]
            s2 = list_of_shapes[j]

            s2 = s2.buffer(abs_tol)
            if watershed_workflow.utils.intersects(s1, s2):
                s2 = s2.difference(s1)
                list_of_shapes[j] = s2.difference(s1)

    # remove holes
    union = shapely.ops.unary_union(list_of_shapes)
    assert (type(union) is shapely.geometry.Polygon)

    # -- deal with disjoint sections separately
    if type(union) is shapely.geometry.Polygon:
        union = [union, ]

    for part in union:
        # find all holes
        for hole in part.interiors:
            hole = shapely.geometry.Polygon(hole)
            if hole.area < abs_tol or hole.area < rel_tol * part.area:
                # give it to someone, anyone, doesn't matter who
                logging.info("Found a little hole: area = {}".format(hole.area))
                for i, poly in enumerate(list_of_shapes):
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
            if i != j and watershed_workflow.utils.non_point_intersection(s1, s2):
                inter = s1.intersection(s2)

                if type(inter) is shapely.geometry.MultiLineString:
                    inter = shapely.ops.linemerge(inter)

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

    # merge uniques, as we have a bunch of segments.
    for i, u in enumerate(uniques):
        if type(u) is shapely.geometry.MultiLineString:
            uniques[i] = shapely.ops.linemerge(uniques[i])

    uniques_r = [None, ] * len(uniques)
    for i, u in enumerate(uniques):
        if not watershed_workflow.utils.is_empty_shapely(u):
            uniques_r[i] = u
    return uniques_r, intersections


def find_outlets_by_crossings(hucs, river, tol=None, debug_plot=False):
    """For each HUC, find all outlets using a river network's crossing points."""
    if tol is None:
        tol = 10
    # next determine the outlet, and all boundary edges within x m of that outlet
    polygons = list(hucs.polygons())
    poly_crossings = []
    for i_sub, poly in enumerate(polygons):
        my_crossings = []
        for reach in river.preOrder():
            if poly.exterior.intersects(reach.segment):
                my_crossings.append(poly.exterior.intersection(reach.segment))

        # cluster my_crossings to make sure that multiple crossings are only counted once
        my_crossing_centroids = []
        for crossing in my_crossings:
            my_crossing_centroids.append([crossing.centroid.xy[0][0], crossing.centroid.xy[1][0]])
        my_crossing_centroids = np.array(my_crossing_centroids)
        if len(my_crossing_centroids) > 1:
            clusters, cluster_centroids = watershed_workflow.utils.cluster(
                my_crossing_centroids, tol)
        else:
            cluster_centroids = my_crossing_centroids
        poly_crossings.append(cluster_centroids)

    logging.info("Crossings by Polygon:")
    for i, c in enumerate(poly_crossings):
        logging.info(f'  Polygon {i}')
        for p in c:
            logging.info(f'    crossing: {p}')

    # unravel the clusters
    all_crossings = [c for p in poly_crossings for c in p]

    # cluster crossings that are within tolerance across polygons
    crossings_clusters_indices, crossings_clusters_centroids = \
        watershed_workflow.utils.cluster(all_crossings, tol)

    # now group cluster ids by polygon and polygon ids by cluster
    poly_cluster_indices = dict()
    cluster_poly_indices = collections.defaultdict(list)
    lcv = 0
    for lcv_poly, pc in enumerate(poly_crossings):
        my_inds = []
        for c in pc:
            my_inds.append(crossings_clusters_indices[lcv])
            lcv += 1
        poly_cluster_indices[lcv_poly] = my_inds
        for ci in my_inds:
            cluster_poly_indices[ci].append(lcv_poly)

    # assert equivalent
    for pi, clusters in poly_cluster_indices.items():
        for ci in clusters:
            assert (pi in cluster_poly_indices[ci])
    for ci, polys in cluster_poly_indices.items():
        for pi in polys:
            assert (ci in poly_cluster_indices[pi])

    # create a tree, recursively finding all polygons with only
    # one crossing -- this must be an outlet -- then removing it
    # from the list, hopefully leaving a downstream polygon with
    # only one outlet.  This must be done N iterations, where N is
    # the maximal number of polygons crossed from 0th order to
    # maximal order.
    logging.info('Constructing outlet list')
    outlets = dict()
    inlets = collections.defaultdict(list)
    itercount = 0
    done = False
    while not done:
        logging.info(f'Iteration = {itercount}')
        logging.info(f'-----------------')
        new_outlets = dict()

        # look for polygons with only one crossing -- this must be an outlet.
        for pi, clusters in poly_cluster_indices.items():
            if len(clusters) == 1 and pi not in outlets:
                # only one crossing cluster, this is the outlet
                cluster_id = clusters[0]
                new_outlets[pi] = cluster_id
                cluster_poly_indices[cluster_id].remove(pi)
                logging.info(
                    f' poly outlet {pi} : {cluster_id}, {crossings_clusters_centroids[cluster_id]}')
                last_outlet = cluster_id
                last_outlet_poly = pi

        # look for clusters with only one poly -- this must be an inlet
        to_remove = []
        for ci, polys in cluster_poly_indices.items():
            if len(polys) == 1:
                poly_id = polys[0]
                poly_cluster_indices[poly_id].remove(ci)
                logging.info(f' poly inlet {poly_id} : {ci}, {crossings_clusters_centroids[ci]}')
                to_remove.append(ci)
                inlets[poly_id].append(ci)
        for ci in to_remove:
            cluster_poly_indices.pop(ci)

        if debug_plot and len(new_outlets) > 0:
            fig, ax = watershed_workflow.plot.get_ax(None)
            watershed_workflow.plot.shplys(polygons, None, color='k', ax=ax)
            watershed_workflow.plot.rivers([river, ], None, color='b', ax=ax)
            for pi, ci in outlets.items():
                outlet = crossings_clusters_centroids[ci]
                ax.scatter([outlet[0], ], [outlet[1], ], s=100, c='b', marker='o')
            for pi, ci in new_outlets.items():
                outlet = crossings_clusters_centroids[ci]
                ax.scatter([outlet[0], ], [outlet[1], ], s=100, c='r', marker='o')
            for ci in range(len(crossings_clusters_centroids)):
                if ci not in outlets.values() and ci not in new_outlets.values():
                    crossing = crossings_clusters_centroids[ci]
                    ax.scatter([crossing[0], ], [crossing[1], ], s=100, c='k', marker='o')
            from matplotlib import pyplot as plt
            ax.set_title(f'Outlets after iteration {itercount}')
            plt.show()

        outlets.update(new_outlets)
        itercount += 1
        done = itercount > 50 or len(outlets) == len(polygons) or len(new_outlets) == 0

    logging.info(
        f'last outlet is {last_outlet} in polygon {last_outlet_poly} at {crossings_clusters_centroids[last_outlet]}'
    )

    # create the output
    outlet_locs = {}
    inlet_locs = {}
    for pi, ci in outlets.items():
        outlet = crossings_clusters_centroids[ci]
        outlet_locs[pi] = shapely.geometry.Point(outlet[0], outlet[1])
    for pi, cis in inlets.items():
        my_inlet_locs = []
        for ci in cis:
            inlet = crossings_clusters_centroids[ci]
            my_inlet_locs.append(shapely.geometry.Point(inlet[0], inlet[1]))
        inlet_locs[pi] = my_inlet_locs

    last_outlet_p = crossings_clusters_centroids[last_outlet]
    last_outlet_loc = shapely.geometry.Point(last_outlet_p[0], last_outlet_p[1])

    hucs.exterior_outlet = last_outlet_loc
    hucs.polygon_outlets = outlet_locs


def find_outlets_by_elevation(hucs, crs, elev_raster, elev_raster_profile):
    """Find outlets by the minimum elevation on the boundary."""
    import watershed_workflow
    exterior = hucs.exterior().exterior
    mesh_points = np.array([exterior.coords])[0, :, :]
    mesh_points = watershed_workflow.elevate(mesh_points, crs, elev_raster, elev_raster_profile)
    i = np.argmin(mesh_points[:, 2])
    hucs.exterior_outlet = shapely.geometry.Point(mesh_points[i, 0], mesh_points[i, 1])

    outlets = []
    for poly in hucs.polygons():
        mesh_points = np.array([poly.exterior.coords])[0, :, :]
        mesh_points = watershed_workflow.elevate(mesh_points, crs, elev_raster, elev_raster_profile)
        i = np.argmin(mesh_points[:, 2])
        outlets.append(shapely.geometry.Point(mesh_points[i, 0], mesh_points[i, 1]))
    hucs.polygon_outlets = outlets


def find_outlets_by_hydroseq(hucs, river, tol=0):
    """Find outlets using the HydroSequence VAA of NHDPlus.

    Finds the minimum hydroseq reach in each HUC, and intersects that
    with the boundary to find the outlet.
    """
    polygons = list(hucs.polygons())
    polygon_outlets = [None for poly in hucs.polygons()]

    # iterate over the reaches, sorted by hydrosequence, looking for
    # the first one that intersects the polygon boundary.
    assert (river.is_hydroseq_consistent())
    reaches = sorted(river.preOrder(), key=lambda r: r.properties['HydrologicSequence'])
    if tol > 0:
        reaches = [r.segment.buffer(tol) for r in reaches]
    else:
        reaches = [r.segment for r in reaches]
    first = True

    poly_ids = [(i, poly) for (i, poly) in enumerate(polygons)]
    for lcv, reach in enumerate(reaches):
        try:
            j, (poly_i, poly) = next((j,(i,poly)) for (j,(i,poly)) in enumerate(poly_ids) \
                                     if poly.intersects(reach))
        except StopIteration:
            continue
        else:
            # find the intersection
            logging.debug(f'hydroseq {lcv} is a match for polygon {poly_i}')
            intersect = poly.exterior.intersection(reach)
            if intersect.is_empty:
                # find the nearest point instead
                intersect = shapely.ops.nearest_points(poly.exterior, reach)[0]
            else:
                intersect = intersect.centroid

            if first:
                hucs.exterior_outlet = intersect
                first = False
            polygon_outlets[poly_i] = intersect
            poly_ids.pop(j)
        if len(poly_ids) == 0:
            break

    hucs.polygon_outlets = polygon_outlets
