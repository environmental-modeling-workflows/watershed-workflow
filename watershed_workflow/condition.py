import numpy as np
import attr
import sortedcontainers
import logging

@attr.s
class Point:
    """POD struct that stores coords, a np array of length 3 (x,y,z) and neighbors, 
    a list of IDs of neighboring points.
    """
    coords = attr.ib()
    neighbors = attr.ib(factory=set)

def points_from_mesh(m2):
    """Generates a Point dictionary from a surface mesh, for use with fill_pits"""
    points = dict( (i, Point(c)) for (i,c) in enumerate(m2.coords) )
    for conn in m2.conn:
        for c in conn:
            points[c].neighbors.update(conn)
    for i,p in points.items():
        p.neighbors.remove(i)
    return points


def fill_pits1(points, outletID=None):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is the origional, 2-pass algorithm.
    """
    if outletID is None:
        outletID = np.argmin(np.array([points[i].coords[2] for i in range(len(points))]))

    # create a sorted list of elevations, from largest to smallest
    elev = sorted(list(points.items()), key=lambda id_p:-id_p[1].coords[2])

    visited = set([outletID,])
    pits = set()
    waterway = set([outletID,])

    # loop over elevation list from small to large
    while len(elev) != 0:
        current, current_p = elev.pop()
        if current in visited:
            # still in the waterway
            waterway.add(current)
            visited.update(current_p.neighbors)
        else:
            # not in the waterway, add to pits
            pits.add(current)
    
    # post-conditions
    assert(len(pits.union(waterway)) == len(points))

    # loop over waterway and raise up pits as they touch the waterway
    waterway = sorted([ (ID,points[ID]) for ID in waterway], key=lambda id_p:-id_p[1].coords[2])
    while len(waterway) != 0:
        current, current_p = waterway.pop()
        for n in current_p.neighbors:
            if n in pits:
                points[n].coords[2] = max(current_p.coords[2], points[n].coords[2])
                pits.remove(n)
                waterway.append( (n,points[n]) )

    # post-conditions
    assert(len(pits) == 0)
    return


def fill_pits2(points, outletID):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is a refactored, single pass algorithm that leverages a sorted list.
    """

    # create a sorted list of elevations, from largest to smallest
    elev = sortedcontainers.SortedList(list(points.items()), key=lambda id_p:id_p[1].coords[2])
    waterway = set([outletID,])

    # loop over elevation list from small to large
    while len(elev) != 0:
        current, current_p = elev.pop(0)
        if current in waterway:
            # still in the waterway
            waterway.update(current_p.neighbors)
        else:
            # not in the waterway, fill
            ww_neighbors = [n for n in current_p.neighbors if n in waterway]
            if len(ww_neighbors) != 0:
                current_p.coords[2] = min(points[n].coords[2] for n in ww_neighbors)
            else:
                current_p.coords[2] = min(points[n].coords[2] for n in current_p.neighbors)
            
            # push back into elev list with new, higher elevation
            elev.add( (current,current_p) )
    return


def fill_pits3(points, outletID):
    """Conditions a mesh, in place, by removing pits.

    Inputs:
      points    | A dictionary of the form {ID, Point()} 
      outletID  | ID of the outlet

    This is a third algorithm, based on a boundary marching method.
    """
    # Waterway is the list of things that are already conditioned and
    # can be reached.
    waterway = set()

    # Boundary is an elevation-sorted list of (ID, point) tuples which
    # are NOT yet in the waterway, but have a neighbor in the
    # waterway.  Additionally, points in the boundary have been
    # conditioned such that all boundary points must be at least as
    # high as the highest elevation in the waterway.
    boundary = sortedcontainers.SortedList([(outletID,points[outletID]), ], key=lambda id_p:id_p[1].coords[2])
    waterway_max = -1e10

    while len(boundary) > 0:
        # pop the lowest boundary point and stick it in the waterway
        next_p = boundary.pop(0)
        waterway.add(next_p[0])

        # increment the waterway elevation
        assert(next_p[1].coords[2] >= waterway_max)
        waterway_max = next_p[1].coords[2]

        # Insert all neighbors (that aren't in the waterway already)
        # into the boundary, first checking that their elevation is at
        # least as high as the new waterway elevation.  Note that
        # there can be no "alternate" pathway to the waterway, as this
        # pathway would already be in the boundary (somewhere along
        # that pathway) and therefore would have been popped before
        # this path due to our sorted elevation boundary list.
        for n in next_p[1].neighbors:
            if n not in waterway and (n,points[n]) not in boundary:
                points[n].coords[2] = max(points[n].coords[2], waterway_max)
                boundary.add( (n,points[n]) )

    assert(len(waterway) == len(points))
    return
                    

def fill_pits_dual(m2, is_waterbody=None, outlet_edge=None, eps=1e-3):
    """Conditions a dual mesh IN PLACE, ensuring the property that,
    starting with an outlet cell, there is a path to every cell by way
    of faces that is monotonically increasing in elevation (except in
    cells which are a part of waterbodies).

    If the is_waterbody mask is provided, these cells are special
    cells that may be pits -- e.g. lakes, reservoirs, etc.

    Parameters
    ----------
    m2 : mesh.Mesh2D object
      The mesh to condition.
    is_waterbody : np.ndarray(int, shape=ncells), optional
      A boolean/integer mask of length m2.num_cells, with True
      values indicating that the given cell is a part of a waterbody
      that may be a depression.  This is important for
      reservoirs/lakes/etc where bathymetry is known and pits are
      physical.
    outlet_edge : (int,int), optional
      If provided, the point to start conditioning from.  If not
      provided, will use the edge on the boundary of m2 with the
      lowest elevation.
    eps : float, optional=1e-3
      A small vertical displacement for soft tolerances.

    """
    if outlet_edge is None:
        # determine the outlet edge -- the lowest edge point
        boundary_edges = m2.boundary_edges
        outlet_edge = boundary_edges[0]
        be_elev = (m2.coords[outlet_edge[0],2] + m2.coords[outlet_edge[1],2])/2.

        for e in boundary_edges[1:]:
            eh = (m2.coords[e[0],2] + m2.coords[e[1],2])/2.
            if eh < be_elev:
                be_elev = eh
                outlet_edge = e
    outlet_edge = m2.edge_hash(*outlet_edge)

    outlet_cell = m2.edges_to_cells[outlet_edge]
    assert(len(outlet_cell) == 1)
    outlet_cell = outlet_cell[0]

    class Waterway:
        """Waterway is the set of cells that are already conditioned and can be reached."""
        def __init__(self):
            self.cells = set()

            # Waterway edges is the set of edges whose cells are all in waterway
            self.edges = set()
            self.max_z = -1e10

        def add(self, be):
            """Add BoundaryEntry object to the waterway"""
            logging.debug(f"adding cell {be.cell} (z = {be.z})")
            assert(be.cell not in self.cells)
            self.cells.add(be.cell)
            for e in be.edges:
                assert(e not in self.edges)
                self.edges.add(e)
            assert(be.z >= self.max_z)
            self.max_z = be.z
    waterway = Waterway()
    
    class BoundaryEntry:
        """A cell that is not yet in the waterway, but has at least one edge whose other cell is in the waterway."""
        def __init__(self, cell, edges):
            assert(type(cell) is int)
            assert(0 <= cell < m2.num_cells)
            assert(type(edges) is list)
            for e in edges:
                assert(type(e) is tuple)

            self.cell = cell
            self.edges = edges
            self.z = m2.compute_centroid(self.cell)[2]

    boundary = sortedcontainers.SortedList([BoundaryEntry(outlet_cell, [outlet_edge,]),], key=lambda be : be.z)

    # masked cells are always in the boundary, allowing them to be picked up as we reach that elevation.
    if is_waterbody is not None:
        assert(len(is_waterbody) == m2.num_cells)
        masked_cells = [BoundaryEntry(c, list()) for (c, mask) in enumerate(is_waterbody) if mask]
        boundary.update(masked_cells)

    while len(boundary) > 0:
        # pop the lowest boundary cell and stick its edge and cell
        next_be = boundary.pop(0)
        waterway.add(next_be)

        # find all other edges of the cell just added
        for other_e in m2.cell_edges(m2.conn[next_be.cell]):
            if other_e in waterway.edges: continue

            # find the cell on the other side of other_e
            other_e_cells = m2.edges_to_cells[other_e]
            if len(other_e_cells) == 1:
                # boundary edge, add it to the waterway
                assert(next_be.cell == other_e_cells[0])
                waterway.edges.add(other_e)
                continue 

            assert(len(other_e_cells) == 2)
            assert(next_be.cell in other_e_cells)
            if next_be.cell == other_e_cells[0]:
                other_c = other_e_cells[1]
            else:
                assert(next_be.cell == other_e_cells[1])
                other_c = other_e_cells[0]

            # this would break assumption of what it means to be
            # in boundary.
            assert(other_c not in waterway.cells)

            # now we have an other_e, other_c pair to add into
            # boundary.  But first we may need to condition.
            other_c_centroid = m2.compute_centroid(other_c)
            if other_c_centroid[2] < waterway.max_z:
                other_c_nodes = m2.conn[other_c]
                    
                # for this to be possible, there must be at least
                # one free node in the nodes of next_c.  By free,
                # we mean that its elevation can be changed
                # without breaking everything.  This means that
                # neither of that node's edges can be in
                # waterway.edges or boundary.
                #
                # we also need the fixed (non-free) node elevations
                fixed_node_elevs = dict()

                for e in m2.cell_edges(other_c_nodes):
                    if (e == other_e) or (e in waterway.edges) or any((e == i) for be in boundary for i in be.edges):
                        if e[0] not in fixed_node_elevs:
                            fixed_node_elevs[e[0]] = m2.coords[e[0],2]
                        if e[1] not in fixed_node_elevs:
                            fixed_node_elevs[e[1]] = m2.coords[e[1],2]
                free_nodes = [n for n in other_c_nodes if n not in fixed_node_elevs]

                # should not be possible to be both lower
                # elevation and not have a free node, or it would
                # already be in boundary, and therefore have no
                # free nodes
                assert(len(free_nodes) > 0) 

                # calculate the z of the free node required to
                # make the triangle's centroid == waterway_max
                #
                # this formula is likely triangle-only?
                z_free = (waterway.max_z * len(other_c_nodes) - sum(fixed_node_elevs.values())) / len(free_nodes) + eps

                # for now, we'll assume triangular.  I'm not sure
                # what to do if this is bigger than length
                # 1... something like evenly raise up all free
                # nodes?  But with triangles, there can only be
                # one free node so it is easy.
                assert(len(free_nodes) == 1)
                logging.debug(f'  moving z node {free_nodes[0]} from {m2.coords[free_nodes[0],2]} to {z_free}')
                m2.coords[free_nodes[0],2] = z_free

            # now it is conditioned, add it to the boundary
            try:
                # is it already in the boundary?
                other_be = next(be for be in boundary if be.cell == other_c)
            except StopIteration:
                # no, add it
                logging.debug(f'  adding to boundary: edge: {other_e}  cell: {other_c}')
                boundary.add(BoundaryEntry(other_c, [other_e,]))
            else:
                # yes, just add this edge to that entry
                assert(other_e not in other_be.edges)
                other_be.edges.append(other_e)
                        
    # when this is done, all cells should be in waterway
    assert(len(waterway.cells) == m2.num_cells)
    assert(len(waterway.edges) == m2.num_edges)
    return


def fill_pits(mesh, outlet=None, algorithm=3):

    """Condition a 2D mesh, in place.
    
    Starts at outlet, if not provided, this defaults to the lowpoint on the boundary.

    Available algorithms:
     1: original, 2-pass algorithm
     2: refactored single-pass algorithm based on sorted lists
     3: boundary marching method.  Should be fastest, and likely equivalent?
    """
    points_dict = points_from_mesh(mesh)
    if outlet is None:
        boundary_nodes = mesh.boundary_nodes
        outlet = boundary_nodes[np.argmin(mesh.coords[boundary_nodes,2])]

    if algorithm == 1:
        fill_pits1(points_dict, outlet)
    elif algorithm == 2:
        fill_pits2(points_dict, outlet)
    elif algorithm == 3:
        fill_pits3(points_dict, outlet)
    else:
        raise RuntimeError('Unknown algorithm "%r"'%(algorithm))

    mesh.points = np.array([p.coords for p in points_dict.values()])


def smooth(img_in, algorithm='gaussian', **kwargs):
    """Smooths an image according to an algorithm, passing kwargs on to that algorithm."""
    if algorithm == 'gaussian':
        import scipy.ndimage
        if 'method' not in kwargs:
            kwargs['method'] = 'nearest'
        if 'sigma' not in kwargs:
            sigma = 5
        else:
            sigma = kwargs.pop('sigma')
        return scipy.ndimage.gaussian_filter(img_in, sigma, **kwargs)
    else:
        raise ValueError(f'Unknown smoothing algorithm: "{algorithm}"')


def fill_gaps(img_in, nodata=np.nan):
    import scipy.ndimage
    import scipy.interpolate

    if nodata is np.nan:
        mask = ~np.isnan(img_in)
    else:
        mask = ~(img_in == nodata)

    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(img_in.shape[1]), np.arange(img_in.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values, as 1D arrays (in the same order as their coordinates in xym)
    img_in0 = np.ravel(img_in[:,:][mask])

    # interpolator
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, img_in0)

    # interpolate the whole image
    return interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

def condition_river_mesh2(m2, river ,poly_smooth=False, filter_mode=None,use_nhd_elev=False,cut_off_order=0,treat_banks=False ,depress_by=0):
    """ makes the bed profile of the stream smooth, removes hills and ponds
            m2: mesh2D object generated from Watershed Workflow
            river: river_tree object
    """
    import math
    import copy 
    river_corr_ids=[] # collecting IDs of all nodes in the river/stream
    for node in river.preOrder():
        for elem in node.elements:
            for id in elem:
                if id not in river_corr_ids: 
                    river_corr_ids.append(id)
             
    for node in river.preOrder():
        order=node.properties["StreamOrder"]
        if order <= cut_off_order: 
            assert('elev_profile' in node.properties.keys())
            
            # get bed profile
            stream_bed_coords=list(reversed(node.segment.coords)) # node that node_elems are downstream to upstream, while segment coords are upstream to downstream        dists=[math.dist(node.segment.coords[0],point) for point in node.segment.coords]
            dists=[math.dist(stream_bed_coords[0],point) for point in stream_bed_coords]
            elevs=node.properties['elev_profile'][::-1] # this is also reversed
            profile=np.array([dists,elevs]).T


            if use_nhd_elev and 'MaximumElevationSmoothed'in node.segment.properties.keys() and 'MinimumElevationSmoothed' in node.segment.properties.keys():
                    nhd_limits=[node.segment.properties['MinimumElevationSmoothed'],node.segment.properties['MaximumElevationSmoothed']]
            else:
                    nhd_limits=None

            if poly_smooth:
                profile=poly_fit(profile, degree=5) # it is highly recommended to filter after this

            if not filter_mode == None:
                profile=filter_pass(profile, filtermode=filter_mode,nhd_limits=nhd_limits)

            # higher order streams might not need conditioning as river quads will sit in nicely into DEM depressions
            for i, elem in enumerate(node.elements): 
                
                if i ==0: # for the first point
                    m2.coords[elem[0]][2]= m2.coords[elem[-1]][2]=profile[i,1]-depress_by
                    
                # we are assgining elevations only to the upstream points of the river elements
                if len(elem)==3:          
                    m2.coords[elem[1]][2]= profile[i+1,1]-depress_by

                elif len(elem)==4: 
                    m2.coords[elem[1]][2]= m2.coords[elem[2]][2]= profile[i+1,1]-depress_by
                    
                elif len(elem)==5:  
                    m2.coords[elem[1]][2]= m2.coords[elem[2]][2]= m2.coords[elem[3]][2]= profile[i+1,1]-depress_by
                    
                if treat_banks: # this needs to be rewritten
                    bank_node_ids=bank_nodes_from_elem(elem, m2)
                    for node_id in bank_node_ids:
                         if node_id not in river_corr_ids:
                            if m2.coords[node_id][2]<profile[i,1]-depress_by:
                                   m2.coords[node_id][2]= profile[i,1]+0.25 # is 0.25m enough to raise the bank?



# def condition_river_mesh(m2, river, mode="upper", cut_off_order=0, depress_by=0):
#     """ makes the bed profile of the stream smooth, removes hills and ponds
#             m2: mesh2D object generated from Watershed Workflow
#             river: river_tree object
#     """
#     river_corr_ids=[] # collecting IDs of all nodes in the river/stream
#     for node in river.preOrder():
#         for elem in node.elements:
#             for id in elem:
#                 if id not in river_corr_ids: 
#                     river_corr_ids.append(id)
    
#     for node in river.preOrder():
#         order=node.segment.properties["StreamOrder"]
#         profile=get_reach_profile(node, m2)
#         profile_new=condition_reach(profile, mode=mode)

#         if order <= cut_off_order: # if we wish to put control of river order for conditioning. 
#             # higher order streams might not need conditioning as river quads will sit in nicely into DEM depressions
#             for i, elem in enumerate(node.elements):
#                 for j in range(len(elem)):
#                     m2.coords[elem[j]][2]=profile_new[i,1]-depress_by # lower by 1 m
                
#                 bank_node_ids=bank_nodes_from_elem(elem, m2)
#                 for node_id in bank_node_ids:
#                      if node_id not in river_corr_ids:
#                         if m2.coords[node_id][2]<profile_new[i,1]-depress_by:
#                                m2.coords[node_id][2]= profile_new[i,1]+0.25

def get_reach_profile(node,m2):
    """ for a given node segment of a river tree, this function return the profile
        of the stream bed based on the centroid elevation of the quads

        node: node of a river_tree object
        m2: mesh2D object
    """
    from math import dist
    n_quads=len(node.elements); # this includes traingles and pentagons
    profile=np.zeros((n_quads,2))

    for i, elem in enumerate(node.elements):
        if i ==0:
            d=0
            ind=m2.conn.index(node.elements[i])
            centroid=m2.centroids()[ind]
            p=(centroid[0],centroid[1])
        else:
            ind=m2.conn.index(node.elements[i])
            centroid=m2.centroids()[ind]
            p_new=(centroid[0],centroid[1])
            delta=dist(p,p_new)
            d=d+delta
            p=p_new
        profile[i,0]=d
        profile[i,1]=centroid[2]
    return profile
        
# def condition_reach(profile, mode='upper'):
#     """performs a condiotning of the stream bed by using a high-order polynomial and pass for monotonicity enforcement
#     """
#     import copy 
#     import numpy
#     from numpy.polynomial import polynomial as poly

#     profile_new=copy.deepcopy(profile)
#     if not len(profile)==1:
#         p = poly.polyfit(profile[:,0], profile[:,1], deg=5)
#         profile_new[:,1]=poly.polyval(profile_new[:,0],p)
#         if  mode=='upper':
#             for i in range(len(profile_new)-1):
#                 if profile_new[i+1,1] < profile_new[i,1]:
#                     profile_new[i+1,1]=profile_new[i,1]
#         elif mode=='lower':
#             for i in range(len(profile_new)-1, 0,-1):
#                 if profile_new[i-1,1] > profile_new[i,1]:
#                     profile_new[i-1,1] = profile_new[i,1]
#     return profile_new
 

def poly_fit(profile, degree=5):
    profile_new=copy.deepcopy(profile)
    if not len(profile)==1:
        p = poly.polyfit(profile[:,0], profile[:,1], deg=degree)
        profile_new[:,1]=poly.polyval(profile_new[:,0],p)
    return profile_new

def filter_pass(profile, filtermode='upstream',nhd_limits=None):
    profile_new=copy.deepcopy(profile)
    if nhd_limits==None:
        if filtermode=='upstream': # moving suptream
            for i in range(len(profile_new)-1):
                if profile_new[i+1,1] < profile_new[i,1]:
                    profile_new[i+1,1]=profile_new[i,1]

        elif filtermode=='downstream':
            for i in range(len(profile_new)-1, 0,-1):
                if profile_new[i-1,1] > profile_new[i,1]:
                    profile_new[i-1,1] = profile_new[i,1]
                    
    elif nhd_limits is list: 
                 
        if filtermode=='upstream': # moving suptream
            
            profile_new[0,1]=nhd_limits[0]
            for i in range(len(profile_new)-1):
                if profile_new[i+1,1] < profile_new[i,1]:
                    profile_new[i+1,1]=profile_new[i,1]
            
            profile_new[-1,1]=nhd_limits[1]
            for i in range(len(profile_new)-1, 0,-1):
                if profile_new[i-1,1] > profile_new[i,1]:
                    profile_new[i-1,1] = profile_new[i,1]

    return profile_new


def bank_nodes_from_elem(elem, m2):
    # function yileding back id of the bank-node for a given stream-mesh element 
    edge_r=list(m2.cell_edges(elem))[0]   # edge on the right as we look from the downstream direction
    edge_l=list(m2.cell_edges(elem))[2]  # edge on the left as we look from the downstream direction
    
    return [bank_nodes_from_edge(edge_r, elem, m2), bank_nodes_from_edge(edge_l, elem, m2)]

def bank_nodes_from_edge(edge, elem, m2):
    cell_ids=m2._edges_to_cells[edge]
    cells_to_edge=[m2.conn[cell_id] for cell_id in cell_ids] 
    cells_to_edge.remove(elem)
    bank_tri=cells_to_edge[0]
    node_id=(set(bank_tri)-set(edge)).pop()
    return node_id


# def condition_river_mesh(m2, river):
#     for node in river.preOrder():
#         order, min_elev, max_elev=[node.segment.properties['order'], node.segment.properties['min_elev'], node.segment.properties['max_elev']]
#         if not order == 5: # leaving out 4th order streams
#             n_quads=len(node.elements); # this includes traingles and pentagons
#             elevations=np.linspace(min_elev,max_elev,n_quads+1)
#             for i, elem in enumerate(node.elements):
#                 if len(elem)==3:          
#                     m2.coords[elem[0]][2]= m2.coords[elem[2]][2]= min_elev #elevations[i]
#                     m2.coords[elem[1]][2]= min_elev #elevations[i+1]

#                 elif len(elem)==4: 
#                     m2.coords[elem[0]][2]= m2.coords[elem[3]][2]= min_elev #elevations[i]
#                     m2.coords[elem[1]][2]= m2.coords[elem[2]][2]= min_elev #elevations[i+1]
#                 elif len(elem)==5:  
                    
#                     m2.coords[elem[0]][2]= m2.coords[elem[4]][2]= min_elev #elevations[i]
#                     m2.coords[elem[1]][2]= m2.coords[elem[2]][2]= m2.coords[elem[3]][2]= min_elev #elevations[i+1]

#             # conditioning of the stream bank if stream is "placed" into a pond or pit
#             # bank_node_ids=bank_nodes_from_elem(elem, m2)
#             # for node_id in bank_node_ids:
#             #         if m2.coords[node_id][2]<elevations[i]:
#             #                m2.coords[node_id][2]= elevations[i]+1.5
