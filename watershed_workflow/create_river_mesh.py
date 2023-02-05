"""creates river mesh using quad, pentagon and hexagon elements"""

import numpy as np
import pandas as pd
import logging

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree


def sort_children_by_angle(tree, reverse=False):
    """Sorts the children of a given segment by their angle with respect to that segment."""
    for node in tree.preOrder():
        if len(node.children) > 1:
            # compute tangents
            my_seg_tan = np.array(node.segment.coords[0]) - np.array(node.segment.coords[1])

            if reverse: sign = -1
            else: sign = 1

            def angle(c):
                tan = np.array(c.segment.coords[-2]) - np.array(c.segment.coords[-1])
                return sign * watershed_workflow.utils.angle(my_seg_tan, tan)

            node.children.sort(key=angle)


def create_rivers_meshes(rivers, widths=8, enforce_convexity=True):
    """Returns list of elems and river corridor polygons for a given list of river trees

    Parameters:
    -----------
    rivers: List(watershed_workflow.river_tree.RiverTree object)
        List of river tree along which river meshes are to be created
    widths: Float or a dictionary {stream-order: width}
    junction_treatment: boolean 
        flag for enforcing convexity of the pentagons at the junctions
    
    Returns
    -------
    elems: List(List)
        List of river elements
    corrs: List(shapely.geometry.Polygon)
        List of river corridor polygons
    """

    elems = []
    corrs = []
    gid_shift = 0
    for river in rivers:
        if len(elems) != 0:
            gid_shift = np.max([max(map(int, elem)) for elem in elems]) + 1
        elems_river, corr = create_river_mesh(river,
                                              widths=widths,
                                              enforce_convexity=enforce_convexity,
                                              gid_shift=gid_shift)
        elems = elems + elems_river
        corrs = corrs + [corr, ]

    return elems, corrs


def create_river_mesh(river, widths=8, enforce_convexity=True, gid_shift=0, dilation_width=4):
    """Returns list of elems and river corridor polygons for a given river tree

    Parameters:
    -----------
    river: watershed_workflow.river_tree.RiverTree object)
        river tree along which mesh is to be created
    widths: Float or a dictionary {stream-order: width}
    junction_treatment: boolean 
        flag for enforcing convexity of the pentagons at the junctions
    gid_shift: Integer
        all the node-ids used in the element defination are shifted by
        this number to make it consistant with the global ids in the 
        m2 mesh, important in case of multiple rivers
    dilation_width: Integer
        this is used for initial buffering of river tree into river corridor polygon. 
        for typical watershed 8m default should work well, however, for smaller domains, setting smaller
        initial dilation_width might be desirable (much smaller than expected quad element length)    
    Returns
    -------
    elems: List(List)
        List of river elements
    corr: List(shapely.geometry.Polygon)
        a river corridor polygon
    """

    # creating a polygon for river corridor by dilating the river tree
    corr = create_river_corridor(river, dilation_width)
    # defining special elements in the mesh
    elems = to_quads(river, corr, dilation_width, gid_shift=gid_shift)
    # setting river_widths in the river corridor polygon
    corr = set_width_by_order(river, corr, widths=widths, dilation_width = dilation_width)
    # treating non-convexity at junctions
    if enforce_convexity:
        corr = convexity_enforcement(river, corr)

    return elems, corr


def create_river_corridor(river, width):
    """Returns a polygon representing the river corridor.
    
    Parameters
    ----------
    river: watershed_workflow.river_tree.RiverTree object
        river tree along which river corrid polygon is to be created
    delta: Float
        for initial dilation, distance between stream-centerline and edge of corridor

    Returns
    -------
    corr3: shapely.geometry.Polygon
        river corridor polygon for the given river     
    """

    # first sort the river so that in a search we always take paddlers right...
    # check river consistency
    if not river.is_continuous():
        river.make_continuous()
    sort_children_by_angle(river, True)
    delta=width/2

    # find smallest lengthscale as threshold to identify double and triple points
    mins = []
    for line in river.dfs():
        coords = np.array(line.coords[:])
        dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
        mins.append(np.min(dz))
    logging.info(f"  river min seg length: {min(mins)}")
       
    length_scale = max(2.1*delta, min(mins) - 4*delta) # Currently this same for the whole river, should we change it reachwise?
    print('length_scale ', length_scale )

    # buffer by the width
    mls = shapely.geometry.MultiLineString([r for r in river.dfs()])
    corr = mls.buffer(delta,
                      cap_style=shapely.geometry.CAP_STYLE.flat,
                      join_style=shapely.geometry.JOIN_STYLE.mitre)

    # cycle the corridor points to start and end with the 1st point...
    corr_p = list(corr.exterior.coords[:-1])
    outlet_p = river.segment.coords[-1]
    index_min = min(range(len(corr_p)),
                    key=lambda i: watershed_workflow.utils.distance(corr_p[i], outlet_p))
    plus_one = (index_min+1) % len(corr_p)
    minus_one = (index_min-1) % len(corr_p)
    if (watershed_workflow.utils.distance(corr_p[plus_one], outlet_p) <
            watershed_workflow.utils.distance(corr_p[minus_one], outlet_p)):
        corr2_p = corr_p[plus_one:] + corr_p[0:plus_one]
    else:
        corr2_p = corr_p[index_min:] + corr_p[0:index_min]
    corr2 = shapely.geometry.Polygon(corr2_p)

    # remove endpoint-doubles that we want to be a single point and
    # weird artifact triples at junctions
    corr3_p = []
    i = 0
    while i < len(corr2_p):
        logging.debug(f'considering {i}')
        if i == 0 or i == len(corr2_p) - 1:
            # keep first and last always -- first two points make the outlet segment
            logging.debug(f' always keeping')
            corr3_p.append(corr2_p[i])
        else:
            if watershed_workflow.utils.distance(corr2_p[i - 1], corr2_p[i]) < length_scale:
                # is this a triple point?
                if watershed_workflow.utils.distance(corr2_p[i + 1], corr2_p[i]) < length_scale:
                    logging.debug(' triple point!')
                    # triple point, average neighbors and skip the next point
                    corr3_p.append(watershed_workflow.utils.midpoint(corr2_p[i + 1],
                                                                     corr2_p[i - 1]))
                    i += 1
                else:
                    # double point -- an end of a first order stream
                    logging.debug(' double point')
                    corr3_p.append(watershed_workflow.utils.midpoint(corr2_p[i - 1], corr2_p[i]))
            else:
                # will the next point deal with this?
                if watershed_workflow.utils.distance(corr2_p[i], corr2_p[i + 1]) < length_scale:
                    logging.debug(' not my problem')
                    pass
                else:
                    logging.debug(' keeping')
                    corr3_p.append(corr2_p[i])
        i += 1

    # create the polgyon
    corr3 = shapely.geometry.Polygon(corr3_p)
    return corr3


def to_quads(river, corr, width, gid_shift=0, ax=None):
    """Iterate over the rivers, creating quads and pentagons forming the corridor.
    The global_id_adjustment is to keep track of node_id in elements w.r.t to global id in mesh
    mainly relevant for multiple river

    Parameters
    ----------
    rivers : watershed_workflow.river_tree.RiverTree object
        river tree 
    corr : shapely.geometry.Polygons
        a river corridor polygon for the river
    delta : Float
        river width used for creating corr from river in function "create_river_corridor"
    gid_shift: Integer
        all the node-ids used in the element defination are shifted by
        this number to make it consistant with the global ids in the 
        m2 mesh, important in case of multiple rivers

    Returns
    -------
    elems: List(List)
        List of river elements
    """
    delta = width/2
    
    coords = corr.exterior.coords[:-1]

    import time
    def pause():
        time.sleep(0.)
    # number the nodes in a dfs pattern, creating empty space for elements
    for i, node in enumerate(river.preOrder()):
        node.id = i
        node.elements = [list() for l in range(len(node.segment.coords) - 1)]
        assert (len(node.elements) >= 1)
        node.touched = 0

    # iterate over the tree in an out-and-back-and-in-between
    # traversal, where every node appears num_children + 1 times,
    # before and after and between each child.
    ic = 0
    total_touches = 0
    for node in river.prePostInBetweenOrder():
        logging.debug(
            f'touching {node.id} (previously touched {node.touched} times with {len(node.children)} children)'
        )
        if node.touched == 0:
            logging.debug(f'  first time around! {node.touched+1}')
            # not yet touched -- add the first coordinates
            seg_coords = [coords[ic], ]
            for j in range(len(node.elements)):
                node.elements[j].append(ic)
                ic += 1
                node.elements[j].append(ic)
                seg_coords.append(coords[ic])

            node.touched += 1
            total_touches += 1

            seg_coords = np.array(seg_coords)
            if ax != None: 
                ax.plot(seg_coords[:,0], seg_coords[:,1], 'm^', markersize=5)
                pause()

        elif node.touched == 1 and len(node.children) == 0:
            # leaf node, last time
            logging.debug(f' last time around a leaf! {node.touched+1}')
            # increment to avoid double-counting the point in the triangle on the ends
            seg_coords = [coords[ic], ]
            ic += 1
            node.elements[-1].append(ic)
            seg_coords.append(coords[ic])
            for j in reversed(range(len(node.elements) - 1)):
                node.elements[j].append(ic)
                ic += 1
                node.elements[j].append(ic)
                seg_coords.append(coords[ic])
            node.touched += 1
            total_touches += 1

            if ax != None:           
            # plot it...
                seg_coords = np.array(seg_coords)
                ax.plot(seg_coords[:,0], seg_coords[:,1], 'gv',markersize=5)

                # also plot the conn
                for i, elem in enumerate(node.elements):
                    looped_conn = elem[:]
                    looped_conn.append(elem[0])
                    if i == len(node.elements)-1:
                        assert(len(looped_conn) == 4)
                    else:
                        assert(len(looped_conn) == 5)
                    cc = np.array([coords[n] for n in looped_conn])
                    ax.plot(cc[:,0], cc[:,1], 'g-o')
                pause()
            

        elif node.touched == len(node.children):
            logging.debug(f'  last time around! {node.touched+1}')
            seg_coords = [coords[ic], ]
            # touched enough times that this is the last appearance
            # add the last coordinates
            for j in reversed(range(len(node.elements))):
                node.elements[j].append(ic)
                ic += 1
                node.elements[j].append(ic)
                seg_coords.append(coords[ic])
            node.touched += 1
            total_touches += 1

            if ax != None: 
                #plot it...
                seg_coords = np.array(seg_coords)
                ax.plot(seg_coords[:,0], seg_coords[:,1], 'b^',markersize=5)

            # also plot the conn
            for i,elem in enumerate(node.elements):
                looped_conn = elem[:]
                looped_conn.append(elem[0])
                if i == len(node.elements)-1:
                    assert(len(looped_conn) == (node.touched+3))
                else:
                    assert(len(looped_conn) == 5)
                cc = np.array([coords[n] for n in looped_conn])

                for c in cc:
                    # note, the more acute an angle, the bigger this distance can get...
                    # so it is a bit hard to pin this multiple down -- using 5 seems ok?
                    if not (watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+1)], 25*delta) or \
                           watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+2)], 25*delta)):
                           print(c, node.segment.coords[len(node.segment.coords)-(i+1)], node.segment.coords[len(node.segment.coords)-(i+2)])
                           print(node.id)
                           assert(watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+1)], 25*delta) or \
                           watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+2)], 25*delta))
                if ax != None: 
                    ax.plot(cc[:,0], cc[:,1], 'g-o')

            pause()

        else: # adding the junction point 
            logging.debug(f'  middle time around! {node.touched+1}')
            assert (node.touched < len(node.children))
            # touched in between children
            # therefore this is at least a pentagon
            # add the middle node on the last element
            node.elements[-1].append(ic)
            node.touched += 1

            if ax != None: 
                ax.scatter([coords[ic][0],], [coords[ic][1],], c='m', marker='^')
                pause()

    #print(ic)
    assert (len(coords) == (ic + 1))
    assert (len(river) * 2 == total_touches)

    # this nodeid-shift is needed in case of multiple rivers, to make this id consistent with global node ids in a m2 mesh
    for node in river.preOrder():
        for i, elems in enumerate(node.elements):
            elems_new = [node_id + gid_shift for node_id in elems]
            node.elements[i] = elems_new

    elems = [el for node in river.preOrder() for el in node.elements]
    return elems


def set_width_by_order(river, corr, widths=8, dilation_width=8):
    """this functions takes the river-corridor polygon and sets the width of the corridor based on the order 
       dependent width dictionary

    Parameters
    ----------
    river: watershed_workflow.river_tree.RiverTree object)
        river tree along which mesh is to be created
    corr : shapely.geometry.Polygons
        a river corridor polygon for the river    
    widths: Float or a dictionary {stream-order: width}

    Returns
    -------
    shapely.geometry.Polygon(corr_coords_new): 
        river corridor polygon with adjusted width
    """

    corr_coords = corr.exterior.coords[:-1]
    for j, node in enumerate(river.preOrder()):

        if type(widths) == dict:
            order = node.properties["StreamOrder"]
            target_width = width_cal(widths, order)
           
        else:
            target_width= widths

        for i, elem in enumerate(node.elements):  # treating the upstream edge of the element
            if len(elem) == 4:
                p1 = np.array(corr_coords[elem[1]][:2])  # points of the upstream edge of the quad
                p2 = np.array(corr_coords[elem[2]][:2])
                [p1_, p2_] = move_to_target_separation(p1, p2, target_width, dilation_width = dilation_width)
                corr_coords[elem[1]] = tuple(p1_)
                corr_coords[elem[2]] = tuple(p2_)

            if len(elem) == 5:
                p1 = np.array(corr_coords[elem[1]][:2])  # neck of the pent
                p2 = np.array(corr_coords[elem[3]][:2])
                [p1_, p2_] = move_to_target_separation(p1, p2, target_width, dilation_width = dilation_width)
                corr_coords[elem[1]] = tuple(p1_)
                corr_coords[elem[3]] = tuple(p2_)

            if len(elem)==6:
                p1=np.array(corr_coords[elem[2]][:2]) # neck of the hex
                p2=np.array(corr_coords[elem[3]][:2])
                [p1_, p2_]= move_to_target_separation(p1, p2, target_width, dilation_width = dilation_width)
                corr_coords[elem[2]]=tuple(p1_)
                corr_coords[elem[3]]=tuple(p2_)

                p1=np.array(corr_coords[elem[1]][:2]) # neck of the hex
                p2=np.array(corr_coords[elem[4]][:2])
                [p1_, p2_]= move_to_target_separation(p1, p2, target_width, dilation_width = dilation_width)
                corr_coords[elem[1]]=tuple(p1_)
                corr_coords[elem[4]]=tuple(p2_)

            if i == 0:  # this is to treat the most downstream edge which is left out so far
                p1 = np.array(
                    corr_coords[elem[0]][:2])  # points of the upstream edge of the quad/pent
                p2 = np.array(corr_coords[elem[-1]][:2])
                [p1_, p2_] = move_to_target_separation(p1, p2, target_width, dilation_width = dilation_width)
                corr_coords[elem[0]] = tuple(p1_)
                corr_coords[elem[-1]] = tuple(p2_)

    corr_coords_new = corr_coords + [corr_coords[0]]
    return shapely.geometry.Polygon(corr_coords_new)


def move_to_target_separation(p1, p2, target, dilation_width=8):
    """Returns the points after moving them to a target separation from each other"""
    import math
    d_vec = p1 - p2  # separation vector
    d = np.sqrt(d_vec.dot(d_vec))  # distance 
    target = target * min(d/dilation_width, 1.2) # this scales for angled joints (not exactly calculated but should be good enough)
    delta = target - d
    p1_ = p1 + 0.5 * delta * (d_vec) / d
    p2_ = p2 - 0.5 * delta * (d_vec) / d
    d_ = watershed_workflow.utils.distance(p1_, p2_)
    assert (math.isclose(d_, target, rel_tol=1e-5))
    return [p1_, p2_]


def width_cal(width_dict, order):
    """Returns the reach width based using the {order:width dictionary}"""
    if order > max(width_dict.keys()):
        width = width_dict[max(width_dict.keys())]
    elif order < min(width_dict.keys()):
        width = width_dict[min(width_dict.keys())]
    else:
        width = width_dict[order]
    return width


def convexity_enforcement(river, corr):
    """this functions check the river-corridor polygon for convexity, if non-convex, moves the node onto the convex hull of the element-polygon

    Parameters
    ----------
    river: watershed_workflow.river_tree.RiverTree object)
        river tree along which mesh is to be created
    corr : shapely.geometry.Polygons
        a river corridor polygon for the river 
  
    Returns
    -------
    shapely.geometry.Polygon(corr_coords_new): 
        river corridor polygon with adjusted width
    """
    coords = corr.exterior.coords[:-1]

    for j, node in enumerate(river.preOrder()):
        for elem in node.elements:
            if len(elem) == 5 or len(elem) == 6:  # checking and treating this pentagon/hexagon
                points = [coords[id] for id in elem]
                if not watershed_workflow.utils.is_convex(points):
                    convex_ring = shapely.geometry.Polygon(points).convex_hull.exterior
                    for i, point in enumerate(
                            points):  # replace point with nearest point on convex hull
                        p = shapely.geometry.Point(point)
                        new_point = shapely.ops.nearest_points(convex_ring, p)[0].coords[0]
                        points[i] = new_point

                #assert(watershed_workflow.utils.is_convex(points))
              

                # updating coords
                for id, point in zip(elem, points):
                    coords[id] = point

    corr_coords_new = coords + [coords[0]]
    return shapely.geometry.Polygon(corr_coords_new)
