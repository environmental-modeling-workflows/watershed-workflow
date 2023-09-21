"""creates river mesh using quad, pentagon and hexagon elements"""

import numpy as np
import pandas as pd
import logging

import shapely.geometry
import shapely.ops

import watershed_workflow.utils
import watershed_workflow.tinytree
import watershed_workflow.plot


class IntersectionError(Exception):
    pass


def _indexPointInSeg(segment, point, tol=1.e-2):
    ind = segment.coords[:].index(point.coords[0])
    # except ValueError:
    #     ind, p = min(((i,p) for (i,p) in enumerate(segment.coords[:])),
    #                 key=lambda ip : shapely.geometry.Point(ip[1]).distance(point))
    #     assert(shapely.geometry.Point(p).distance(point) < tol)
    return ind


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


def _isOverlappingCorridor(corr, river):
    if len(corr.interiors) > 0:
        # there is an overlap upstream of the junction of two tributaries,
        # creating a hole
        return 2
    if not _isExpectedNumPoints(corr, river):
        # overlaps at the junction result in losing points in the corridor polygon.
        return 1
    return 0


def _isOverlappingCorridors(corrs, rivers):
    """Corridors can overlap"""
    if any(_isOverlappingCorridor(c, river) for (c, river) in zip(corrs, rivers)):
        return True

    corrs_area = shapely.ops.unary_union(corrs).area
    summed_area = sum(c.area for c in corrs)
    if abs(summed_area - corrs_area) > 1.e-3:
        return True
    return False


def _isExpectedNumPoints(corr, river):
    """Check if the points on the corridor are same as calculated theoretically"""
    n_child = []
    for node in river.preOrder():
        n_child.append(len(node.children))
    n = 2  # two outlet points
    for node in river.preOrder():
        n = n + 2 * (len(node.segment.coords) - 1)
    n = n - n_child.count(0) + n_child.count(2) + 2 * n_child.count(3) + 3 * n_child.count(4)
    return (len(corr.exterior.coords) - 1 == n)


def create_rivers_meshes(rivers, widths=8, enforce_convexity=True, ax=None, label=True):
    """Returns list of elems and river corridor polygons for a given list of river trees

    Parameters:
    -----------
    rivers: list(watershed_workflow.river_tree.River object)
        List of river tree along which river meshes are to be created
    widths: float or a dictionary
        Width of streams, as constant or {stream-order: width}
    enforce_convexity: boolean 
        If true, enforce convexity of the pentagons/hexagons at the
        junctions.
    ax : matplotlib Axes object, optional
        For debugging -- plots troublesome reaches as quad elements are
        generated to find tricky areas.
    label : bool, optional = True
        If true and ax is provided, animates the debugging plot with
        reach ID labels as the user hovers over the plot.  Requires a
        widget backend for matplotlib.
    
    Returns
    -------
    elems: list(list)
        List of river elements, each element a list of indices into
        corr.coords.
    corrs: list(shapely.geometry.Polygon)
        List of river corridor polygons, one per river, storing the
        coordinates used in elems.

    """
    elems = []
    corrs = []
    gid_shift = 0

    # set up debugging plot
    if ax is not None:
        logging.info('  ... setting up debug figure')
        fig = ax.get_figure()
        ax.set_title(' magenta ^: first touch \n green o: leaf, final element '
                     'complete \n blue o: internal element complete ')

        nodes = [r for river in rivers for r in river.preOrder()]
        reach_list = [r.segment for r in nodes]
        reach_names = [r.properties['ID'] for r in nodes]
        reach_colors = watershed_workflow.colors.enumerated_colors(len(nodes), 2)
        lines = watershed_workflow.plot.shplys(reach_list, None, reach_colors, ax)

        if label:
            # the next block creates a hoverable plot where reaches are
            # annotated with their IDs.  This makes it easier for the user
            # to find the bad spot.
            annot = ax.annotate("",
                                xy=(0, 0),
                                xytext=(20, 20),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(idx):
                posx, posy = reach_list[idx].centroid.coords[0]
                annot.xy = (posx, posy)
                text = f'ID: {reach_names[idx]}'
                annot.set_text(text)
                annot.get_bbox_patch().set_facecolor(reach_colors[idx])
                annot.get_bbox_patch().set_alpha(0.8)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = lines.contains(event)
                    if cont:
                        update_annot(ind['ind'][0])
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            ax.get_figure().canvas.mpl_connect("motion_notify_event", hover)

    # now do the actual work
    for i, river in enumerate(rivers):
        logging.info(f'River {i}')
        if len(elems) != 0:
            gid_shift = np.max([max(map(int, elem)) for elem in elems]) + 1
        elems_river, corr = create_river_mesh(river,
                                              widths=widths,
                                              enforce_convexity=enforce_convexity,
                                              gid_shift=gid_shift,
                                              ax=ax)

        elems = elems + elems_river
        corrs = corrs + [corr, ]

    if _isOverlappingCorridors(corrs, rivers):
        raise RuntimeError('Overlapping quad elements in neighboring rivers, try '
                           'reducing river width or check reach data')
    return elems, corrs


def create_river_mesh(river,
                      widths=8,
                      enforce_convexity=True,
                      gid_shift=0,
                      dilation_width=4,
                      ax=None):
    """Returns list of elems and river corridor polygons for a given river tree

    Parameters:
    -----------
    river: watershed_workflow.river_tree.River object)
      River tree along which mesh is to be created
    widths: float or a dictionary
      Width of streams, as constant or {stream-order: width}
    enforce_convexity: boolean 
      If true, enforce convexity of the pentagons/hexagons at the
      junctions.
    gid_shift: int
      All the node-ids used in the element defination are shifted by
      this number to make it consistant with the global ids in the m2
      mesh, necessary in the case of multiple rivers
    dilation_width: float
      This is used for initial buffering of the river tree into the
      river corridor polygon.  For a typical watershed the 4 m default
      should work well; for smaller domains, setting smaller initial
      dilation_width might be desirable (much smaller than expected
      quad element length).
    ax : matplotlib Axes object, optional
      For debugging -- plots troublesome reaches as quad elements are
      generated to find tricky areas.

    Returns
    -------
    elems: List(List)
        List of river elements, each element a list of indices into
        corr.coords.
    corr: shapely.geometry.Polygon
        The polygon of all elements, stores the coordinates.

    """
    # creating a polygon for river corridor by dilating the river tree
    if type(widths) == dict:
        dilation_width = min(dilation_width, min(widths.values()))
    else:
        dilation_width = min(dilation_width, widths)
    corr = create_river_corridor(river, dilation_width)

    # defining special elements in the mesh
    elems = to_quads(river, corr, dilation_width, gid_shift=gid_shift, ax=ax)

    # setting river_widths in the river corridor polygon
    corr = set_width_by_order(river,
                              corr,
                              widths=widths,
                              dilation_width=dilation_width,
                              gid_shift=gid_shift)

    # treating non-convexity at junctions
    if enforce_convexity:
        corr = convexity_enforcement(river,
                                     corr,
                                     widths=widths,
                                     dilation_width=dilation_width,
                                     gid_shift=gid_shift)

    # redraw the debug plot with updated elems
    if ax is not None:
        coords = list(corr.exterior.coords)
        shapes = [shapely.geometry.Polygon([coords[e] for e in elem]) for elem in elems]
        watershed_workflow.plot.shplys(shapes, None, 'r', ax)

    oc = _isOverlappingCorridor(corr, river)
    if oc == 1:
        raise RuntimeError('Overlapping quad elements away from a junction -- try '
                           'reducing river width or checking data to ensure no braided '
                           'or nearly braided systems')
    elif oc == 2:
        raise RuntimeError('Overlapping quad elements at a junction -- try '
                           ' increasing junction_angle_limit in densification, or check data.')
    return elems, corr


def create_river_corridor(river, width):
    """Returns a polygon representing the river corridor with a fixed
    dilation width.
    
    Parameters
    ----------
    river : watershed_workflow.river_tree.River object
      River tree along which corridor polygon is to be created.
    width : float
      Width to dilate the river.

    Returns
    -------
    shapely.geometry.Polygon
      Resulting polygon.

    """
    logging.info(f'... generating initial polygon through dilation ({width} m)')
    # first sort the river so that in a search we always take paddlers right...
    # check river consistency
    if not river.is_continuous():
        river.make_continuous()
    sort_children_by_angle(river, True)
    delta = width / 2

    # make there are no three collinear points, else buffer will ignore those points
    logging.info(f'  -- treating collinearity')
    for node in river.preOrder():
        new_seg_coords = watershed_workflow.utils.treat_segment_collinearity(node.segment.coords[:])
        node.segment = shapely.geometry.LineString(new_seg_coords)

    # find smallest lengthscale as threshold to identify double and triple points
    mins = []
    for line in river.depthFirst():
        coords = np.array(line.coords[:])
        dz = np.linalg.norm(coords[1:] - coords[:-1], 2, -1)
        mins.append(np.min(dz))

    logging.info(f"  -- river min seg length: {min(mins)}")

    # Currently this same for the whole river, should we change it reachwise?
    length_scale = max(2.1 * delta, min(mins) - 8*delta)
    logging.info(f"  -- merging points closer than {length_scale} m along the river corridor")

    # buffer by the width
    mls = shapely.geometry.MultiLineString([r for r in river.depthFirst()])
    corr = mls.buffer(delta,
                      cap_style=shapely.geometry.CAP_STYLE.flat,
                      join_style=shapely.geometry.JOIN_STYLE.mitre)

    # cycle the corridor points to start and end with the 1st point.
    corr_p = list(corr.exterior.coords[:-1])
    outlet_p = river.segment.coords[-1]
    index_min = min(range(len(corr_p)),
                    key=lambda i: watershed_workflow.utils.distance(corr_p[i], outlet_p))
    plus_one = (index_min+1) % len(corr_p)
    minus_one = (index_min-1) % len(corr_p)
    if (watershed_workflow.utils.distance(corr_p[plus_one], outlet_p)
            < watershed_workflow.utils.distance(corr_p[minus_one], outlet_p)):
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

    # check if the points on the river corridor are same as calculated theoretically
    if not _isExpectedNumPoints(corr3, river):
        raise RuntimeError(
            f"Broken dilation -- expected {n} coords, got {len(corr3.exterior.coords[:])}"
            " -- recommend running with ax argument to tessalate_river_aligned() to debug!")

    return corr3


def to_quads(river, corr, width, gid_shift=0, ax=None):
    """Iterate over the rivers, creating quads and pentagons forming the corridor.

    The global_id_adjustment is to keep track of node_id in elements
    w.r.t to global id in mesh mainly relevant for multiple river

    Parameters
    ----------
    rivers : watershed_workflow.river_tree.River
      river tree 
    corr : shapely.geometry.Polygon
      a river corridor polygon for the river
    width : float
      fixed width used for dilating the river
    gid_shift: int
      All the node-ids used in the element defination are shifted by
      this number to make it consistant with the global ids in the m2
      mesh, necessary in the case of multiple rivers.

    Returns
    -------
    elems: list(list)
      List of river elements

    """
    logging.info('... defining river-mesh topology (quad elements)')
    delta = width / 2

    coords = corr.exterior.coords[:-1]

    artists = []  # list of things added to ax here
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
                artists.append(ax.plot(seg_coords[:, 0], seg_coords[:, 1], 'm^', markersize=5))
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
                # seg_coords = np.array(seg_coords)
                # artists.append(ax.plot(seg_coords[:, 0], seg_coords[:, 1], 'gv', markersize=5))

                # also plot the conn
                for i, elem in enumerate(node.elements):
                    looped_conn = elem[:]
                    looped_conn.append(elem[0])
                    if i == len(node.elements) - 1:
                        assert (len(looped_conn) == 4)
                    else:
                        assert (len(looped_conn) == 5)
                    cc = np.array([coords[n] for n in looped_conn])
                    artists.append(ax.plot(cc[:, 0], cc[:, 1], 'g-o'))
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
                artists.append(ax.plot(seg_coords[:, 0], seg_coords[:, 1], 'b^', markersize=5))

            # also plot the conn
            for i, elem in enumerate(node.elements):
                looped_conn = elem[:]
                looped_conn.append(elem[0])
                if i == len(node.elements) - 1:
                    assert (len(looped_conn) == (node.touched + 3))
                else:
                    assert (len(looped_conn) == 5)
                cc = np.array([coords[n] for n in looped_conn])

                if ax != None:
                    artists.append(ax.plot(cc[:, 0], cc[:, 1], 'b-o'))

                for c in cc:
                    # What is this actually checking?  Need a comment here -- this is confusing code. --ETC
                    #
                    # note, the more acute an angle, the bigger this distance can get...
                    # so it is a bit hard to pin this multiple down -- using 25 seems ok?
                    if not (watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+1)], 25*delta) or \
                           watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+2)], 25*delta)):
                        logging.debug(c, node.segment.coords[len(node.segment.coords) - (i+1)],
                                      node.segment.coords[len(node.segment.coords) - (i+2)])
                        logging.debug(node.id)
                        assert(watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+1)], 25*delta) or \
                        watershed_workflow.utils.close(tuple(c), node.segment.coords[len(node.segment.coords)-(i+2)], 25*delta))

            pause()

        else:  # adding the junction point
            logging.debug(f'  middle time around! {node.touched+1}')
            assert (node.touched < len(node.children))
            # touched in between children
            # therefore this is at least a pentagon
            # add the middle node on the last element
            node.elements[-1].append(ic)
            node.touched += 1

            if ax != None:
                artists.append(ax.scatter([coords[ic][0], ], [coords[ic][1], ], c='m', marker='^'))
                pause()

    assert (len(coords) == (ic + 1))
    assert (len(river) * 2 == total_touches)

    # this nodeid-shift is needed in case of multiple rivers, to make
    # this id consistent with global node ids in a m2 mesh
    for node in river.preOrder():
        for i, elems in enumerate(node.elements):
            elems_new = [node_id + gid_shift for node_id in elems]
            node.elements[i] = elems_new

    elems = [el for node in river.preOrder() for el in node.elements]

    # once we have reached here, the debug plot is not so needed anymore,
    # and leaving all our dots makes things cluttered.
    for artist in artists:
        if isinstance(artist, list):
            for it in artist:
                it.remove()
        else:
            artist.remove()

    return elems


def set_width_by_order(river, corr, widths=8, dilation_width=8, gid_shift=0):
    """Adjust the river-corridor polygon width by order.

    Parameters
    ----------
    river: watershed_workflow.river_tree.River
      river tree along which mesh is to be created
    corr : shapely.geometry.Polygon
      a river corridor polygon for the river    
    widths: float or dict
      Width of the river, as constant or {stream-order: width}

    Returns
    -------
    shapely.geometry.Polygon
      river corridor polygon with adjusted width

    """
    logging.info("... setting width of quad elements")
    corr_coords = corr.exterior.coords[:-1]

    if type(widths) == dict:
        stream_order = next(
            (i for i in ['StreamOrder', 'streamorder', 'streamorde'] if i in river.properties),
            None)
        if stream_order is None:
            raise RuntimeError(
                'set_width_by_order requested widths by stream order, but cannot guess stream order property name'
            )

    for j, node in enumerate(river.preOrder()):
        if type(widths) == dict:
            order = node.properties[stream_order]
            target_width = width_by_order(widths, order)
        else:
            target_width = widths

        # loop over elems, treating the upstream edge
        for i, elem in enumerate(node.elements):
            elem = [id - gid_shift for id in elem]

            if len(elem) == 3:
                continue  # no upstream edge

            elif len(elem) == 4:
                p1 = np.array(corr_coords[elem[1]][:2])  # points of the upstream edge of the quad
                p2 = np.array(corr_coords[elem[2]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[1]] = tuple(p1_)
                corr_coords[elem[2]] = tuple(p2_)

            elif len(elem) == 5:
                p1 = np.array(corr_coords[elem[1]][:2])  # neck of the pent
                p2 = np.array(corr_coords[elem[3]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[1]] = tuple(p1_)
                corr_coords[elem[3]] = tuple(p2_)

            elif len(elem) == 6:
                p1 = np.array(corr_coords[elem[2]][:2])  # neck of the hex
                p2 = np.array(corr_coords[elem[3]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[2]] = tuple(p1_)
                corr_coords[elem[3]] = tuple(p2_)

                p1 = np.array(corr_coords[elem[1]][:2])  # in one
                p2 = np.array(corr_coords[elem[4]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[1]] = tuple(p1_)
                corr_coords[elem[4]] = tuple(p2_)

            elif len(elem) == 7:
                p1 = np.array(corr_coords[elem[2]][:2])  # neck of the sept
                p2 = np.array(corr_coords[elem[4]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[2]] = tuple(p1_)
                corr_coords[elem[4]] = tuple(p2_)

                p1 = np.array(corr_coords[elem[1]][:2])  # in one
                p2 = np.array(corr_coords[elem[5]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[1]] = tuple(p1_)
                corr_coords[elem[5]] = tuple(p2_)

            else:
                raise NotImplementedError(
                    f"Case with {len(elem)} nodes is not yet treated... good luck!")

            # treat the most downstream edge, which is left out so far
            if i == 0:
                # points of the upstream edge of the quad/pent
                p1 = np.array(corr_coords[elem[0]][:2])
                p2 = np.array(corr_coords[elem[-1]][:2])
                [p1_, p2_] = move_to_target_separation(p1,
                                                       p2,
                                                       target_width,
                                                       dilation_width=dilation_width)
                corr_coords[elem[0]] = tuple(p1_)
                corr_coords[elem[-1]] = tuple(p2_)

    corr_coords_new = corr_coords + [corr_coords[0]]
    return shapely.geometry.Polygon(corr_coords_new)


def move_to_target_separation(p1, p2, target, dilation_width=8):
    """Returns the points after moving them to a target separation from each other"""
    import math
    d_vec = p1 - p2  # separation vector
    d = np.sqrt(d_vec.dot(d_vec))  # distance

    # scales for angled joints (not exactly calculated but should be good enough)
    target = target * min(d / dilation_width, 1.2)
    delta = target - d
    p1_ = p1 + 0.5 * delta * (d_vec) / d
    p2_ = p2 - 0.5 * delta * (d_vec) / d
    d_ = watershed_workflow.utils.distance(p1_, p2_)
    assert (math.isclose(d_, target, rel_tol=1e-5))
    return [p1_, p2_]


def width_by_order(width_dict, order):
    """Returns the reach width based using the {order:width dictionary}"""
    if order > max(width_dict.keys()):
        width = width_dict[max(width_dict.keys())]
    elif order < min(width_dict.keys()):
        width = width_dict[min(width_dict.keys())]
    else:
        width = width_dict[order]
    return width


def convexity_enforcement(river, corr, widths, dilation_width, gid_shift):
    """Ensure convexity of each river-corridor element.

    Moves nodes onto the convex hull of the element if needed.

    Parameters
    ----------
    river: watershed_workflow.river_tree.River
      River tree along which mesh is to be created
    corr : shapely.geometry.Polygon
      The river corridor polygon
  
    Returns
    -------
    shapely.geometry.Polygon
      The new polygon with all convex elements.

    """
    logging.info('... enforcing convexity')
    coords = corr.exterior.coords[:-1]

    for j, node in enumerate(river.preOrder()):
        for elem in node.elements:
            elem = [id - gid_shift for id in elem]
            if len(elem) == 5 or len(elem) == 6 or len(
                    elem) == 7:  # checking and treating this pentagon/hexagon
                points = [coords[id] for id in elem]  # element points
                if not watershed_workflow.utils.is_convex(points):
                    convex_ring = shapely.geometry.Polygon(points).convex_hull.exterior
                    for i, point in enumerate(points):
                        # replace point with nearest point on convex hull
                        p = shapely.geometry.Point(point)
                        new_point = shapely.ops.nearest_points(convex_ring, p)[0].coords[0]
                        points[i] = new_point

                if not (watershed_workflow.utils.is_convex(points)):
                    # go back to original set of points as snapping on
                    # hull might have incorrectly oriented points
                    points = [coords[id] for id in elem]
                    logging.info(
                        f"  could not make these: {points} convex using convex hull, trying nudging...."
                    )
                    points = make_convex_by_nudge(points)

                assert ((watershed_workflow.utils.is_convex(points)))
                for id, point in zip(elem, points):
                    coords[id] = point

    corr_coords_new = coords + [coords[0]]
    return shapely.geometry.Polygon(corr_coords_new)


def make_convex_by_nudge(points):
    """Nudges the neck-points of the junction until the element is convex.

    Used if efficient convexity does not work
    """
    i = 0
    if len(points) == 5:
        while not watershed_workflow.utils.is_convex(points):
            p1, p3 = [np.array(points[1]), np.array(points[3])]
            d = p1 - p3
            p1_ = p3 + 1.01*d
            p3_ = p1 - 1.01*d
            points[1] = tuple(p1_)
            points[3] = tuple(p3_)
            i += 1
        logging.debug(f"... element was adjusted {i} times")

    elif len(points) == 6:
        while not watershed_workflow.utils.is_convex(points):
            p1, p4 = [np.array(points[1]), np.array(points[4])]
            d = p1 - p4
            p1_ = p4 + 1.01*d
            p4_ = p1 - 1.01*d
            points[1] = tuple(p1_)
            points[4] = tuple(p4_)
            i += 1
        logging.debug(f"... element was adjusted {i} times")

    elif len(points) == 7:
        while not watershed_workflow.utils.is_convex(points):
            p1, p5 = [np.array(points[1]), np.array(points[5])]
            d = p1 - p5
            p1_ = p5 + 1.01*d
            p5_ = p1 - 1.01*d
            points[1] = tuple(p1_)
            points[5] = tuple(p5_)

            p2, p4 = [np.array(points[2]), np.array(points[4])]
            d = p2 - p4
            p2_ = p4 + 1.01*d
            p4_ = p2 - 1.01*d
            points[2] = tuple(p2_)
            points[4] = tuple(p4_)
            i += 1
        logging.debug(f"... element was adjusted {i} times")

    else:
        raise NotImplementedError("Case with {len(points)} nodes is not yet treated... good luck!")
    assert (watershed_workflow.utils.is_convex(points))
    return points


## Supporting functions for river meshing: accomodate river corridor with internal huc boundaries
## generally rc = river corridor; rt = river tree


def hucsegs_at_intersection(point, hucs):
    """For a given intersection point, return a list of indices for huc.segments touching this point"""
    intersection_segs = []
    for i, seg in enumerate(hucs.segments):
        if seg.intersects(point):
            intersection_segs.append(i)
    return intersection_segs


def node_at_intersection(point, river):
    # for a given intersection point, find all the huc-segments (indices)
    intersection_node = None
    len_scale = watershed_workflow.utils.distance(river.segment.coords[0], river.segment.coords[1])
    for node in river.preOrder():
        if point.buffer(0.1 * len_scale).intersects(node.segment):
            intersection_node = node
            break
    return intersection_node


def angle_rivers_segs(ref_seg, seg):
    """Returns the angle of incoming-river-segment or huc-segment w.r.t outgoing river.

    The angle is measured clockwise; this is useful to sort
    orientation wise and add river corridor points at junction

    """
    ref_seg_tan = np.array(ref_seg.coords[1]) - np.array(ref_seg.coords[0])
    if type(seg) is shapely.geometry.LineString:
        intersection_point = ref_seg.intersection(seg)
        seg_orientation_flag = np.argmin([
            watershed_workflow.utils.distance(seg_end, intersection_point.coords[0])
            for seg_end in [seg.coords[0], seg.coords[-1]]
        ])
        if seg_orientation_flag == 0:
            seg_tan = np.array(seg.coords[1]) - np.array(seg.coords[0])
        if seg_orientation_flag == 1:
            seg_tan = np.array(seg.coords[-2]) - np.array(seg.coords[-1])
        angle = -watershed_workflow.utils.angle(ref_seg_tan, seg_tan)

    elif type(seg) is watershed_workflow.river_tree.River:
        seg_tan = np.array(seg.segment.coords[-2]) - np.array(seg.segment.coords[-1])
        angle = -watershed_workflow.utils.angle(ref_seg_tan, seg_tan)

    if angle < 0:
        angle = angle + 360
    return angle


def rc_points_for_rt_point(rt_point, node, river_corr):
    """Returns the points (list of indices of coords) on the river-corridor-polygon for a given junction point on river tree."""
    #assert (node.segment.intersects(rt_point))
    rt_point_ind = _indexPointInSeg(node.segment, rt_point)

    # this give id of stream mesh element which has this rt point  at upstream end
    elem_ind = (len(node.segment.coords) - 1 - rt_point_ind) - 1
    elem = node.elements[elem_ind]

    if len(elem) == 4:
        # return two points
        rc_points = [river_corr.exterior.coords[ind] for ind in [elem[1], elem[2]]]
    elif len(elem) == 5:
        # return three points at junction
        rc_points = [river_corr.exterior.coords[ind] for ind in [elem[1], elem[2], elem[3]]]
    elif len(elem) == 6:
        # return three points at junction
        rc_points = [
            river_corr.exterior.coords[ind] for ind in [elem[1], elem[2], elem[3], elem[4]]
        ]
    else:
        raise ValueError('Too many points in elem?')

    return rc_points


def adjust_seg_for_rc(seg, river_corr, new_seg_point, integrate_rc=False):
    """Return the modified segment accomodating river-corridor-polygon (exclude river corridor or integrate with it)."""
    if not integrate_rc:
        len_scale = watershed_workflow.utils.distance(river_corr.exterior.coords[0],
                                                      river_corr.exterior.coords[-1])
        seg = seg.difference(river_corr.buffer(0.1
                                               * len_scale))  # removing seg points inside the RC
        if type(
                seg
        ) is shapely.geometry.MultiLineString:  # sometimes small portion of the segment can end up on the other side of the rc
            seg = seg[np.argmax([seg_.length for seg_ in seg])]
        seg_orientation_flag = np.argmin([
            watershed_workflow.utils.distance(seg_end, new_seg_point)
            for seg_end in [seg.coords[0], seg.coords[-1]]
        ])
        if seg_orientation_flag == 0:
            seg = shapely.geometry.LineString([new_seg_point, ] + seg.coords[1:])
        elif seg_orientation_flag == 1:
            seg = shapely.geometry.LineString(seg.coords[:-1] + [new_seg_point, ])
    else:
        seg_orientation_flag = np.argmin([
            watershed_workflow.utils.distance(seg_end, new_seg_point)
            for seg_end in [seg.coords[0], seg.coords[-1]]
        ])
        if seg_orientation_flag == 0:
            seg = shapely.geometry.LineString([new_seg_point, ] + seg.coords[:])
        elif seg_orientation_flag == 1:
            seg = shapely.geometry.LineString(seg.coords[:] + [new_seg_point, ])
    return seg


def adjust_hucs_for_river_corridors(hucs, rivers, river_corrs, integrate_rc=True, ax=None):
    """Adjusts hucs to accomodate river corridor polygons.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs(), will be modified in place.
    rivers : list(watershed_workflow.river_tree.River)
        A list of river tree object
    river_corrs : list(shapely.geometry.Polygons)
        A list of river corridor polygons for each river
    integrate_rc: bool, optional
        if false, will leave gap in the huc whereever rc crosses huc except at the overall outlet; 
        hence hucs.polygons() will break, this mode is to be used during triangulation to creates NodesEdges object 
        with a rc as a whole 
        if true, will extend the hucs-segments alogn the edge of quads 
    """
    for river, river_corr in zip(rivers, river_corrs):
        adjust_hucs_for_river_corridor(hucs, river, river_corr, integrate_rc=integrate_rc, ax=ax)


def adjust_hucs_for_river_corridor(hucs, river, river_corr, integrate_rc=True, ax=None):
    """Adjusts hucs to accomodate river corridor polygon.

    Parameters
    ----------
    hucs : SplitHUCs
        A split-form HUC object from, e.g., get_split_form_hucs()
    river : watershed_workflow.river_tree.River object
        river tree 
    river_corr : shapely.geometry.Polygons
        A river corridor polygon for given river
    integrate_rc: bool, optional
        If false, this will leave gap in the huc whereever rc crosses
        huc except at the overall outlet; hence hucs.polygons() will
        break, this mode is to be used during triangulation to creates
        NodesEdges object with a rc as a whole.  If true, will extend
        the hucs-segments alogn the edge of quads.

    """
    logging.info("  adjusting HUC boundary to include the river outlet segments")
    # rt = river tree; rc = river corridor
    river_mls = shapely.geometry.MultiLineString(list(river))  # for checking intersection

    huc_segs_adjusted = []  # keep track of already modified hucs
    for i, seg in enumerate(hucs.segments):
        is_unadjusted_outlet_point = False
        # check if this huc is part of already processed junction
        if i not in huc_segs_adjusted and seg.intersects(river_mls):
            logging.info("  ... found an intersection of river and huc seg")
            intersection_point = seg.intersection(river_mls)
            if type(intersection_point) is shapely.geometry.Point:
                parent_node = node_at_intersection(intersection_point, river)

                # making sure it is not a leaf node, though this check
                # fails if there is only one reach in the domain. So
                # this may fail for a single reach that begins and
                # ends on the boundary. -- fix me!
                if len(river) == 1 or len(parent_node.children) != 0:
                    is_unadjusted_outlet_point = True
            elif type(intersection_point) is shapely.geometry.MultiPoint:
                for point in intersection_point:
                    parent_node = node_at_intersection(point, river)
                    if len(parent_node.children) != 0:
                        is_unadjusted_outlet_point = True
                        intersection_point = point

        if is_unadjusted_outlet_point:
            logging.info("  ... it is an unadjusted outlet!")

            # hopefully no LineStrings or MultiPoints
            # assert(type(intersection_point) is shapely.geometry.Point)

            # find all the huc-segments at this junction
            intersection_segs = hucsegs_at_intersection(intersection_point, hucs)
            huc_segs_adjusted = huc_segs_adjusted + intersection_segs

            # mark them as modified (in the following steps)

            # find the downstream node (outgoing river reach) at this junction
            parent_node = node_at_intersection(intersection_point, river)
            if parent_node.parent == None:  # check if it is the outlet node for this river
                outlet_junction = True
            else:
                outlet_junction = False
            logging.info(f"  ... is there a parent to this? {parent_node.parent}")

            # find the index of the intersection point (at this
            # junction) on the rt-node-segment (needed to find rc
            # points)
            try:
                ind_intersection_point = _indexPointInSeg(parent_node.segment, intersection_point)
            except ValueError:
                err = IntersectionError()
                err.point = intersection_point
                err.river_seg = parent_node.segment
                err.river = river
                err.huc_segment = seg
                raise err

            if outlet_junction:
                logging.info('  found outlet junction')
                elem = parent_node.elements[0]
                # rc points at junction
                rc_points = [river_corr.exterior.coords[ind] for ind in [elem[0], elem[-1]]]

                # reference segment for angles
                ref_seg = shapely.geometry.LineString(
                    parent_node.segment.coords[ind_intersection_point - 2:])

                # orientations of hucs-segments
                seg_angles = [
                    angle_rivers_segs(ref_seg, hucs.segments[seg_id])
                    for seg_id in intersection_segs
                ]
                incoming_river_angles = [180, ]  # this hardcoded only for outlet junction

                # all line segments (hucs-segments and river-segments) at this junction
                all_segs = intersection_segs + [parent_node, ]
            else:
                # internal junction
                rc_points = rc_points_for_rt_point(intersection_point, parent_node, river_corr)

                # this is small part of the parent node.segment just downstream of the intersection point
                ref_seg = shapely.geometry.LineString(
                    parent_node.segment.coords[ind_intersection_point:ind_intersection_point + 2])
                seg_angles = [
                    angle_rivers_segs(ref_seg, hucs.segments[seg_id])
                    for seg_id in intersection_segs
                ]

                # orientations of incoming river-segments
                if intersection_point.intersects(
                        shapely.geometry.Point(parent_node.segment.coords[0])):
                    # huc and rt intersect at river-merging point
                    incoming_river_angles = [
                        angle_rivers_segs(ref_seg, child) for child in parent_node.children
                    ]
                    all_segs = intersection_segs + parent_node.children
                else:
                    upstream_seg = shapely.geometry.LineString(
                        parent_node.segment.coords[ind_intersection_point - 1:ind_intersection_point
                                                   + 1])
                    incoming_river_angles = [angle_rivers_segs(ref_seg, upstream_seg), ]
                    all_segs = intersection_segs + [parent_node, ]

            # orientation of all line segments (hucs-segments and river-segments) at this junction
            all_angles = seg_angles + incoming_river_angles

            junction_seg_angles = {all_segs[i]: all_angles[i] for i in range(len(all_segs))}

            # sort segments by their orientation angles
            junction_seg_angles_sorted = dict(
                sorted(junction_seg_angles.items(), key=lambda item: item[1]))

            # This polygon is used to remove overlapping
            # huc-segment and river corridor. To avoid issues at
            # snapped leaf node intersecting with this with this
            # huc segment, we create local river corridor polygon
            elem = parent_node.elements[0]
            river_corr_part = shapely.geometry.Polygon(
                [river_corr.exterior.coords[ind] for ind in [elem[0], elem[-1]]] + rc_points)

            if integrate_rc or outlet_junction:
                # Modify the huc boundary to integrate quad edges
                if len(hucs.segments) == 1:
                    key = 0
                    hucs.segments[key] = adjust_seg_for_rc(hucs.segments[key], river_corr_part,
                                                           rc_points[0])
                    hucs.segments[key] = adjust_seg_for_rc(hucs.segments[key], river_corr_part,
                                                           rc_points[1])

                else:
                    # identify which river corridor point is added to hucs-segment
                    rc_point_ind = 0
                    for key in junction_seg_angles_sorted.keys():
                        if type(key) is int:
                            logging.info(f"  modifying HUC segment {key}")
                            # removing part of huc-segment overlappig with rc and snapping huc-segment end to "right" rc point
                            hucs.segments[key] = adjust_seg_for_rc(hucs.segments[key],
                                                                   river_corr_part,
                                                                   rc_points[rc_point_ind])
                            key_hold = key
                        else:
                            rc_point_ind += 1
                            # extending huc-segment along smaller edge of the quad
                            hucs.segments[key_hold] = adjust_seg_for_rc(hucs.segments[key_hold],
                                                                        river_corr_part,
                                                                        rc_points[rc_point_ind],
                                                                        integrate_rc=integrate_rc
                                                                        or outlet_junction)

            else:
                # Just remove the part of of the huc-segment overlapping with the river corridor
                rc_point_ind = 0
                for key in junction_seg_angles_sorted.keys():
                    if type(key) is int:
                        logging.info(f"  modifying HUC Segment {key}")
                        hucs.segments[key] = adjust_seg_for_rc(hucs.segments[key], river_corr_part,
                                                               rc_points[rc_point_ind])
                    else:
                        rc_point_ind += 1
