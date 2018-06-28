"""Triangulates polygons"""
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

import shapely
import meshpy.triangle

import workflow.hydrography

def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]
def oneway_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)]

def triangulate(shp, max_area=0.0001, needs_refinement_func=None, debug=False):
    """Triangulates a domain.

    Refines based upon max_area, unless a function is provided.

    Such a function must be of the form:
 
    def needs_refinement_func(vertex_list, area)

    where this function takes the list of vertices making up the
    triangle and the area of the triangle.
    """
    points = [(x,y) for x,y in zip(*shp.boundary.xy)][:-1] # pop the last point as it is duplicate
    facets = round_trip_connect(0, len(points)-1)
    
    def my_needs_refinement_func(vertices, area):
        return bool(area > max_area)
    if needs_refinement_func is None:
        needs_refinement_func = my_needs_refinement_func
        
    info = meshpy.triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)
    mesh = meshpy.triangle.build(info, debug, refinement_func=needs_refinement_func)

    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    return mesh_points, mesh_tris

def _checkclose(all_points):
    someclose = False
    for i,p in enumerate(all_points):
        for j,q in enumerate(all_points):
            if j > i:
                if np.allclose(p,q, 1.e-7):
                    someclose = True
                    print("GOT CLOSE", i,j, p, q)
    return someclose

def triangulate_with_rivers(shp, reaches, **kwargs):
    """Triangulates a domain with reaches.

    Refines based upon max_area, unless a function is provided.

    Such a function must be of the form:
 
    def refinement_func(vertex_list, area)

    where this function takes the list of vertices making up the
    triangle and the area of the triangle.
    """
    # make the multiline object
    river_networks = workflow.hydrography.graph(reaches)

    # find the where water enters/leaves the domain
    spillpoints = shp.boundary.intersection(reaches)
    assert(type(spillpoints) is shapely.geometry.MultiPoint)
    spillpoint_indices = [next(i for (i,c) in enumerate(shp.boundary.coords[:-1]) if np.allclose(sp.coords, c, 1.e-7)) for sp in spillpoints]

    # set up the boundary points and facets
    g_points = [np.array(shp.boundary.coords[:-1]),] # pop the last point as it is duplicate
    g_facets = [np.array(round_trip_connect(0, len(g_points[0])-1), 'i'),]
    assert(g_facets[0].min() == 0)
    assert(g_facets[0].max() == len(g_points[0])-1)
    

    print("Triangulating polygon: %i points and %i facets"%(len(g_points[0]), len(g_facets[0])))
    print("  with spill points at:")
    print("  %r"%spillpoints)
    print("  %r"%spillpoint_indices)
    
    def get_points_and_connections(reaches, i, subnetwork):
        """Helper to collect points and facets"""
        my_points = np.array(reaches[i].coords)
        my_facets = np.array(oneway_trip_connect(0,len(my_points)-1),'i')
        assert(len(my_points) == len(my_facets)+1)
        if len(subnetwork.keys()) is 0:
            # leaf node
            # check if start node is an outlet
            try:
                start_is_spillpoint = next(i for (i,sp) in enumerate(spillpoints) if np.allclose(my_points[0], sp.coords, 1.e-7))
            except StopIteration:
                pass
            else:
                print("found inlet")
                my_points = my_points[1:]
                my_facets = my_facets - 1
                my_facets[0][0] = - 10 - start_is_spillpoint # negative at -10 or lower indicates map to spillpoint index
            #print("leaf  %i with total %i points and %i facets"%(i,len(my_points), len(my_facets)))
            #print("  facets:", my_facets)
            return my_points, my_facets
        else:
            points = []
            n_points = 0
            facets = []

            #print("interior %i with children: %r"%(i,subnetwork.keys()))
            #print("   and  %i local points and %i local facets"%(len(my_points), len(my_facets)))
            for j, child in subnetwork.items():
                p, f = get_points_and_connections(reaches, j, child)
                assert(np.allclose(p[-1], my_points[0], 1.e-7))

                # take off the last, it will be my first
                points.append(p[:-1])
                
                # shift facets
                f = np.where(f >= 0, f + n_points, f)

                # set the last facet to end at null until we know the actual index
                f[-1][-1] = -1

                facets.append(f)
                n_points += len(p)-1

            # note n_points now is the index of my first point, which is the junction
            points.append(my_points)
            facets.append(my_facets+n_points)

            fin_points = np.concatenate(points)
            fin_facets = np.concatenate(facets)
            fin_facets = np.where(fin_facets == -1, n_points, fin_facets)
            #print("interior %i with total %i points and %i facets"%(i,len(fin_points), len(fin_facets)))
            #print("  facets:", fin_facets)
            
            #assert(fin_facets.min() == 0)
            assert(fin_facets.max() == len(fin_points)-1)
            #fail = _checkclose(fin_points)
            #if fail:
            #    print(fin_points)
            #    assert(False)
            return fin_points, fin_facets

    n_g_points = len(g_points[0])
    for network in river_networks:
        k,v = list(network.items())[0]
        p, f = get_points_and_connections(reaches, k, v)

        # check if the end node is an outlet
        try:
            end_is_spillpoint = next(i for (i,sp) in enumerate(spillpoints) if np.allclose(p[-1], sp.coords, 1.e-7))
        except StopIteration:
            pass
        else:
            print("found outlet")
            p = p[:-1]
            f[-1][-1] = -10 - end_is_spillpoint

        g_facets.append(np.where(f >= 0, f + n_g_points, f))
        g_points.append(p)
        n_g_points += len(p)

    all_points = np.concatenate(g_points)
    all_facets = np.concatenate(g_facets)

    # fix the spillpoints
    for i,sp in enumerate(spillpoint_indices):
        all_facets = np.where(all_facets == -10 -i, sp, all_facets)
    
    # plot to check the data
    if False:
        for f in all_facets:
            plt.plot([all_points[f[0],0],all_points[f[1],0]],
                     [all_points[f[0],1],all_points[f[1],1]], 'r')
        plt.scatter(all_points[:,0], all_points[:,1], color='r', marker='x')
        plt.show()
    
    if False:
        print(all_points.shape)
        print(all_facets.shape)
        for i,p in enumerate(all_points):
            for j,q in enumerate(all_points):
                if j > i:
                    if np.allclose(p,q, 1.e-7):
                        print("GOT CLOSE", p, q)
        
    centroid = np.mean(all_points, axis=0)
    assert(len(centroid) == 2)

    all_points = all_points - np.expand_dims(centroid,0)

    info = meshpy.triangle.MeshInfo()
    print("Triangulating: %i points and %i facets"%(len(all_points), len(all_facets)))
    np.savetxt("points.txt", all_points)
    np.savetxt("facets.txt", all_facets)

    pdata = [tuple(p) for p in all_points]
    info.set_points(pdata)
    fdata = [[int(i) for i in f] for f in all_facets]
    info.set_facets(fdata)

    print("building mesh...")
    mesh = meshpy.triangle.build(info, **kwargs)
    print("  built")

    mesh_points = np.array(mesh.points) + np.expand_dims(centroid,0)
    mesh_tris = np.array(mesh.elements)
    return mesh_points, mesh_tris




if __name__ == "__main__":
    import workflow.conf
    import workflow.smooth
    import workflow.download
    import shapely
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    # load shapefiles for all HUC 12s in the Obed HUC 8.
    profile, hucs = workflow.conf.load_hucs_in('06010208', 12)

    # convert to shapely
    hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]

    # intersect, finding shared boundaries
    uniques, intersections = workflow.smooth.intersect_and_split(hucs_s)

    # smooth/simplify/resample to a given spacing (in meters)
    uniques_sm, intersections_sm = workflow.smooth.smooth(uniques,intersections,100)

    # recombine
    hucs_sm = workflow.smooth.recombine(uniques_sm, intersections_sm)

    # triangulate (to a refinement with max_area, units a bit unclear?
    # I believe these should be degrees^2, but the magnitude seems
    # wrong for that.  Takes some fiddling.
    for huc in hucs_sm:
        mesh_points, mesh_tris = triangulate(huc, max_area=0.0001)
        plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    plt.show()
        
