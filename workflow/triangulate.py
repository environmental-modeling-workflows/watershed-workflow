"""Triangulates polygons"""


import meshpy.triangle
import numpy as np
import numpy.linalg as la

def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]


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
    #info.set_holes([(0, 0)])
    info.set_facets(facets)

    mesh = meshpy.triangle.build(info, debug, refinement_func=needs_refinement_func)

    mesh_points = np.array(mesh.points)
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
        
