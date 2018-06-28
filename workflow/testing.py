from matplotlib import pyplot as plt
import shapely.geometry
import shapely.ops

import workflow
import workflow.smooth
import workflow.conf
import workflow.hydrography
import workflow.triangulate
import workflow.plot
import workflow.colors

def get_hucs():
    profile, hucs = workflow.conf.load_hucs_in('06010208', 12)
    return profile, hucs

def get_shapes():
    profile, hucs = workflow.conf.load_hucs_in('06010208', 12)

    # convert to shapely
    hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]

    # intersect
    uniques, intersections = workflow.smooth.intersect_and_split(hucs_s)
    #_plot(uniques,intersections,'-x')

    # smooth
    uniques_sm, intersections_sm = workflow.smooth.smooth(uniques,intersections,100)
    #_plot(uniques_sm,intersections_sm,'-+')

    # recombine
    hucs_sm = workflow.smooth.recombine(uniques_sm, intersections_sm)
    return hucs_sm

def river_sorting():
    plt.figure()

    # load hucs, convert to shapely
    hprofile, hucs = workflow.conf.load_hucs_in('06010208', 12)
    hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]

    # load rivers, convert to shapely
    rprofile, rivers = workflow.conf.load_hydro('06010208')
    rivers_s = [shapely.geometry.shape(r['geometry']) for r in rivers]
    rivers_part = workflow.hydrography.sort(hucs_s, rivers_s)
    
    # plot
    cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
    for i, (s,r) in enumerate(zip(hucs_s, rivers_part)):
        c = cm(i)
        c_huc = workflow.colors.darken(c)
        workflow.plot.huc(s, color=c_huc, style='-x')
        workflow.plot.river(r, color=c)
    plt.show()

def river_simplify_then_sort():
    plt.figure()

    # load hucs, convert to shapely
    hprofile, hucs = workflow.conf.load_hucs_in('06010208', 12)
    hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]

    # simplify the hucs
    uniques, intersections = workflow.smooth.intersect_and_split(hucs_s)
    uniques_sm, intersections_sm = workflow.smooth.simplify(uniques,intersections,1./111000.*10)  # units = degrees
    hucs_s = workflow.smooth.recombine(uniques_sm, intersections_sm)

    # load rivers, convert to shapely
    rprofile, rivers = workflow.conf.load_hydro('06010208')
    rivers_s = [shapely.geometry.shape(r['geometry']) for r in rivers]

    # simplify the rivers
    rivers_s = workflow.hydrography.simplify(rivers_s, 1./111000.*100)  # units = degrees

    # # plot
    # cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
    # for i, s in enumerate(hucs_s):
    #     c = cm(i)
    #     c_huc = workflow.colors.darken(c)
    #     workflow.plot.huc(s, color=c_huc, style='-x')

    # workflow.plot.river(rivers_s, color='b', style='-+')
    # plt.show()

    
    # sort
    rivers_part = workflow.hydrography.sort(hucs_s, rivers_s)
    
    # plot
    cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
    for i, (s,r) in enumerate(zip(hucs_s, rivers_part)):
        c = cm(i)
        c_huc = workflow.colors.darken(c)
        workflow.plot.huc(s, color=c_huc, style='-x')
        workflow.plot.river(r, color=c)
    plt.show()



def river_sort_then_simplify():
    plt.figure()

    # load hucs, convert to shapely
    print("loading hucs")
    hprofile, hucs = workflow.conf.load_hucs_in('06010208', 12)
    hucs_s = [shapely.geometry.shape(s['geometry']) for s in hucs]
    assert(all(type(h) is shapely.geometry.Polygon for h in hucs_s))

    # load rivers, convert to shapely
    print("loading rivers")
    rprofile, rivers = workflow.conf.load_hydro('06010208')
    rivers_s = [shapely.geometry.shape(r['geometry']) for r in rivers]

    # combine rivers to improve simplify.  this step is lossless currently
    assert(all(type(r) is shapely.geometry.LineString for r in rivers_s))
    rivers_s = list(shapely.ops.linemerge(rivers_s))
    assert(all(type(r) is shapely.geometry.LineString for r in rivers_s))

    # split polygons into spine and boundary
    uniques, intersections = workflow.smooth.intersect_and_split(hucs_s)
    assert(all(type(u) is shapely.geometry.LineString or
               type(u) is shapely.geometry.MultiLineString or
               type(u) is type(None) for u in uniques))
    assert(all(type(s) is shapely.geometry.LineString or
               type(s) is shapely.geometry.MultiLineString or
               type(s) is type(None) for i in intersections for s in i))

    # intersect rivers with spine, adding points at the intersection to both
    all_segs = []
    for s in uniques:
        if type(s) is shapely.geometry.LineString:
            all_segs.append(s)
        elif type(s) is shapely.geometry.MultiLineString:
            for seg in s:
                assert(type(seg) is shapely.geometry.LineString)
                all_segs.append(seg)
    for i in intersections:
        for s in i:
            if type(s) is shapely.geometry.LineString:
                all_segs.append(s)
            elif type(s) is shapely.geometry.MultiLineString:
                for seg in s:
                    assert(type(seg) is shapely.geometry.LineString)
                    all_segs.append(seg)
                    
    assert(all(type(r) is shapely.geometry.LineString for r in rivers_s))
    assert(all(type(s) is shapely.geometry.LineString for s in all_segs))
    workflow.hydrography.split_spine(all_segs, rivers_s)
    assert(all(type(r) is shapely.geometry.LineString for r in rivers_s))
    assert(all(type(s) is shapely.geometry.LineString for s in all_segs))

    # smooth the whole durn thing at the coarse resolution
    all_segs = rivers_s + all_segs
    all_segs = shapely.geometry.MultiLineString(all_segs)
    all_segs_simp = list(all_segs.simplify(1./111000.*100))  # units = degrees
    assert(len(all_segs_simp) == len(all_segs))
    assert(all(type(s) is shapely.geometry.LineString for s in all_segs_simp))

    # restructure back to the original format
    # -- pop the rivers
    rivers_simp = all_segs_simp[0:len(rivers_s)]
    all_segs_simp = all_segs_simp[len(rivers_s):]

    # -- next the uniques
    uniques_simp = [None,]*len(uniques)
    pos = 0
    for i,u in enumerate(uniques):
        if type(u) is shapely.geometry.LineString:
            uniques_simp[i] = all_segs_simp[pos]
            pos += 1
        elif type(u) is shapely.geometry.MultiLineString:
            num_segs = len(u)
            uniques_simp[i] = shapely.geometry.MultiLineString(all_segs_simp[pos:pos+num_segs])
            pos += num_segs

    # -- finally the intersections
    intersections_simp = [[None for i in range(len(intersections))] for j in range(len(intersections))]
    for i,inter in enumerate(intersections):
        for j,u in enumerate(inter):
            if type(u) is shapely.geometry.LineString:
                intersections_simp[i][j] = all_segs_simp[pos]
                pos += 1
            elif type(u) is shapely.geometry.MultiLineString:
                num_segs = len(u)
                intersections_simp[i][j] = shapely.geometry.MultiLineString(all_segs_simp[pos:pos+num_segs])
                pos += num_segs
    # -- check the final tally -- we better have gotten them all            
    assert(pos == len(all_segs_simp))

    # recombine the simplified objects
    hucs_simp = workflow.smooth.recombine(uniques_simp, intersections_simp)

    # sort rivers by containing poly
    rivers_part = workflow.hydrography.sort_precut(hucs_simp, rivers_simp)

    # plot to check before triangulating
    workflow.plot.huc(hucs_simp[0], style='-x')
    workflow.plot.river(rivers_part[0], style='-+')
    plt.show()
    
    # triangulate
    print("triangulating:")
    def refine(*args):
        return False
    mesh_points, mesh_tris = workflow.triangulate.triangulate_with_rivers(hucs_simp[0], rivers_part[0], needs_refinement_func=refine)


    # plot
    workflow.plot.tri(mesh_points, mesh_tris)
    cm = workflow.colors.cm_mapper(0,len(hucs_s)-1)
    for i, (s,r) in enumerate(zip(hucs_simp, rivers_part)):
        c = cm(i)
        c_huc = workflow.colors.darken(c)
        workflow.plot.huc(s, color=c_huc, style='-x')
        workflow.plot.river(r, color=c)
    plt.show()



    
def river_triangulation():
    plt.figure()

    # load 1 hucs, convert to shapely
    hprofile, hucs = workflow.conf.load_hucs_in('06010208', 12)
    hucs_s = [shapely.geometry.shape(hucs[0]['geometry']),]
    hucs_s[0] = hucs_s[0].simplify(1./111000.*100)

    # load rivers, convert to shapely
    rprofile, rivers = workflow.conf.load_hydro('06010208')
    rivers_s = [shapely.geometry.shape(r['geometry']) for r in rivers]
    rivers_s = shapely.ops.linemerge(rivers_s)

    rivers_part = workflow.hydrography.sort(hucs_s, rivers_s)
    rivers_sm = workflow.hydrography.simplify(rivers_part, 1./111000.*100) # converts 100m to degrees
    river_list = shapely.geometry.MultiLineString(rivers_sm[0])

    # triangulate
    def refine(*args):
        return False
    mesh_points, mesh_tris = workflow.triangulate.triangulate_with_rivers(hucs_s[0], river_list, needs_refinement_func=refine)

    workflow.plot.tris(mesh_points, mesh_tris)
    workflow.plot.huc(hucs_s[0], color='m')
    workflow.plot.river(river_list)
    plt.show()
    
if __name__ == "__main__":
    #river_triangulation()
    #river_sorting()
    river_sort_then_simplify()
