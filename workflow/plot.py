import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as pltc
import shapely
import workflow.colors


def huc(huc, color=None, style='-', linewidth=1):
    if color is not None:
        plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], style, color=color, linewidth=linewidth)
    else:
        plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], style, linewidth=linewidth)


def hucs(hucs, color=None, style='-', linewidth=1):
    for huc in hucs.polygons():
        plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], style, color=color, linewidth=linewidth)


def rivers(rivers, color=None, style='-', linewidth=1):
    if style.endswith('-') or style.endswith('.'):
        marker = None
    else:
        marker = style[-1]
        if len(style) is 1:
            style = None
        else:
            style = style[:-1]        

    if len(rivers) is 0:
        return

    # gather lines
    if type(rivers[0]) is workflow.tree.Tree:
        lines = []
        for tree in rivers:
            lines.extend([river.coords[:] for river in tree.dfs()])
    elif type(rivers[0]) is shapely.geometry.LineString:
        lines = [river.coords[:] for river in rivers]

    # plot lines
    if style is not None:
        lc = pltc.LineCollection(lines, colors=color, linewidths=linewidth, linestyle=style)
        plt.gca().add_collection(lc)
    if marker is not None:
        marked_points = np.concatenate([np.array(l) for l in lines])
        assert(marked_points.shape[-1] == 2)
        plt.scatter(marked_points[:,0], marked_points[:,1], c=color, marker=marker)
        
    plt.gca().autoscale()
    plt.gca().margins(0.1)


def river(river, color='b', style='-', linewidth=1):
    for r in river:
        plt.plot(r.xy[0], r.xy[1], style, color=color, linewidth=linewidth)
        
def points(points, **kwargs):
    x = [p.xy[0][0] for p in points]
    y = [p.xy[1][0] for p in points]
    plt.scatter(x,y,**kwargs)

def triangulation(points, tris, color=None, linewidth=1, edgecolor='gray', colorbar=True):
    monocolor = True
    if color is None:
        if points.shape[1] is 3:
            monocolor = False
        else:
            color = 'gray'
        
    if monocolor:
        plt.triplot(points[:,0], points[:,1], tris, color=color, linewidth=linewidth)
    else:
        plt.tripcolor(points[:,0], points[:,1], tris, points[:,2], linewidth=linewidth, edgecolor=edgecolor)

    if colorbar:
        plt.colorbar()

    
    
        
    

    
