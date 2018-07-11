from matplotlib import pyplot as plt

def tri(points, tris, color='gray'):
    plt.triplot(points[:,0], points[:,1], tris, color=color)

def huc(huc, color=None, style='-', linewidth=1):
    if color is not None:
        plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], style, color=color, linewidth=linewidth)
    else:
        plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], style, linewidth=linewidth)


def hucs(hucs, color=None, style='-', linewidth=1):
    for huc in hucs.polygons():
        plt.plot(huc.exterior.xy[0], huc.exterior.xy[1], style, color=color, linewidth=linewidth)

        

def river(river, color='b', style='-', linewidth=1):
    for r in river:
        plt.plot(r.xy[0], r.xy[1], style, color=color, linewidth=linewidth)

        
def points(points, **kwargs):
    x = [p.xy[0][0] for p in points]
    y = [p.xy[1][0] for p in points]
    plt.scatter(x,y,**kwargs)
