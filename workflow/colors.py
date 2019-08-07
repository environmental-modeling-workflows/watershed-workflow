import matplotlib
import matplotlib.colors
import matplotlib.cm
import numpy as np

# black-zero jet is jet, but with the 0-value set to black, with an immediate jump to blue
def blackzerojet_cmap(data):
    blackzerojet_dict = {'blue': [[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.5],
                                  [0.11, 1, 1],
                                  [0.34000000000000002, 1, 1],
                                  [0.65000000000000002, 0, 0],
                                  [1, 0, 0]],
                        'green': [[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.125, 0, 0],
                                  [0.375, 1, 1],
                                  [0.64000000000000001, 1, 1],
                                  [0.91000000000000003, 0, 0],
                                  [1, 0, 0]],
                          'red': [[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.34999999999999998, 0, 0],
                                  [0.66000000000000003, 1, 1],
                                  [0.89000000000000001, 1, 1],
                                  [1, 0.5, 0.5]]
                          }
    minval = data[np.where(data > 0.)[0]].min(); print(minval)
    maxval = data[np.where(data > 0.)[0]].max(); print(maxval)
    oneminval = .9*minval/maxval
    for color in ['blue', 'green', 'red']:
        for i in range(1,len(blackzerojet_dict[color])):
            blackzerojet_dict[color][i][0] = blackzerojet_dict[color][i][0]*(1-oneminval) + oneminval

    return matplotlib.colors.LinearSegmentedColormap('blackzerojet', blackzerojet_dict)

# ice color map
def ice_cmap():
    x = np.linspace(0,1,7)
    b = np.array([1,1,1,1,1,0.8,0.6])
    g = np.array([1,0.993,0.973,0.94,0.893,0.667,0.48])
    r = np.array([1,0.8,0.6,0.5,0.2,0.,0.])

    bb = np.array([x,b,b]).transpose()
    gg = np.array([x,g,g]).transpose()
    rr = np.array([x,r,r]).transpose()
    ice_dict = {'blue': bb, 'green': gg, 'red': rr}
    return matplotlib.colors.LinearSegmentedColormap('ice', ice_dict)

# water color map
def water_cmap():
    x = np.linspace(0,1,8)
    b = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    g = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    r = np.array([1.0, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])

    bb = np.array([x,b,b]).transpose()
    gg = np.array([x,g,g]).transpose()
    rr = np.array([x,r,r]).transpose()
    water_dict = {'blue': bb, 'green': gg, 'red': rr}
    return matplotlib.colors.LinearSegmentedColormap('water', water_dict)

# water color map
def gas_cmap():
    x = np.linspace(0,1,8)
    r = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    #    g = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    b = np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    g = np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])

    bb = np.array([x,b,b]).transpose()
    gg = np.array([x,g,g]).transpose()
    rr = np.array([x,r,r]).transpose()
    gas_dict = {'blue': bb, 'green': gg, 'red': rr}
    return matplotlib.colors.LinearSegmentedColormap('gas', gas_dict)


# jet-by-index
def cm_mapper(vmin=0., vmax=1., cmap=matplotlib.cm.jet):
    """Factory for a Scalar Mappable, which gives a color based upon a scalar value.

    Typical Usage:
      >>> # plots 11 lines, with color scaled by index into jet
      >>> mapper = cm_mapper(vmin=0, vmax=10, cmap=matplotlib.cm.jet)
      >>> for i in range(11):
      ...     data = np.load('data_%03d.npy'%i)
      ...     plt.plot(x, data, color=mapper(i))
      ...
      >>> plt.show()
    """

    norm = matplotlib.colors.Normalize(vmin, vmax)
    sm = matplotlib.cm.ScalarMappable(norm, cmap)
    def mapper(value):
        return sm.to_rgba(value)
    return mapper


def float_list_type(mystring):
    """Convert string-form list of doubles into list of doubles."""
    colors = []
    for f in mystring.strip("(").strip(")").strip("[").strip("]").split(","):
        try:
            colors.append(float(f))
        except:
            colors.append(f)
    return colors


def desaturate(color, amount=0.4, is_hsv=False):
    if not is_hsv:
        hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))
    else:
        hsv = color

    hsv[1] = max(0,hsv[1] - amount)
    return matplotlib.colors.hsv_to_rgb(hsv)

def darken(color, fraction=0.6):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.maximum(rgb - fraction*rgb,0))

def lighten(color, fraction=0.6):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.minimum(rgb + fraction*(1-rgb),1))


import collections
nlcd_color_map_values = collections.OrderedDict({
    0  : '#00000000',
    11 : '#526095FF',     # Open Water
    12 : '#FFFFFFFF',     # Perennial Ice/Snow
    21 : '#D28170FF',     # Low Intensity Residential
    22 : '#EE0006FF',     # High Intensity Residential
    23 : '#990009FF',     # Commercial/Industrial/Transportation
    31 : '#BFB8B1FF',     # Bare Rock/Sand/Clay
    32 : '#969798FF',     # Quarries/Strip Mines/Gravel Pits
    33 : '#382959FF',     # Transitional
    41 : '#579D57FF',     # Deciduous Forest
    42 : '#2A6B3DFF',     # Evergreen Forest
    43 : '#A6BF7BFF',     # Mixed Forest
    51 : '#BAA65CFF',     # Dwarf Shrubland
    52 : '#BAA65CFF',     # Shrubland
    61 : '#45511FFF',     # Orchards/Vineyards/Other
    71 : '#D0CFAAFF',     # Grasslands/Herbaceous
    81 : '#CCC82FFF',     # Pasture/Hay
    82 : '#9D5D1DFF',     # Row Crops
    83 : '#CD9747FF',     # Small Grains
    84 : '#A7AB9FFF',     # Fallow
    85 : '#E68A2AFF',     # Urban/Recreational Grasses
    91 : '#B6D8F5FF',     # Woody Wetlands
    92 : '#B6D8F5FF'})    # Emergent Herbaceous Wetlands

_nlcd_labels = collections.OrderedDict({
    0  : 'None',
    11 : 'Open Water',
    12 : 'Perennial Ice/Snow',
    21 : 'Low Intensity Residential',
    22 : 'High Intensity Residential',
    23 : 'Commercial/Industrial/Transporation',
    31 : 'Bare Rock/Sand/Clay',
    32 : 'Quarries/Strip Mines/Gravel Pits',
    33 : 'Transitional',
    41 : 'Deciduous Forest',
    42 : 'Evergreen Forest',
    43 : 'Mixed Forest',
    51 : 'Dwarf Shrubland',
    52 : 'Shrubland',
    61 : 'Orchards/Vineyards/Other',
    71 : 'Grasslands/Herbaceous',
    81 : 'Pasture/Hay',
    82 : 'Row Crops',
    83 : 'Small Grains',
    84 : 'Fallow',
    85 : 'Urban/Recreational Grasses',
    91 : 'Woody Wetlands',
    92 : 'Emergent Herbaceous Wetlands'})

nlcd_cmap = matplotlib.colors.ListedColormap(list(nlcd_color_map_values.values()))

_nlcd_indices = np.array(list(_nlcd_labels.keys()),'d')
#_nlcd_ind_bins = [_nlcd_indices[0] - .5 * (_nlcd_indices[1] - _nlcd_indices[0]),] + \
#                 list((_nlcd_indices[1:] + _nlcd_indices[:-1])/2.) + \
#                 [_nlcd_indices[-1] + .5 * (_nlcd_indices[-1] - _nlcd_indices[-2]),]
#nlcd_norm = matplotlib.colors.BoundaryNorm(_nlcd_ind_bins, len(nlcd_color_map_values.keys()))
nlcd_norm = matplotlib.colors.BoundaryNorm(list(_nlcd_labels.keys())+[93,], len(_nlcd_labels))
nlcd_ticks = list(_nlcd_labels.keys()) + [93,]
nlcd_labels = list(_nlcd_labels.values()) + ['',]

                                                                
