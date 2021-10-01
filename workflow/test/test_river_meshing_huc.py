import workflow.river_tree
import shapely
import numpy as np
from matplotlib import pyplot as plt
import workflow
import workflow.source_list
import workflow.crs

sources = workflow.source_list.get_default_sources()
crs = workflow.crs.default_crs()

_, huc = workflow.get_split_form_hucs(sources['hydrography'], '140200010204', level=12, out_crs=crs)
_, reaches = workflow.get_reaches(sources['hydrography'], '140200010204', out_crs=crs)
rivers = workflow.simplify_and_prune(huc,reaches, simplify=100, ignore_small_rivers=10)

river = rivers[0]
corr1, corr2, corr3 = workflow.river_tree.create_river_corridor(river, 6)
corr = corr3

plt.ion()
fig = plt.figure()
plt.show()
ax = fig.add_subplot(111)
ax.plot(huc.exterior().exterior.xy[0], huc.exterior().exterior.xy[1], 'k')
for reach in rivers[0].dfs():
    ax.plot(reach.xy[0], reach.xy[1], 'r-x')
ax.plot(corr.exterior.xy[0], corr.exterior.xy[1], 'b-x')

coords = list(corr.exterior.coords)
conn = workflow.river_tree.to_quads(river, 6, huc, coords[:-1], ax)
for elem in conn:
    looped_conn = elem[:]
    looped_conn.append(elem[0])
    cc = np.array([coords[n] for n in looped_conn])
    ax.plot(cc[:,0], cc[:,1], 'g-^')

plt.show()
