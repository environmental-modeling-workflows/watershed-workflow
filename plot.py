import os
import visit

dirname = "/Users/uec/research/water/data/meshing/data/meshes/06010208/12"
db = os.listdir(dirname)
dbf = [os.path.join(dirname, d) for d in db]

opacity = 0.5
vmin = 100
vmax = 1000

for d in dbf:
    visit.OpenDatabase(d)
    visit.AddPlot("Mesh", "mesh")
    ma = visit.MeshAttributes()
    ma.legendFlag = 0
    ma.opaqueMode = ma.On
    ma.opacity = opacity
    visit.SetPlotOptions(ma)

    visit.DefineScalarExpression("z", "coord(mesh)[2]")
    visit.AddPlot('Pseudocolor', "z")
    pa = visit.PseudocolorAttributes()
    pa.minFlag = 1
    pa.min = vmin
    pa.maxFlag = 1
    pa.max = vmax
    visit.SetPlotOptions(pa)
    

    
