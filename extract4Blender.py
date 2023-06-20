import pandas as pd

# set Variables
dwPathCSV   = 'H:\\Blender\\csv4blender\\dw.csv'
treePathCSV = 'H:\\Blender\\csv4blender\\tree.csv'
dwPathObj   = 'H:\\Blender\\layingCylinder.obj'
treePathObj = 'H:\\Blender\\standingCylinder.obj'
# load CSV
dw      = pd.read_csv(dwPathCSV)
tree    = pd.read_csv(treePathCSV)

for i in range(len(dw[dw.keys()[0]])):
    print(dw.iloc[i])
    d = dw.iloc[i]

    location = [d['Xc']-d['Xo'],d['Yc']-d['Yo'], 0]
    if kd != None:
        location, _, _ = kd.find(location)
    rotation = [0,0,d['A']]
    scale = [d['D1']/1000,d['L'],d['D1']/1000]
    obj = import_obj(filepath, location=location, rotation=rotation, scale=scale)
    obs.append(obj)