import laspy
import os
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

loadNew     = False
pointsFile  = 'H:\\data\\Python\\Helios\\a0_006_points.npy'
labelsFile  = 'H:\\data\\Python\\Helios\\a0_006_labels.npy'

if loadNew:
    cloudFolder = Path('H:\\CCOutput\\Simulated\\')

    points = []
    labels = []
    for file in os.listdir(cloudFolder):
        inFile  = laspy.read(str(cloudFolder.joinpath(file)))
        for p in range(inFile.__len__()):
            points.append([inFile[p].points.array[0]*inFile.header.x_scale+inFile.header.x_offset,
                           inFile[p].points.array[1]*inFile.header.y_scale+inFile.header.y_offset,
                           inFile[p].points.array[2]*inFile.header.z_scale+inFile.header.z_offset])
            labels.append(inFile[p].points.array[11])
    points = np.vstack(points)

    np.save(pointsFile,points)
    np.save(labelsFile,labels)

else:
    points = np.load(pointsFile)
    labels = np.load(labelsFile)


fig, axs = plt.subplots(1,len(np.unique(labels)))
for idx,label in enumerate(np.unique(labels)):
    axs[idx].scatter(points[labels==label,0],points[labels==label,1])

