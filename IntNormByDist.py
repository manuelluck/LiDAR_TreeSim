import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.cluster import KMeans

plt.switch_backend('tkagg')

# Load the .laz file
inFile  = laspy.read("C:\\Users\\luckmanu\\Desktop\\2023-06-29_09-45-16_a0_006_100pct_height_world_clip2Traj_ground_heb_norm.laz")
binSize = 0.1

# Get the point ext
points  = inFile.points



points.x[0]
points.y[0]
points.z[0]
points.