import numpy as np
import laspy
import pyvista as pv
from scipy.spatial import Delaunay


folder  = 'H:\\pytreedb\\pointclouds'
file    = 'QuePet_BR01_10_2019-07-03_q2_TLS-on.laz'

# Load .las or .laz file
inFile = laspy.read(f'{folder}\\{file}')

# Get points from file
arr = inFile.xyz
arr = arr[0:-1:200,:]

del inFile, folder, file
# Perform Delaunay triangulation on the point cloud
tri = Delaunay(arr)

# Create a PyVista mesh from the Delaunay triangulation
mesh = pv.PolyData(arr, np.c_[np.full(len(tri.simplices), 3), tri.simplices])

tri_mesh = mesh.triangulate()

# Perform mesh decimation
decimated_mesh = tri_mesh.decimate(0.5)  # Reduce to 50% of the original number of vertices

# Perform mesh smoothing
smooth_mesh = decimated_mesh.smooth(n_iter=5)  # Adjust this parameter as needed

# Save the smoothed, decimated mesh as a .obj file
smooth_mesh.save("D:\\Models\\output.obj")

