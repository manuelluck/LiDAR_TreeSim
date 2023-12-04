# import numpy as np
import laspy
# import alphashape
# from scipy.spatial import Delaunay
import open3d as o3d



folder  = 'H:\\pytreedb\\pointclouds'
file    = 'QuePet_BR01_10_2019-07-03_q2_TLS-on.laz'


# Load .las or .laz file
inFile = laspy.read(f'{folder}\\{file}')

# Get points from file
coordinates = inFile.xyz[0:-1:100,:]

# Create an Open3D PointCloud object from the numpy array
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coordinates)

# Estimate normals
pcd.estimate_normals()

# Create mesh using Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15)


# Create a voxel grid from the point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

# Save as .ply file
o3d.io.write_voxel_grid("D:\\Models\\voxel.ply", voxel_grid)

# Decimate and smooth the mesh
#dec_mesh = mesh.simplify_quadric_decimation(100000)
#smooth_mesh = dec_mesh.filter_smooth_simple()

# Save as .obj file
#o3d.io.write_triangle_mesh("D:\Models\Smoothed.obj", smooth_mesh)
#o3d.io.write_triangle_mesh("D:\Models\Simplified.obj", dec_mesh)
o3d.io.write_triangle_mesh("D:\\Models\\Raw.obj", mesh)


# Load .las or .laz file
#inFile = laspy.read(f'{folder}\\{file}')

# Get points from file
#arr = inFile.xyz

#del inFile, folder, file

# Define alpha parameter (you may need to adjust this)
#for alpha in [0.05,0.1,0.2,0.4,0.5,0.75,1]:

#    # Calculate optimized alpha shape
#    shape = alphashape.alphashape(arr, alpha)

#    # Save the alpha shape as a .obj file
#    shape.export(f'D:\\Models\\alphashape_{alpha}.obj')






# Perform Delaunay triangulation on the point cloud
#tri = Delaunay(arr)

#mesh = trimesh.Trimesh(vertices=arr, faces=tri.simplices)

#mesh.export("D:\\Models\\mesh.obj")


#tri_mesh = mesh.triangulate()

# Perform mesh decimation
#decimated_mesh = tri_mesh.decimate(0.5)  # Reduce to 50% of the original number of vertices

# Perform mesh smoothing
#smooth_mesh = decimated_mesh.smooth(n_iter=5)  # Adjust this parameter as needed

# Save the smoothed, decimated mesh as a .obj file
#smooth_mesh.save("D:\\Models\\smooth_mesh.obj")

