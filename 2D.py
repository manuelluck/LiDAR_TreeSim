import laspy
import numpy as np
import matplotlib.pyplot as plt

# Load the .laz file
inFile  = laspy.read("C:\\Users\\luckmanu\\Desktop\\2023-06-29_09-45-16_a0_006_100pct_height_world_clip2Traj_ground_heb_norm.laz")
binSize = 0.25

# Get the point ext
points  = inFile.points
ext     = [np.floor(np.min(inFile.xyz,axis=0)/5)*5,np.ceil(np.max(inFile.xyz,axis=0)/5)*5]

# Get bins
xBins   = np.linspace(start=ext[0][0],stop=ext[1][0],num=int((abs(ext[0][0])+abs(ext[1][0]))/binSize))
yBins   = np.linspace(start=ext[0][1],stop=ext[1][1],num=int((abs(ext[0][1])+abs(ext[1][1]))/binSize))
zBins   = np.linspace(start=ext[0][2],stop=ext[1][2],num=int((abs(ext[0][2])+abs(ext[1][2]))/binSize))

# Prepare matrix
mat     = np.zeros((len(xBins)-1,len(yBins)-1,len(zBins)-1))

# Fill matrix
for ix,x in enumerate(xBins):
    if ix != 0:

        for iy,y in enumerate(yBins):
            if iy != 0:
                
                for iz,z in enumerate(zBins):
                    if iz != 0:
                        pass

