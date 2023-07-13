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

trj = np.loadtxt('C:\\Users\\luckmanu\\Desktop\\2023-06-29_09-45-16_a0_006_results_traj.txt',skiprows=1)

pointArr    = np.zeros((len(points['x']),5))
counter     = 0
trjT        = trj[:,0]

for i in range(len(points['x'])):
    pT  = points['gps_time'][i]
    idx = np.argmin(abs(trjT-pT))
    trjX,trjY,trjZ = trj[idx,1:4]

    d = ((points['x'][i]-trjX)**2+
          (points['y'][i]-trjY)**2+
          (points['z'][i]-trjZ)**2)**(1/2)

    pointArr[i,:] = [points['x'][i],points['y'][i],points['z'][i],points['intensity'][i]/d,d]

f,ax = plt.subplots(1,len(pointArr[0,:]))
for i in range(len(pointArr[0,:])):
    ax[i].hist(pointArr[:,i],bins=100)

plt.figure()
plt.scatter(pointArr[((pointArr[:,2]<0.5)*(pointArr[:,3]<5000)),0],
            pointArr[((pointArr[:,2]<0.5)*(pointArr[:,3]<5000)),1],
            c=pointArr[((pointArr[:,2]<0.5)*(pointArr[:,3]<5000)),3],s=0.1)
#
intensity = np.multiply(pointArr[:,4], pointArr[:,3])
p = np.poly1d(np.polyfit(pointArr[:,4],
                         np.multiply(pointArr[:,4],
                                     pointArr[:,3]), 1))
pfs = np.poly1d(np.polyfit(intensity,pointArr[:,4], 1))
#
plt.figure()
plt.plot(np.linspace(start=min(pointArr[pointArr[:,2]<0.5,4]),stop=max(pointArr[pointArr[:,2]<0.5,4]),num=100),
         pfs(np.linspace(start=min(pointArr[pointArr[:,2]<0.5,4]),stop=max(pointArr[pointArr[:,2]<0.5,4]),num=100)))
plt.scatter(pointArr[0:-1:1000,4],p(intensity[0:-1:1000]),s=0.1)
