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
points = points[points.z < 2.5]

# Get bins
xBins   = np.linspace(start=np.floor(min(points.x)),stop=np.ceil(max(points.x)),num=int((abs(np.floor(min(points.x)))+
                                                                                         abs(np.ceil(max(points.x))))/
                                                                                        binSize))
yBins   = np.linspace(start=np.floor(min(points.y)),stop=np.ceil(max(points.y)),num=int((abs(np.floor(min(points.y)))+
                                                                                         abs(np.ceil(max(points.y))))/
                                                                                        binSize))
zBins   = np.linspace(start=np.floor(min(points.z)),stop=np.ceil(max(points.z)),num=int((abs(np.floor(min(points.z)))+
                                                                                         abs(np.ceil(max(points.z))))/
                                                                                        binSize))

# Prepare matrix
dat = {'std x':np.zeros((len(xBins)-1,len(yBins)-1,len(zBins)-1)),
       'std y':np.zeros((len(xBins)-1,len(yBins)-1,len(zBins)-1)),
       'std z':np.zeros((len(xBins)-1,len(yBins)-1,len(zBins)-1)),
       'std xyz':np.zeros((len(xBins)-1,len(yBins)-1,len(zBins)-1)),
       'count':np.zeros((len(xBins)-1,len(yBins)-1,len(zBins)-1)),}

# Fill matrix
for ix,x in enumerate(xBins):
    if ix != 0:
        xPoints = points[(points.x > xBins[ix - 1]) * (points.x < xBins[ix])]
        for iy,y in enumerate(yBins):
            if iy != 0:
                yPoints = xPoints[(xPoints.y > yBins[iy - 1]) * (xPoints.y < yBins[iy])]
                for iz,z in enumerate(zBins):
                    if iz != 0:
                        zPoints = yPoints[(yPoints.z > zBins[iz - 1]) * (yPoints.z < zBins[iz])]
                        dat['std x'][ix-1,iy-1,iz-1]    = np.nanstd(zPoints.x)
                        dat['std y'][ix-1,iy-1,iz-1]    = np.nanstd(zPoints.y)
                        dat['std z'][ix-1,iy-1,iz-1]    = np.nanstd(zPoints.z)
                        dat['std xyz'][ix-1,iy-1,iz-1]  = (dat['std x'][ix-1,iy-1,iz-1]+
                                                           dat['std y'][ix-1,iy-1,iz-1]+
                                                           dat['std z'][ix-1,iy-1,iz-1])
                        dat['count'][ix-1,iy-1,iz-1]    = len(zPoints.z)


mat = dat['count']





#
def reshape_2D_2_1D(arr):
    m,n = np.shape(arr)
    return np.reshape(arr,m*n)

max_m = np.argmax(mat[:,:,:],axis=2)


plt.figure()
plt.imshow(max_m)

h = st.mode(reshape_2D_2_1D(max_m)).mode

#
f, ax = plt.subplots(1, len(dat.keys()), sharey=True, sharex=True)
f.set_figheight(4)
f.set_figwidth(len(dat.keys())*3)
plt.suptitle('Voxel Point Distribution', fontsize=10)
for i,key in enumerate(dat.keys()):
    ax[i].imshow(dat[key][:,:,h])
    ax[i].set_title(f'{key}', fontsize=8)
    ax[i].axis('off')

matBin = np.zeros(np.shape(mat))
matBin[mat>0] = 1
matSum = np.zeros(np.shape(mat))
for i in range(len(mat[0,0,:])):
    matSum[:,:,i] = np.divide(mat[:,:,i],np.sum(mat,axis=2))
matMax = np.zeros(np.shape(mat))
for i in range(len(mat[0,0,:])):
    matMax[:,:,i] = np.divide(mat[:,:,i],np.max(mat,axis=2))

plt.figure()
plt.imshow(mat[:,:,h]*(max_m == h),vmin=10,vmax=200)

def plotPolyMats(polyMat,suptitle=''):
    _,_,p = np.shape(polyMat)

    f, ax = plt.subplots(1, p, sharey=True, sharex=True)
    f.set_figheight(4)
    f.set_figwidth(15)
    plt.suptitle(suptitle,fontsize=10)
    for i in range(p):
        ax[i].imshow(polyMat[:,:,i])
        ax[i].set_title(f'Polynomial Coefficient: {i}',fontsize=8)
        ax[i].axis('off')






hl=h-2
hu=h+10
n = 3
polyMat     = np.zeros((np.shape(mat)[0],np.shape(mat)[1],n+1))
polyMatBin  = np.zeros((np.shape(mat)[0],np.shape(mat)[1],n+1))
polyMatSum  = np.zeros((np.shape(mat)[0],np.shape(mat)[1],n+1))
polyMatMax  = np.zeros((np.shape(mat)[0],np.shape(mat)[1],n+1))
polyMatStd  = np.zeros((np.shape(mat)[0],np.shape(mat)[1],n+1))
polyMatStdZ = np.zeros((np.shape(mat)[0],np.shape(mat)[1],n+1))

def polyfitPixel(pixel,zStep=0.25,polyOrder=3,removeNaN=True):
    steps = [s*zStep for s in range(len(pixel))]
    if removeNaN:
        steps   = steps[~np.isnan(pixel)]
        pixel   = pixel[~np.isnan(pixel)]
        return np.polyfit(steps,pixel)
    else:
        return np.polyfit(steps,pixel)


for i in range(np.shape(mat)[0]):
    for j in range(np.shape(mat)[1]):
        # polyMat[i, j, :]      = np.polyfit(x, mat[i, j, hl:hu], n)
        # polyMatBin[i,j,:]     = np.polyfit(x,matBin[i,j,hl:hu],n)
        # polyMatSum[i, j, :]   = np.polyfit(x, matSum[i, j, hl:hu], n)
        # polyMatMax[i, j, :]   = np.polyfit(x, matMax[i, j, hl:hu], n)
        polyMatStd[i, j, :]     = polyfitPixel(dat['std xyz'][i,j,:])

# plotPolyMats(polyMat,suptitle=f'Raw (Polynom Degree {n})')
# plotPolyMats(polyMatBin,suptitle=f'Binary (Polynom Degree {n})')
# plotPolyMats(polyMatMax,suptitle=f'Norm Max (Polynom Degree {n})')
# plotPolyMats(polyMatSum,suptitle=f'Norm Sum (Polynom Degree {n})')
plotPolyMats(polyMatStd,suptitle=f'Standard Deviation Z (Polynom Degree {n})')


plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for i in range(20):
    ix,iy = [int(np.floor(np.random.rand(1)*np.shape(matBin)[0])),
             int(np.floor(np.random.rand(1)*np.shape(matBin)[1]))]
    ax1.plot(matSum[ix,iy,hl:hu],x)

    p = np.poly1d(np.polyfit(x,matSum[ix,iy,hl:hu], n))
    ax2.plot(p(x),x,'--')

for i in range(np.shape(mat)[0]):
    for j in range(np.shape(mat)[1]):
        if i == 0 and j == 0:
            #datArr = np.hstack((polyMat[i,j,:],polyMatSum[i,j,:],polyMatMax[i,j,:]))
            datArr = polyMatSum[i,j,:]
        else:
            #datArr = np.vstack((datArr,np.hstack((polyMat[i,j,:],polyMatSum[i,j,:],polyMatMax[i,j,:]))))
            datArr = np.vstack((datArr,polyMatSum[i,j,:]))

dat_noNan = datArr[[not any(np.isnan(datArr[x,:])) for x in range(len(datArr[:,0]))],:]

cMax = 6
f,ax = plt.subplots(1,cMax-2)
for c in range(2,cMax):
    y_pred = KMeans(n_clusters=c+1).fit_predict(dat_noNan)

    classes1d = np.zeros(np.shape(datArr)[0]) * np.nan
    classes1d[[not any(np.isnan(datArr[x,:])) for x in range(len(datArr[:,0]))]] = y_pred
    img = np.zeros((np.shape(mat)[0],np.shape(mat)[1]))
    counter=0
    for i in range(np.shape(mat)[0]):
        for j in range(np.shape(mat)[1]):
            img[i,j] = classes1d[counter]
            counter+=1

    ax[c-2].imshow(img)
    ax[c-2].axis('off')