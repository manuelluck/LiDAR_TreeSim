import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as St
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.switch_backend('tkagg')


# %% Functions:
def plotPolyMats(polyMat, suptitle=''):
    _, _, p = np.shape(polyMat)

    f, ax = plt.subplots(p, 1, sharey=True, sharex=True)
    f.set_figheight(12)
    f.set_figwidth(4)
    plt.suptitle(suptitle, fontsize=10)
    for i in range(p):
        ax[i].imshow(polyMat[:, :, i])
        ax[i].set_title(f'Polynomial Coefficient: {i}', fontsize=8)
        ax[i].axis('off')


def getPolyCoef(mat, polyOrder=3, removeNaN=False, hl=0, hu=-1):
    def polyfitPixel(pixel, zStep=0.25, polyOrder=3, removeNaN=False):
        steps = np.array([s * zStep for s in range(len(pixel))])
        if removeNaN:
            steps = steps[~np.isnan([float(nan) for nan in pixel])]
            pixel = pixel[~np.isnan([float(nan) for nan in pixel])]
            if len(pixel) > 2:
                return np.polyfit(steps, pixel, polyOrder)
            else:
                return [np.nan for _ in range(polyOrder+1)]
        else:
            return np.polyfit(steps, pixel, polyOrder)

    polyMat = np.zeros((np.shape(mat)[0], np.shape(mat)[1], polyOrder + 1))
    for i in range(np.shape(mat)[0]):
        for j in range(np.shape(mat)[1]):
            polyMat[i, j, :] = polyfitPixel(mat[i, j, hl:hu], polyOrder=polyOrder, removeNaN=removeNaN)
    return polyMat


def plotPolyLines(mat,n=50,polyOrder=3,suptitle='',zSteps=0.1):
    f, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    f.set_figheight(4)
    f.set_figwidth(6)
    plt.suptitle(suptitle, fontsize=10)
    x = [i*zSteps for i in range(np.shape(mat)[2])]

    for i in range(n):
        ix,iy = [int(np.floor(np.random.rand(1)*np.shape(mat)[0])),
                 int(np.floor(np.random.rand(1)*np.shape(mat)[1]))]

        ax[0].plot(mat[ix,iy,:],x,'--', linewidth=0.3)
        ax[0].set_title(f'Raw Data', fontsize=8)
        #ax[0].axis('off')

        p = np.poly1d(np.polyfit(x,mat[ix,iy,:],polyOrder))
        ax[1].plot(p(x),x,'--', linewidth=0.3)
        ax[1].set_title(f'Polynomial Fit', fontsize=8)
        #ax[1].axis('off')


def reshape_2D_2_1D(arr):
    m, n = np.shape(arr)
    return np.reshape(arr, m * n)


def prepareData(pointCloudFile, referenceDataFile,
                binSize=0.1,
                polyOrder=3,
                extend={'x': [-10, 10],
                        'y': [-10, 10],
                        'z': [-0.2, 5]}):
    var = {'pcFile': pointCloudFile,
           'refFile': referenceDataFile,
           'binSize': binSize,
           'polyOrder': polyOrder,
           'rasterExtend': extend}
    ref = dict()

    # Load the .laz file
    inFile = laspy.read(var['pcFile'])

    # Get the point ext
    points = inFile.points
    points = points[(points.x < var['rasterExtend']['x'][1]) * (points.x > var['rasterExtend']['x'][0])]
    points = points[(points.y < var['rasterExtend']['y'][1]) * (points.y > var['rasterExtend']['y'][0])]
    points = points[(points.z < var['rasterExtend']['z'][1]) * (points.z > var['rasterExtend']['z'][0])]

    var['pcExtend']= {'x':[min(points.x), max(points.x)],
                      'y':[min(points.y), max(points.y)],
                      'z':[min(points.z), max(points.z)]}

    # Get bins
    xBins = np.linspace(start=np.floor(var['pcExtend']['x'][0]), stop=np.ceil(var['pcExtend']['x'][1]),
                        num=int((abs(np.floor(var['pcExtend']['x'][0])) +
                                 abs(np.ceil(var['pcExtend']['x'][1]))) /
                                var['binSize']))
    yBins = np.linspace(start=np.floor(var['pcExtend']['y'][0]), stop=np.ceil(var['pcExtend']['y'][1]),
                        num=int((abs(np.floor(var['pcExtend']['y'][0])) +
                                 abs(np.ceil(var['pcExtend']['y'][1]))) /
                                var['binSize']))
    zBins = np.linspace(start=np.floor(var['pcExtend']['z'][0]), stop=np.ceil(var['pcExtend']['z'][1]),
                        num=int((abs(np.floor(var['pcExtend']['z'][0])) +
                                 abs(np.ceil(var['pcExtend']['z'][1]))) /
                                var['binSize']))

    xShape = len(xBins) - 1
    yShape = len(yBins) - 1
    zShape = len(zBins) - 1
    mShape = [xShape, yShape, zShape]

    # Prepare matrix
    dat = {'std x': np.zeros(mShape) * np.nan,
           'std y': np.zeros(mShape) * np.nan,
           'std z': np.zeros(mShape) * np.nan,
           'std xyz': np.zeros(mShape) * np.nan,
           'count': np.zeros(mShape) * np.nan,
           'PC1_std': np.zeros(mShape) * np.nan,
           'PC2_std': np.zeros(mShape) * np.nan,
           'PC3_std': np.zeros(mShape) * np.nan}

    # Fill matrix
    for ix, x in enumerate(xBins):
        if ix != 0:
            xPoints = points[(points.x > xBins[ix - 1]) * (points.x < xBins[ix])]
            for iy, y in enumerate(yBins):
                if iy != 0:
                    yPoints = xPoints[(xPoints.y > yBins[iy - 1]) * (xPoints.y < yBins[iy])]
                    for iz, z in enumerate(zBins):
                        if iz != 0:
                            zPoints = yPoints[(yPoints.z > zBins[iz - 1]) * (yPoints.z < zBins[iz])]
                            if len(zPoints.x) >= 1:
                                dat['std x'][ix - 1, iy - 1, iz - 1] = np.nanstd(zPoints.x)
                                dat['std y'][ix - 1, iy - 1, iz - 1] = np.nanstd(zPoints.y)
                                dat['std z'][ix - 1, iy - 1, iz - 1] = np.nanstd(zPoints.z)
                                dat['std xyz'][ix - 1, iy - 1, iz - 1] = (dat['std x'][ix - 1, iy - 1, iz - 1] +
                                                                          dat['std y'][ix - 1, iy - 1, iz - 1] +
                                                                          dat['std z'][ix - 1, iy - 1, iz - 1])
                                if len(zPoints.x) >= 3:
                                    pca = PCA(n_components=3)
                                    pcPoints = pca.fit_transform(np.vstack([zPoints.x, zPoints.y, zPoints.z]).T)
                                    dat['PC1_std'][ix - 1, iy - 1, iz - 1] = np.nanstd(pcPoints[:, 0])
                                    dat['PC2_std'][ix - 1, iy - 1, iz - 1] = np.nanstd(pcPoints[:, 1])
                                    dat['PC3_std'][ix - 1, iy - 1, iz - 1] = np.nanstd(pcPoints[:, 2])

                            dat['count'][ix - 1, iy - 1, iz - 1] = len(zPoints.z)

    dat['countNormSum'] = np.dstack([dat['count'][:, :, i] / np.nanmax(dat['count'], axis=2)
                                     for i in range(np.shape(dat['count'])[2])])

    dat['groundLevel']  = St.mode(reshape_2D_2_1D(np.argmax(dat['count'][:, :, :], axis=2))).mode
    dat['polyOrder']    = polyOrder
    dat['binSize']      = binSize

    return dat, var


def plotImgDatOnGroundLevel(dat):
    f, ax = plt.subplots(1, len([key for key in dat.keys() if len(np.shape(dat[key])) > 2]), sharey=True, sharex=True)
    f.set_figheight(4)
    f.set_figwidth(len(dat.keys()) * 3)
    plt.suptitle('Voxel Point Distribution', fontsize=10)
    for i, key in enumerate(dat.keys()):
        if len(np.shape(dat[key])) > 2:
            ax[i].imshow(dat[key][:, :, dat['groundlevel']])
            ax[i].set_title(f'{key}', fontsize=8)
            ax[i].axis('off')

# %% Code
[dat,var] = prepareData(pointCloudFile="C:\\Users\\luckmanu\\Desktop\\"
                           "2023-06-29_09-45-16_a0_006_100pct_height_world_clip2Traj_ground_heb_norm.laz",
                        referenceDataFile='H:\\Simulation\\Project_003\\Blender\\plot_000\\dw.csv',
                        binSize=0.25,
                        polyOrder=3,
                        extend={'x': [-20, 20],
                                'y': [-20, 20],
                                'z': [-0.2, 5]}
                        )

plotImgDatOnGroundLevel(dat)

plotPolyMats(getPolyCoef(dat['countNormSum'],
                         hl=dat['groundlevel'] - 1,
                         hu=dat['groundlevel'] + 8,
                         removeNaN=True,
                         polyOrder=dat['polyOrder']),
             suptitle=f'countNormSum Std in Voxels\n(Polynom Degree {dat["polyOrder"]})')

plotPolyLines(dat['countNormSum'],polyOrder=dat['polyOrder'],n=50)

#
# for i in range(np.shape(mat)[0]):
#     for j in range(np.shape(mat)[1]):
#         if i == 0 and j == 0:
#             #datArr = np.hstack((polyMat[i,j,:],polyMatSum[i,j,:],polyMatMax[i,j,:]))
#             datArr = polyMatSum[i,j,:]
#         else:
#             #datArr = np.vstack((datArr,np.hstack((polyMat[i,j,:],polyMatSum[i,j,:],polyMatMax[i,j,:]))))
#             datArr = np.vstack((datArr,polyMatSum[i,j,:]))
#
# dat_noNan = datArr[[not any(np.isnan(datArr[x,:])) for x in range(len(datArr[:,0]))],:]
#
# cMax = 6
# f,ax = plt.subplots(1,cMax-2)
# for c in range(2,cMax):
#     y_pred = KMeans(n_clusters=c+1).fit_predict(dat_noNan)
#
#     classes1d = np.zeros(np.shape(datArr)[0]) * np.nan
#     classes1d[[not any(np.isnan(datArr[x,:])) for x in range(len(datArr[:,0]))]] = y_pred
#     img = np.zeros((np.shape(mat)[0],np.shape(mat)[1]))
#     counter=0
#     for i in range(np.shape(mat)[0]):
#         for j in range(np.shape(mat)[1]):
#             img[i,j] = classes1d[counter]
#             counter+=1
#
#     ax[c-2].imshow(img)
#     ax[c-2].axis('off')
