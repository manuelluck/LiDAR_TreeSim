import matplotlib.pyplot as plt

from matplotlib import cm

from skimage.feature import canny
from skimage.io import imread
from skimage.transform import probabilistic_hough_line

# Line finding using the Probabilistic Hough Transform
image = imread('C:\\Users\\luckmanu\\Pictures\\Screenshots\\Capture.PNG')
image = image[:,:,1]
edges = canny(image, 2, 15, 25)
lines = probabilistic_hough_line(edges, threshold=1, line_length=1,
                                 line_gap=5)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()