import numpy as np
from scipy import signal
from scipy import misc
from scipy import ndimage
# from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy.linalg

# SIFT Detector 
#--------------

imagename = "building.jpg"
original = ndimage.imread(imagename, flatten=True)

# SIFT Parameters
s = 3
k = 2 ** (1.0 / s)
# threshold variable is the contrast threshold. Set to at least 1

# Standard deviations for Gaussian smoothing
kvec1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4)])
kvec2 = np.array([1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7)])
kvec3 = np.array([1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10)])
kvec4 = np.array([1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11), 1.6 * (k ** 12), 1.6 * (k ** 13)])
kvectotal = np.array([1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])

# Downsampling images
doubled = misc.imresize(original, 200, 'bilinear').astype(int)
normal = misc.imresize(doubled, 50, 'bilinear').astype(int)
halved = misc.imresize(normal, 50, 'bilinear').astype(int)
quartered = misc.imresize(halved, 50, 'bilinear').astype(int)

# Initialize Gaussian pyramids
pyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 6))
pyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 6))
pyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 6))
pyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 6))

print "Constructing pyramids..."