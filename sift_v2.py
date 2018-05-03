import cv2
from scipy import misc
import numpy as np


original = cv2.imread("building.jpg",0)

sigma = 0.707107
scale_factor = 2**(0.5)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

# img = cv2.imread('building.jpg')

# blur = cv2.blur(img,(5,5),sigma*(scale_factor))
# blur2 = cv2.blur(img,(5,5),sigma*(2*scale_factor))
# blur3 = cv2.blur(img,(5,5),sigma*(3*scale_factor))
# blur4 = cv2.blur(img,(5,5),sigma*(4*scale_factor))

# cv2.imshow('image',img)
# cv2.imshow('image2',blur)
# cv2.imshow('image3',blur2)
# cv2.imshow('image4',blur3)
# cv2.imshow('image5',blur4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


####################  Create Octaves and Scales  ####################

double = misc.imresize(arr=original, size=2.0)
normal = original
half = misc.imresize(arr=original, size=0.5)
quater = misc.imresize(arr=original, size=0.25)

def calculate_scales(image, sigma = 0.707107, scale_factor = 2**(0.5), n = 6):
	octave = []
	octave.append(image)
	for i in range(1, n):
		blurred = cv2.blur(image, (5,5), sigma*scale_factor*i)
		octave.append(blurred)
		# print "appending", blurred.shape
	octave = np.array(octave)
	# print "returning", octave.shape
	return octave

octaves = []
octaves.append(calculate_scales(double, sigma, scale_factor))
octaves.append(calculate_scales(normal, sigma, scale_factor))
octaves.append(calculate_scales(half, sigma, scale_factor))
octaves.append(calculate_scales(quater, sigma, scale_factor))
octaves = np.array(octaves)
# print octaves.shape, octaves[0].shape, octaves[0][0].shape

###################        Construct DoG      ########################


DoGs = []

for octave in octaves:
	temp = []
	for i in range(len(octave)-1):
		temp.append(octave[i+1] - octave[i])
	DoGs.append(temp)
DoGs = np.array(DoGs)

print DoGs.shape, DoGs[0][0].shape
print DoGs.shape, DoGs[1][0].shape
print DoGs.shape, DoGs[2][0].shape
print DoGs.shape, DoGs[3][0].shape



####################      Locate Maxima/Minima   ####################

# Initialize pyramids to store extrema locations
extremas = []
extremas.append(np.zeros((DoGs.shape[1]-2, double.shape[0], double.shape[1])))
extremas.append(np.zeros((DoGs.shape[1]-2, normal.shape[0], normal.shape[1])))
extremas.append(np.zeros((DoGs.shape[1]-2, half.shape[0], half.shape[1])))
extremas.append(np.zeros((DoGs.shape[1]-2, quater.shape[0], quater.shape[1])))
extremas = np.array(extremas)


def maxmin_neighbours(img, i, j):
	maxm, minm = -1, -1
	for p in range(-1,2):
		for q in range(-1,2):
			for r in range(-1,2):
				if p==i && q==j, 
				if maxm == -1:
					maxm = img[p][q]
				if minm == -1:
					minm = img[p][q]
				minm = min(img[p][q], minm)
				maxm = max(img[p][q], maxm)
	return (maxm, minm)


for dog in DoGs:
	print "S:",dog.shape
	for i in range(1,len(dog)-1):
		for p_i in range(30, dog[i].shape(0)-30):
			for p_j in range(30, dog[i].shape(1)-30):
				maxm, minm = maxmin_neighbours(dog[i], p_i, p_j)
		print dog[i].shape







