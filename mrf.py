import random
import cv2
from math import *
import numpy as np
import random
import matplotlib.pyplot as plt

B = 2.0
di = [-1,0,1,0]
dj = [0,1,0,-1]
t0 = 0.7
ITERS = 1000000
K = 2

count = np.zeros(K)
summ = np.zeros(K)
summ2 = np.zeros(K)
Energy = np.zeros(K)
total_energy = 0.0

image = cv2.imread('Lenna.png')
image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
arr = image.copy()

R,C = image.shape

def calc_energy(i):
	if(count[i] < 0.5):
		return 0.0
	return count[i]*np.log(sqrt(2*pi*calc_variance(i)))

def calc_mean(i):
	return summ[i]/count[i]

def calc_variance(i):
	Mean = calc_mean(i)
	return summ2[i]/count[i] - Mean*Mean

def update(i, j, New):
	global count
	global summ
	global summ2
	global Energy
	global total_energy
	global arr

	Old = arr[i][j]
	arr[i][j] = New
	for d in range(0,4):
		if(i+di[d]<0 or i+di[d]>=R or j+dj[d]<0 or j+dj[d]>=C):
			continue
		if(arr[i+di[d]][j+dj[d]] == Old):
			total_energy += 2*B
		if(arr[i+di[d]][j+dj[d]] == Old):
			total_energy -= 2*B
	total_energy -= (Energy[Old]+Energy[New])

	count[Old] -= 1
	summ[Old] -= image[i][j]
	summ2[Old] -= (1.0*image[i][j])*image[i][j]
	Energy[Old] = calc_energy(Old)

	count[New] += 1
	summ[New] += image[i][j]
	summ2[New] += (1.0*image[i][j])*image[i][j]
	Energy[New] = calc_energy(New)

	total_energy += (Energy[Old]+Energy[New])

def Temp(it):
	return (t0*(ITERS-it))/ITERS

print type(arr)

calc_intensities = []

# Total Energy

for i in range(0,R):
	for j in range(0,C):
		arr[i][j] = random.randint(0,K-1)
		count[arr[i][j]] += 1
		summ[arr[i][j]] += image[i][j]
		summ2[arr[i][j]] += (1.0*image[i][j])*image[i][j]
		if(i+1<R):
			total_energy += (2*(arr[i][j]!=arr[i+1][j])-1)*B
		if(j+1<C):
			total_energy += (2*(arr[i][j]!=arr[i][j+1])-1)*B


# Calculate Intensities and Total Energy

for i in range(0,K):
	calc_intensities.append(i*(255/(K-1)))
print calc_intensities

######################################

for i in range(0,K):
	Energy[i] = calc_energy(i)
	total_energy += Energy[i]
print total_energy

print_count = 0

for it in range(0,ITERS):
	Old_energy = total_energy
	while(True):
		i = random.randint(0,R-1)
		j = random.randint(0,C-1)
		Old = New = arr[i][j]
		while(New == Old):
			New = random.randint(0,K-1)
		update(i,j,New)
		if(total_energy < Old_energy):
			break
		if(random.random() < Temp(it)):
			break
		update(i,j,Old)
	print_count += 1
	if(print_count >= 1000):
		print_count = 0
		print it, total_energy

for i in range(0,R):
	for j in range(0,C):
		arr[i][j] = calc_intensities[arr[i][j]]

plt.imshow(arr, cmap=plt.cm.gray)
plt.show()
plt.savefig("result.jpg")
