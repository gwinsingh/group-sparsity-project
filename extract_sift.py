import os
import cv2
import pickle
import numpy as np
import sys
import time

rootfolder = "../256_ObjectCategories"


imgcount = 1
errors = 0
start = time.time()
verify_write = False
check_existence = True

for class_dir in os.listdir(rootfolder):
	if not os.path.isdir(os.path.join(rootfolder,class_dir)):
		continue
	for image_name in os.listdir(os.path.join(rootfolder,class_dir)):
		sift = cv2.SIFT()

		# Dont process .sift files
		if os.path.splitext(image_name)[-1] != ".jpg":
			continue
		print "\rProcessing images: (",imgcount,"/30609), Errors: ",errors, "Time Elapsed: ",round(time.time()-start,2),"sec"
		print class_dir, image_name
		imgcount +=1

		# Skip big images
		if os.path.getsize(os.path.join(rootfolder,class_dir,image_name)) > 2000000:
			continue

		# Skip if already done sift calculation
		if check_existence and os.path.exists(os.path.join(rootfolder,class_dir,image_name+".sift")):
			# f = open(os.path.join(rootfolder,class_dir,image_name+".sift"), "r")
			# temp = pickle.load(f)
			# if temp.shape[1] == 128:
			# 	print "Already calculated for this image - Skipping.."
			continue
		img = cv2.imread(os.path.join(rootfolder,class_dir,image_name),0)
		kp, desc = sift.detectAndCompute(img, None)
		with open(os.path.join(rootfolder,class_dir,image_name+".sift"), "w+") as f:
			pickle.dump(desc, f)
		if verify_write:
			with open(os.path.join(rootfolder,class_dir,image_name+".sift"), "r") as f:
				temp = pickle.load(f)
				if not np.array_equal(temp,desc):
					errors += 1

print "DONE!!!"
