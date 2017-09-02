import numpy as np
import pdb
import os
import pickle
import sys
import time


from sklearn.datasets import make_classification
from sklearn.mixture import GMM
from sklearn.preprocessing import normalize


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.

    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf

    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))



rootfolder = "../256_ObjectCategories"

imgcount = 1
errors = 0
start = time.time()
verify_write = False
check_existence = True

for class_dir in os.listdir(rootfolder):
	
	if not os.path.isdir(os.path.join(rootfolder,class_dir)):
		continue

	calculated_already = False
	siftvec_list = []
	for image_name in os.listdir(os.path.join(rootfolder,class_dir)):
		if check_existence and os.path.splitext(image_name)[-1] == ".fisher":
			calculated_already = True
			continue
		
		# Dont process other than .sift files
		if os.path.splitext(image_name)[-1] != ".sift":
			continue
		print "\rProcessing images: (",imgcount,"/30609), Class: ",class_dir, ", Time Elapsed: ",round(time.time()-start,2),"sec"
		# print class_dir, image_name
		imgcount +=1

		# Read the sift vector
		f = open(os.path.join(rootfolder,class_dir,image_name), "r")
		temp = pickle.load(f)

		if temp is not None:
			siftvec_list.append(normalize(temp))
	
	# Check if calculation needed.
	if calculated_already or class_dir[:3] == "200":
		print "*"*30, class_dir, "*"*30
		print "Already Calculated for this class, Skipping.."
		continue
		

	# All the sift vecs for a calss obtained
	class_siftvector = np.concatenate(np.array(siftvec_list))
	print "Obtained class sift vector:", class_dir, class_siftvector.shape


	# Fit GMM.
	K = 64
	N = len(class_siftvector)

	xx_tr = class_siftvector
	print "*"*30, class_dir, "*"*30
	print "Fitting the sift data in GMM.."
	gmm = GMM(n_components=K, covariance_type='diag')
	gmm.fit(xx_tr)


	# Calculate SIFT vectors for each image.
	for image_name in os.listdir(os.path.join(rootfolder,class_dir)):
		# Dont process other than ".sift" files.
		if os.path.splitext(image_name)[-1] != ".sift":
			continue
		print "*"*30, class_dir, "*"*30
		print "Calculating fisher vector for:",image_name

		# Read the sift vector.
		f = open(os.path.join(rootfolder,class_dir,image_name), "r")
		temp = pickle.load(f)

		# Calculate fisher vector.
		if temp is not None:
			xx_te = normalize(temp)
			fv = fisher_vector(xx_te, gmm)
			with open(os.path.join(rootfolder,class_dir,image_name+".fisher"), "w+") as f:
				pickle.dump(fv, f)
			print "Obtained fisher vector:", image_name, fv.shape, ", Time Elapsed: ",round(time.time()-start,2),"sec"

print "DONE!!!"


