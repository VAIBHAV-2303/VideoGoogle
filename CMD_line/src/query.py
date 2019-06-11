import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pickle

with open('inverted.pkl', 'rb') as f:
	inverted = pickle.load(f)

with open('centers.pkl', 'rb') as f:
	centers = pickle.load(f)	

with open('list_of_frames.pkl', 'rb') as f:
	list_of_frames = pickle.load(f)	

NUM_OF_CLUSTERS = centers.shape[0]

# Extracting sift features for the query image
I = cv2.imread(sys.argv[1])
sift = cv2.xfeatures2d.SIFT_create(250)
kp, desc = sift.detectAndCompute(I, None)

query_image_vector = np.zeros(NUM_OF_CLUSTERS)

for i in range(np.shape(desc)[0]):
	distances = np.linalg.norm(centers - desc[i], axis = 1)
	belonging_cluster = np.argmin(distances)
	query_image_vector[belonging_cluster] += 1

# TFIDF
query_image_vector = query_image_vector/np.sum(query_image_vector)
for i in range(NUM_OF_CLUSTERS):
	try:
		query_image_vector[i] *= np.log(len(list_of_frames)/np.count_nonzero(inverted[:, i]))
	except:
		pass

# Finding best image matches
distances = 1-((inverted*query_image_vector).sum(axis=1))/(np.linalg.norm(inverted, axis=1)*np.linalg.norm(query_image_vector))
sorted_matches = distances.argsort()

# Writing the best images in sorted order along with showing the matches
for i in range(10):
	I2 = cv2.imread(list_of_frames[sorted_matches[i]])
	kp2, desc2 = sift.detectAndCompute(I2, None)

	bf = cv2.BFMatcher()

	matches = bf.knnMatch(desc, desc2, k=2)

	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append([m])

	I3 = cv2.drawMatchesKnn(I, kp, I2, kp2, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv2.imwrite('../Output/'+str(i)+'.jpg', I3)
