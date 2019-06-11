import cv2
import numpy as np
import glob
import pickle

NUM_OF_CLUSTERS = 1000

# List of all frames in video(after preprocessing)
list_of_frames = glob.glob("../Frames/*")

# Pickling the list of frames
with open('list_of_frames.pkl', 'wb') as f:
	pickle.dump(list_of_frames, f)

# Extracintg sift features for all the images
sift = cv2.xfeatures2d.SIFT_create()
frames_desc = []

for i in list_of_frames:
	print(i)
	I = cv2.imread(i)
	temp_kp, temp_desc = sift.detectAndCompute(I, None)
	frames_desc.append(temp_desc)

print('xxxxxxxxxxxxxxxxx Feature Extraction Completed xxxxxxxxxxxxxxxxxxxx')

just_features = []
for i in frames_desc:
	for j in i:
		just_features.append(j)

just_features = np.array(just_features)
print(just_features.shape)

# define criteria and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
ret, labels, centers = cv2.kmeans(just_features, NUM_OF_CLUSTERS, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)

print('xxxxxxxxxxxxxxx Clustering completed xxxxxxxxxxxxxxxxxxxxxxx')

# Pickling the just_features centers
with open('centers.pkl', 'wb') as f:
	pickle.dump(centers, f)

# Generating the inverted index structure
inverted = np.zeros([len(list_of_frames), NUM_OF_CLUSTERS])

total = 0
for i in range(len(frames_desc)):
	for j in range(np.shape(frames_desc[i])[0]):
		inverted[i][labels[total]] += 1
		total += 1

# Normalizing the matrix and getting the final TF term
row_sums = inverted.sum(axis = 1)
inverted = inverted/row_sums[:, np.newaxis]

# Multiplying with the IDF term
for i in range(NUM_OF_CLUSTERS):
	inverted[:, i] *= np.log(len(list_of_frames)/np.count_nonzero(inverted[:, i]))

print('xxxxxxxxxxxxxxxxx Inverted Matrix Generated xxxxxxxxxxxxxxxxxxxxxx')

with open('inverted.pkl', 'wb') as f:
	pickle.dump(inverted, f)
	