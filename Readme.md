# VideoGoogle 

## Description

This is a python-openCV implementation of this [research paper](http://www.robots.ox.ac.uk/~vgg/publications/papers/sivic03.pdf) by Josef Sivic and Andrew Zisserman. The basic goal of this piece of code is to do some PreProcessing on a given video, and then given a query image, find out in what all frames of the video, does that query image occur in real time. To make the interface more user friendly, a Flask based webapp has been made above the Command Line method. Due to the limit on file sizes in github, the pickle files have been removed, please generate them yourself by running the code(it might take a while).

## Basic Algorithm and Working

### Frame Extraction(achieved by frame_extraction.py)

Firstly, key frames from the video are sampled. This can be as simple as taking frames which are more than a threshold different from the previous frame.

### Pre-processing(achieved by preproc.py)

Then, SIFT features from all the given frames are extracted and clustered using K-means clustering algorithm. These clusters constitute our visual vocabulary. Then a inverted index datastructure is constructed using this clusters as the building block of each frame, the data is stored in a similar way as TFIDF does so for text retrieval.

### Query(achieved by query.py)

Finally, once we have the query image, its sift features are computed and just as the frames were represented using a bag of visual words in the preprocessing step, the query image is also represented in a similar way. Then we just output the frames which have a smaller cosine distance with the query image. This gives us the relevance of frames with the query image in a sorted order.

## How To

* Run the webapp: 	
	```console
	bar@foo:~/VideoGoogle/WebApp$ python3 webapp.py
	```

## Built With

* [Python3](https://www.python.org/download/releases/3.0/)
* [OpenCV](https://docs.opencv.org/)
* [Flask](http://flask.pocoo.org/docs/1.0/)

## Author

* Vaibhav Garg
