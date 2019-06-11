from flask import Flask, render_template, request, redirect
import pickle
import cv2
import os

app = Flask(__name__)
SELECTED_FRAME = ''

with open('list_of_frames.pkl', 'rb') as f:
	list_of_frames = pickle.load(f)

for i in range(len(list_of_frames)):
	list_of_frames[i]='../static/'+list_of_frames[i].split('/')[2]

@app.route('/')
def index():
	return render_template('home.html', framesList = list_of_frames)

@app.route('/receiveFrame', methods=['POST', 'GET'])
def receiveFrame():
	global SELECTED_FRAME
	SELECTED_FRAME = request.get_json(force=True)
	return "Great"

@app.route('/receiveCoords', methods=['POST', 'GET'])
def receiveCoords():
	data = request.get_json(force=True)
	I = cv2.imread(SELECTED_FRAME[1:])
	query_image = I[data['Y1']:data['Y2'], data['X1']:data['X2']]
	cv2.imwrite('./query.jpg', query_image)
	os.system('python3 query.py ./query.jpg')
	return "Great"

@app.route('/selectRegion')
def selectRegion():
	return render_template('selectRegion.html', frame = SELECTED_FRAME)

@app.route('/results')
def results():
	return render_template('results.html')

app.run()
