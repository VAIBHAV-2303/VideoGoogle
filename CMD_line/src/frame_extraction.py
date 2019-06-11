import cv2
import numpy as np

v = cv2.VideoCapture('../bbt.mkv')

count = 0
prev = np.zeros([720, 1280], np.uint8)
while v.isOpened():
    ret,frame = v.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if np.sum((frame-prev)*(frame-prev)) > 7e7:
    	cv2.imwrite("../Frames/frame%d.jpg" % count, cv2.resize(frame, (0, 0), fx = 0.3, fy = 0.3))
    	count = count + 1
    	print(count, 'frames generated')

    prev = frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

v.release()
cv2.destroyAllWindows() 