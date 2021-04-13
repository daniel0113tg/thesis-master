# import the necessary packages
from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import cv2 as cv
import copy
import numpy as np
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5558".format(
	args["server_ip"]))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
print("paso")
rpiName =socket.gethostname()
print(rpiName)
vs = cv.VideoCapture(0)
#vs = VideoStream(src=0).start()

while True:
	ret, frame = vs.read()
	if frame is not None:
		frame = cv.resize(frame,(360, 360), interpolation = cv.INTER_AREA)
		frame = cv.flip(frame,1)
		##sender.send_image(rpiName, frame)
		key = cv.waitKey(1)
		if key == ord('q'):  # ESC
			break
		cv.imshow('ServerCODDO Demo',frame)
##sender.send_image(rpiName, np.array([]))
cv.destroyAllWindows()
